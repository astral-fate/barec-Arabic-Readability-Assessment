
# =====================================================================================
# 0. INSTALLATIONS
# =====================================================================================
# This will install all necessary libraries quietly.
# Make sure to run this in your terminal if you don't have these installed:
# pip install transformers[torch] datasets pandas scikit-learn arabert accelerate
# =====================================================================================

import pandas as pd
import numpy as np
import os
import gc
import torch
import torch.nn as nn
import zipfile
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, EarlyStoppingCallback
from arabert.preprocess import ArabertPreprocessor

# =====================================================================================
# 1. CONFIGURATION
# =====================================================================================
# --- Model & Preprocessing ---
# You can switch between different models by uncommenting the desired one.
# MODEL_NAME = "aubmindlab/bert-large-arabertv2"
MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
arabert_preprocessor = ArabertPreprocessor(model_name=MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- Training & Classification ---
RANDOM_STATE = 42
NUM_LABELS = 19

# --- File Paths (Local Environment) ---
# Assumes the script is in the same directory as the data files.
BASE_DIR = "." 

# Data paths
BAREC_TRAIN_PATH = 'train.csv'
BAREC_DEV_PATH = 'dev.csv'
# The script will generate predictions for the blind test set.
BLIND_TEST_PATH = 'blind_test.csv' 

# Submission file paths
SUBMISSION_FILE_NAME = "prediction"
SUBMISSION_PATH = os.path.join(BASE_DIR, SUBMISSION_FILE_NAME)
ZIPPED_SUBMISSION_PATH = os.path.join(BASE_DIR, f"{SUBMISSION_FILE_NAME}.zip")

# =====================================================================================
# 2. DATA LOADING
# =====================================================================================
def load_data():
    """
    Loads data from train, dev, and blind_test CSV files.
    """
    print("--- Loading BAREC Data from local CSV files ---")
    try:
        # Load the datasets, which are one sentence per row
        train_df = pd.read_csv(BAREC_TRAIN_PATH)
        val_df = pd.read_csv(BAREC_DEV_PATH)
        blind_test_df = pd.read_csv(BLIND_TEST_PATH)

        # --- Process Training and Validation Data ---
        # Select the text and label columns, and rename them for consistency
        train_df = train_df[['Sentence', 'Readability_Level_19']].rename(columns={'Sentence': 'text', 'Readability_Level_19': 'label'})
        val_df = val_df[['Sentence', 'Readability_Level_19']].rename(columns={'Sentence': 'text', 'Readability_Level_19': 'label'})
        
        # --- Process Blind Test Data ---
        # The 'Document' column is the identifier needed for aggregation
        blind_test_df = blind_test_df[['Sentence', 'Document']].rename(columns={'Sentence': 'text', 'Document': 'doc_id'})
        
        # Preprocess text
        print("Preprocessing text data...")
        train_df['text'] = train_df['text'].apply(arabert_preprocessor.preprocess)
        val_df['text'] = val_df['text'].apply(arabert_preprocessor.preprocess)
        blind_test_df['text'] = blind_test_df['text'].apply(arabert_preprocessor.preprocess)
        
        # Clean and prepare training data
        train_df.dropna(subset=['text', 'label'], inplace=True)
        train_df['label'] = train_df['label'].astype(int)
        train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    except FileNotFoundError as e:
        print(f"❗️ ERROR: {e}. Make sure train.csv, dev.csv, and blind_test.csv are in the same folder as the script.")
        return None, None, None
    except KeyError as e:
        print(f"❗️ ERROR: A required column was not found: {e}. Please check the CSV file format.")
        return None, None, None

    print(f"Loaded {len(train_df)} sentences from {BAREC_TRAIN_PATH}.")
    print(f"Loaded {len(val_df)} sentences from {BAREC_DEV_PATH} (for validation).")
    print(f"Loaded {len(blind_test_df)} sentences from {BLIND_TEST_PATH} (for prediction).")

    return train_df, val_df, blind_test_df

# Execute loading function
train_df, val_df, blind_test_df = load_data()
if train_df is None:
    exit()

# =====================================================================================
# 3. SIMPLIFIED MODEL, DATASET, AND TRAINER
# =====================================================================================
class ReadabilityModel(nn.Module):
    """
    A simplified model that uses only the transformer output for classification.
    """
    def __init__(self, model_name, num_labels):
        super(ReadabilityModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        transformer_output_dim = self.transformer.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(transformer_output_dim, 512),
            nn.BatchNorm1d(512), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token's embedding for classification
        cls_embedding = transformer_outputs.last_hidden_state[:, 0, :]
        logits = self.head(cls_embedding)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return (loss, logits) if loss is not None else logits

class ReadabilityDataset(TorchDataset):
    """
    Torch Dataset that tokenizes text. It no longer handles extra numerical features.
    """
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=256)
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            # Labels are 1-19, so we subtract 1 to make them 0-18 for CrossEntropyLoss
            item['labels'] = torch.tensor(self.labels[idx] - 1, dtype=torch.long)
        return item
        
    def __len__(self):
        return len(self.encodings['input_ids'])

def compute_metrics(p):
    """
    Computes the Quadratic Weighted Kappa score for evaluation.
    """
    logits, labels = p.predictions, p.label_ids
    preds = np.argmax(logits, axis=1)
    return {"qwk": cohen_kappa_score(labels, preds, weights='quadratic')}

# =====================================================================================
# 4. TRAINING AND EVALUATION
# =====================================================================================
print("\n===== PREPARING FOR CLASSIFICATION TRAINING RUN =====\n")

train_dataset = ReadabilityDataset(train_df['text'].tolist(), train_df['label'].tolist())
val_dataset = ReadabilityDataset(val_df['text'].tolist(), val_df['label'].tolist())
blind_test_dataset = ReadabilityDataset(blind_test_df['text'].tolist())

model = ReadabilityModel(MODEL_NAME, num_labels=NUM_LABELS)
OUTPUT_DIR = os.path.join(BASE_DIR, "results_classification_model")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, 
    num_train_epochs=10,
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=32,
    learning_rate=2e-5, 
    warmup_ratio=0.1, 
    weight_decay=0.01,
    logging_steps=100, 
    eval_strategy="epoch", 
    save_strategy="epoch",
    load_best_model_at_end=True, 
    metric_for_best_model="qwk", 
    greater_is_better=True,
    save_total_limit=1, 
    fp16=torch.cuda.is_available(), 
    report_to="none"
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset,
    eval_dataset=val_dataset, 
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("Starting model training for classification...")
trainer.train()
print("Training finished.")

# =====================================================================================
# 5. PREDICTION AND SUBMISSION FILE GENERATION
# =====================================================================================
print("\n===== GENERATING CLASSIFICATION PREDICTIONS ON THE BLIND TEST SET =====\n")
predictions = trainer.predict(blind_test_dataset)
test_logits = predictions.predictions

# Get the predicted class (0-18) and add 1 to map it back to the original label (1-19)
sentence_level_preds = np.argmax(test_logits, axis=1) + 1
blind_test_df['prediction'] = sentence_level_preds

print("Aggregating sentence predictions to document-level using MAX rule...")
# Group by document ID and take the max prediction as the document-level readability
doc_level_preds = blind_test_df.groupby('doc_id')['prediction'].max()

submission_df = pd.DataFrame({
    'Document ID': doc_level_preds.index,
    'Prediction': doc_level_preds.values
})

print(f"Saving prediction file to: {SUBMISSION_PATH}")
submission_df.to_csv(SUBMISSION_PATH, index=False)

print(f"\n===== CREATING SUBMISSION ZIP FILE =====\n")
print(f"Compressing '{SUBMISSION_FILE_NAME}' into '{ZIPPED_SUBMISSION_PATH}'...")
with zipfile.ZipFile(ZIPPED_SUBMISSION_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # The arcname parameter ensures the file inside the zip doesn't have the directory structure.
    zipf.write(SUBMISSION_PATH, arcname=SUBMISSION_FILE_NAME)

print(f"Submission file '{os.path.basename(ZIPPED_SUBMISSION_PATH)}' created successfully.")
print("\n--- Script Finished ---")
