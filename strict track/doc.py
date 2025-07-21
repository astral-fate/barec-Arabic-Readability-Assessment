# =====================================================================================
# 0. SETUP FOR GOOGLE COLAB
# =====================================================================================
# This section mounts your Google Drive to make your files accessible.
# You will be prompted to authorize Colab to access your Google Drive.
from google.colab import drive
drive.mount('/content/drive')

# --- Install Necessary Libraries ---
# This will install all required libraries quietly in your Colab environment.


# =====================================================================================
# 1. CONFIGURATION
# =====================================================================================
import pandas as pd
import numpy as np
import os
import gc
import torch
import torch.nn as nn
import zipfile
import sys
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, EarlyStoppingCallback
from arabert.preprocess import ArabertPreprocessor

# --- Model & Preprocessing ---
MODEL_NAME = "aubmindlab/bert-large-arabertv2"
arabert_preprocessor = ArabertPreprocessor(model_name=MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- Training & Classification ---
RANDOM_STATE = 42
NUM_LABELS = 19

# --- File Paths (Google Colab Environment) ---
BASE_DIR = "/content/drive/MyDrive/BAREC_Competition/"
os.makedirs(BASE_DIR, exist_ok=True)

# Data paths (CORRECTED FILE AND COLUMN NAMES)
BAREC_TRAIN_PATH = os.path.join(BASE_DIR, 'train.csv')
BAREC_DEV_PATH = os.path.join(BASE_DIR, 'dev.csv')
# CORRECTED: Changed to the actual uploaded filename
BLIND_TEST_PATH = os.path.join(BASE_DIR, 'blind_test_data.csv')

# Submission and model output paths
SUBMISSION_FILE_NAME = "prediction.csv"
SUBMISSION_PATH = os.path.join(BASE_DIR, SUBMISSION_FILE_NAME)
ZIPPED_SUBMISSION_PATH = os.path.join(BASE_DIR, "prediction.zip")
OUTPUT_DIR = os.path.join(BASE_DIR, "results_classification_model")

# =====================================================================================
# 2. DATA LOADING
# =====================================================================================
def load_data():
    """
    Loads data from train, dev, and blind_test CSV files from Google Drive.
    """
    print(f"--- Loading BAREC Data from: {BASE_DIR} ---")
    try:
        train_df = pd.read_csv(BAREC_TRAIN_PATH)
        val_df = pd.read_csv(BAREC_DEV_PATH)
        blind_test_df = pd.read_csv(BLIND_TEST_PATH)

        # --- Process Training and Validation Data ---
        # CORRECTED: Changed 'Sentence' to 'Sentences'
        train_df = train_df[['Sentences', 'Readability_Level_19']].rename(columns={'Sentences': 'text', 'Readability_Level_19': 'label'})
        val_df = val_df[['Sentences', 'Readability_Level_19']].rename(columns={'Sentences': 'text', 'Readability_Level_19': 'label'})

        # --- Process Blind Test Data ---
        # CORRECTED: Changed 'Sentence' to 'Sentences'
        blind_test_df = blind_test_df[['Sentences', 'Document']].rename(columns={'Sentences': 'text', 'Document': 'doc_id'})

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
        print(f"❗️ ERROR: {e}. Make sure train.csv, dev.csv, and blind_test_data.csv are in your '{BASE_DIR}' folder.")
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

# Stop execution if data loading failed to prevent further errors
if train_df is None:
    print("\nExecution stopped due to data loading errors.")
    sys.exit()

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
        cls_embedding = transformer_outputs.last_hidden_state[:, 0, :]
        logits = self.head(cls_embedding)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return (loss, logits) if loss is not None else logits

class ReadabilityDataset(TorchDataset):
    """
    Torch Dataset that tokenizes text.
    """
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=256)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
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

gc.collect()
torch.cuda.empty_cache()

train_dataset = ReadabilityDataset(train_df['text'].tolist(), train_df['label'].tolist())
val_dataset = ReadabilityDataset(val_df['text'].tolist(), val_df['label'].tolist())
blind_test_dataset = ReadabilityDataset(blind_test_df['text'].tolist())

model = ReadabilityModel(MODEL_NAME, num_labels=NUM_LABELS)

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

print(f"Starting model training... Checkpoints will be saved to: {OUTPUT_DIR}")
trainer.train()
print("Training finished.")

# =====================================================================================
# 5. PREDICTION AND SUBMISSION FILE GENERATION
# =====================================================================================
print("\n===== GENERATING CLASSIFICATION PREDICTIONS ON THE BLIND TEST SET =====\n")
predictions = trainer.predict(blind_test_dataset)
test_logits = predictions.predictions

sentence_level_preds = np.argmax(test_logits, axis=1) + 1
blind_test_df['prediction'] = sentence_level_preds

print("Aggregating sentence predictions to document-level using MAX rule...")
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
    zipf.write(SUBMISSION_PATH, arcname=os.path.basename(SUBMISSION_PATH))

print(f"Submission file '{os.path.basename(ZIPPED_SUBMISSION_PATH)}' created successfully in '{BASE_DIR}'.")
print("\n--- Script Finished ---")


