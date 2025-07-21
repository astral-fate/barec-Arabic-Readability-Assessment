# =====================================================================================
# 0. INSTALLATIONS
# =====================================================================================
# This will install all necessary libraries quietly.
# !pip install transformers[torch] datasets pandas scikit-learn arabert accelerate -q

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
from google.colab import drive

# =====================================================================================
# 1. SETUP AND GOOGLE DRIVE INTEGRATION
# =====================================================================================
print("Mounting Google Drive...")
drive.mount('/content/drive')
print("Google Drive mounted successfully.")

# =====================================================================================
# 2. CONFIGURATION
# =====================================================================================
# --- Model & Preprocessing ---
# MODEL_NAME = "aubmindlab/bert-large-arabertv2"
MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
arabert_preprocessor = ArabertPreprocessor(model_name=MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- Training & Classification ---
RANDOM_STATE = 42
NUM_LABELS = 19

# --- File Paths in Google Drive ---
DRIVE_MOUNT_PATH = "/content/drive/MyDrive/"
PROJECT_FOLDER = "BAREC_Competition"
BASE_DIR = os.path.join(DRIVE_MOUNT_PATH, PROJECT_FOLDER)
os.makedirs(BASE_DIR, exist_ok=True)

# Data paths
BAREC_TRAIN_PATH = 'train.csv'
BAREC_DEV_PATH = 'dev.csv'
BAREC_TEST_PATH = 'test.csv'

# Submission file paths
SUBMISSION_FILE_NAME = "prediction"
SUBMISSION_PATH = os.path.join(BASE_DIR, SUBMISSION_FILE_NAME)
ZIPPED_SUBMISSION_PATH = os.path.join(BASE_DIR, f"{SUBMISSION_FILE_NAME}.zip")

# =====================================================================================
# 3. REVISED: DATA LOADING AND PREPARATION
# =====================================================================================
def load_and_prepare_data():
    """
    Loads data from document-level CSVs and transforms it into a
    sentence-level format for training.
    """
    print("--- Loading BAREC Data from Document-Level CSV files ---")
    try:
        train_docs = pd.read_csv(BAREC_TRAIN_PATH)
        val_docs = pd.read_csv(BAREC_DEV_PATH)
        test_docs = pd.read_csv(BAREC_TEST_PATH)

        # --- Process Training Data ---
        train_sentences = []
        for _, row in train_docs.iterrows():
            sentences = str(row['Sentences']).split('\n')
            level = row['Readability_Level_19']
            for s in sentences:
                if s.strip():  # Ensure sentence is not empty
                    train_sentences.append({'text': s, 'label': level})
        train_df = pd.DataFrame(train_sentences)

        # --- Process Validation Data ---
        val_sentences = []
        for _, row in val_docs.iterrows():
            sentences = str(row['Sentences']).split('\n')
            level = row['Readability_Level_19']
            for s in sentences:
                if s.strip():
                    val_sentences.append({'text': s, 'label': level})
        val_df = pd.DataFrame(val_sentences)

        # --- Process Test Data ---
        test_sentences = []
        for _, row in test_docs.iterrows():
            sentences = str(row['Sentences']).split('\n')
            doc_id = row['Document']
            for s in sentences:
                if s.strip():
                    test_sentences.append({'text': s, 'doc_id': doc_id})
        test_df = pd.DataFrame(test_sentences)

        # Clean and prepare data
        train_df.dropna(subset=['text', 'label'], inplace=True)
        train_df['label'] = train_df['label'].astype(int)
        train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    except FileNotFoundError as e:
        print(f"❗️ ERROR: {e}. Make sure train.csv, dev.csv, and test.csv are in your project folder.")
        return None, None, None
    except KeyError as e:
        print(f"❗️ ERROR: A required column was not found: {e}. This means the CSV format is unexpected.")
        return None, None, None

    print(f"Created {len(train_df)} sentences from BAREC train.csv.")
    print(f"Created {len(val_df)} sentences from BAREC dev.csv (for validation).")
    print(f"Created {len(test_df)} sentences from BAREC test.csv.")

    return train_df, val_df, test_df

# Execute loading function
train_df, val_df, test_df = load_and_prepare_data()
if train_df is None:
    exit()

# Preprocess text data
print("\n--- Preprocessing text for all datasets ---")
train_df['text'] = train_df['text'].apply(arabert_preprocessor.preprocess)
val_df['text'] = val_df['text'].apply(arabert_preprocessor.preprocess)
test_df['text'] = test_df['text'].apply(arabert_preprocessor.preprocess)
print("Text preprocessing complete.")


# =====================================================================================
# 4. MODEL, DATASET, AND TRAINER FOR CLASSIFICATION
# =====================================================================================
class ReadabilityModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(ReadabilityModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        transformer_output_dim = self.transformer.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(transformer_output_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
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
    logits, labels = p.predictions, p.label_ids
    preds = np.argmax(logits, axis=1)
    return {"qwk": cohen_kappa_score(labels, preds, weights='quadratic')}

# =====================================================================================
# 5. TRAINING AND EVALUATION
# =====================================================================================
print("\n===== PREPARING FOR CLASSIFICATION TRAINING RUN =====\n")

train_dataset = ReadabilityDataset(train_df['text'].tolist(), train_df['label'].tolist())
val_dataset = ReadabilityDataset(val_df['text'].tolist(), val_df['label'].tolist())
test_dataset = ReadabilityDataset(test_df['text'].tolist())

model = ReadabilityModel(MODEL_NAME, num_labels=NUM_LABELS)
OUTPUT_DIR = os.path.join(BASE_DIR, "results_classification_model")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=10,
    per_device_train_batch_size=16, per_device_eval_batch_size=32,
    learning_rate=2e-5, warmup_ratio=0.1, weight_decay=0.01,
    logging_steps=100, eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="qwk", greater_is_better=True,
    save_total_limit=1, fp16=torch.cuda.is_available(), report_to="none"
)

trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset,
    eval_dataset=val_dataset, compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("Starting model training for classification...")
trainer.train()
print("Training finished.")

# =====================================================================================
# 6. PREDICTION AND SUBMISSION FILE GENERATION
# =====================================================================================
print("\n===== GENERATING CLASSIFICATION PREDICTIONS ON THE TEST SET =====\n")
predictions = trainer.predict(test_dataset)
test_logits = predictions.predictions

sentence_level_preds = np.argmax(test_logits, axis=1) + 1
test_df['prediction'] = sentence_level_preds

print("Aggregating sentence predictions to document-level using MAX rule...")
doc_level_preds = test_df.groupby('doc_id')['prediction'].max()

submission_df = pd.DataFrame({
    'Document ID': doc_level_preds.index,
    'Prediction': doc_level_preds.values
})

print(f"Saving prediction file to: {SUBMISSION_PATH}")
submission_df.to_csv(SUBMISSION_PATH, index=False)

print(f"\n===== CREATING SUBMISSION ZIP FILE =====\n")
print(f"Compressing '{SUBMISSION_FILE_NAME}' into '{ZIPPED_SUBMISSION_PATH}'...")
with zipfile.ZipFile(ZIPPED_SUBMISSION_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(SUBMISSION_PATH, arcname=SUBMISSION_FILE_NAME)

print(f"Submission file '{os.path.basename(ZIPPED_SUBMISSION_PATH)}' created successfully.")
print("\n--- Script Finished ---")
