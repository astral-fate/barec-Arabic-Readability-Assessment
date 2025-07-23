# =====================================================================================
# 0. INSTALLATIONS
# =====================================================================================

import pandas as pd
import numpy as np
import os
import torch
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from arabert.preprocess import ArabertPreprocessor
from google.colab import drive
import zipfile

# =====================================================================================
# 1. SETUP AND GOOGLE DRIVE INTEGRATION
# =====================================================================================
print("Mounting Google Drive...")
drive.mount('/content/drive')
print("Google Drive mounted successfully.")

# =====================================================================================
# 2. CONFIGURATION
# =====================================================================================
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
DRIVE_MOUNT_PATH = "/content/drive/MyDrive/"
PROJECT_FOLDER = "BAREC_Competition"
BASE_DIR = os.path.join(DRIVE_MOUNT_PATH, PROJECT_FOLDER)
CHECKPOINT_DIR = os.path.join(BASE_DIR, "results_training_19_class")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# File Paths
BAREC_TRAIN_PATH = 'train.csv'
BAREC_DEV_PATH = 'dev.csv'
BLIND_TEST_PATH = 'blind_test_data.csv'
SUBMISSION_PATH = os.path.join(BASE_DIR, "submission_final_19_class.csv")
ZIPPED_SUBMISSION_PATH = os.path.join(BASE_DIR, "submission_final_19_class.zip")

NUM_LABELS = 19 # Set to the correct number of classes

# =====================================================================================
# 3. DATA LOADING AND PREPROCESSING
# =====================================================================================
arabert_preprocessor = ArabertPreprocessor(model_name=MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_training_and_validation_data():
    """Loads training and validation data using the correct columns."""
    print("--- Loading BAREC Data from CSV files ---")
    try:
        train_df = pd.read_csv(BAREC_TRAIN_PATH)
        val_df = pd.read_csv(BAREC_DEV_PATH)

        # --- CORRECTED: Use 'Sentence' (singular) to match your CSV header ---
        train_df = train_df[['Sentence', 'Readability_Level_19']].rename(columns={'Sentence': 'text', 'Readability_Level_19': 'label'})
        val_df = val_df[['Sentence', 'Readability_Level_19']].rename(columns={'Sentence': 'text', 'Readability_Level_19': 'label'})

        # Drop any rows with missing data
        train_df.dropna(subset=['text', 'label'], inplace=True)
        val_df.dropna(subset=['label', 'text'], inplace=True)

        # Ensure labels are integers
        train_df['label'] = train_df['label'].astype(int)
        val_df['label'] = val_df['label'].astype(int)

        # Convert labels from 1-19 to 0-18 for the model
        train_df['label'] = train_df['label'] - 1
        val_df['label'] = val_df['label'] - 1
        
        print(f"Successfully loaded {len(train_df)} training and {len(val_df)} validation records.")
        
    except Exception as e:
        print(f"❗️ ERROR loading data: {e}")
        return None, None
    return train_df, val_df

# --- Execute Data Loading and Preprocessing ---
train_df, val_df = load_training_and_validation_data()
if train_df is None:
    exit()

print("\n--- Preprocessing Text (this may take a moment) ---")
train_df['text'] = train_df['text'].apply(arabert_preprocessor.preprocess)
val_df['text'] = val_df['text'].apply(arabert_preprocessor.preprocess)
print("Text preprocessing finished.")

# =====================================================================================
# 4. DATASET CLASS
# =====================================================================================
class ReadabilityDataset(TorchDataset):
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=256)
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.encodings.get('input_ids', []))

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"qwk": cohen_kappa_score(p.label_ids, preds, weights='quadratic')}

# =====================================================================================
# 5. MODEL TRAINING
# =====================================================================================
print("\n===== ✨ INITIALIZING MODEL AND TRAINER =====\n")

# Initialize the standard model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# Create Datasets
train_dataset = ReadabilityDataset(train_df['text'].tolist(), train_df['label'].tolist())
val_dataset = ReadabilityDataset(val_df['text'].tolist(), val_df['label'].tolist())

# Define Training Arguments
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=6,
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
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Start or Resume Training
print("Starting or resuming training...")
# The Trainer will automatically find and resume from the latest checkpoint if it exists.
# trainer.train(resume_from_checkpoint=True) 
trainer.train()
print("✅ Training finished.")

# =====================================================================================
# 6. FINAL PREDICTION AND SUBMISSION
# =====================================================================================
print("\n===== 🏆 FINAL PREDICTION AND SUBMISSION =====\n")
try:
    test_df_docs = pd.read_csv(BLIND_TEST_PATH)
    # Explode documents into sentences, keeping track of the original doc_id
    test_df = test_df_docs.assign(text=test_df_docs['Sentence'].str.split('\n')).explode('text').reset_index()
    test_df.rename(columns={'ID': 'doc_id', 'text': 'sentence_text'}, inplace=True)
    test_df.dropna(subset=['sentence_text'], inplace=True)
    
    print("Preprocessing blind test text...")
    test_df['processed_text'] = test_df['sentence_text'].apply(arabert_preprocessor.preprocess)
    
    print("Generating predictions on the test set...")
    test_dataset = ReadabilityDataset(test_df['processed_text'].tolist())
    predictions = trainer.predict(test_dataset)
    # The model predicts 0-18, so we add 1 to get back to the 1-19 scale
    test_df['prediction'] = np.argmax(predictions.predictions, axis=1) + 1

    print("Aggregating sentence predictions to document-level using MAX rule...")
    doc_level_preds = test_df.groupby('doc_id')['prediction'].max()
    
    submission_df = pd.DataFrame({'Document ID': doc_level_preds.index, 'Prediction': doc_level_preds.values})

    print(f"Saving prediction file to: {SUBMISSION_PATH}")
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print(f"\nCompressing '{os.path.basename(SUBMISSION_PATH)}' into '{os.path.basename(ZIPPED_SUBMISSION_PATH)}'...")
    with zipfile.ZipFile(ZIPPED_SUBMISSION_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(SUBMISSION_PATH, arcname=os.path.basename(SUBMISSION_PATH))

    print(f"Submission file '{os.path.basename(ZIPPED_SUBMISSION_PATH)}' created successfully.")
except Exception as e:
    print(f"An error occurred during final prediction: {e}")

print("\n--- Script Finished ---")
