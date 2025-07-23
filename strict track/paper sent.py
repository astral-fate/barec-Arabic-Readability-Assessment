# =====================================================================================
# 0. INSTALLATIONS
# =====================================================================================
# Installing necessary libraries, including camel-tools for D3Tok preprocessing
# !pip install transformers[torch] pandas numpy scikit-learn accelerate arabert


# =====================================================================================
# 1. IMPORTS AND SETUP
# =====================================================================================
import pandas as pd
import numpy as np
import os
import torch
import zipfile
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
# MODIFICATION: Import Camel-tools for D3Tok preprocessing as per the paper [cite: 232]
from camel_tools.disambig.mle import MLEDisambiguator
from google.colab import drive

print("Mounting Google Drive...")
drive.mount('/content/drive')
print("Google Drive mounted successfully.")

# =====================================================================================
# 2. CONFIGURATION
# =====================================================================================
# MODIFICATION: Changed MODEL_NAME to the one you specified.
# NOTE: The paper's 84% QWK result was with 'aubmindlab/bert-base-arabertv2'.
# Using a different model may yield different results.
MODEL_NAME = "aubmindlab/bert-base-arabertv2"

# MODIFICATION: As we are training for a regression task, NUM_LABELS is 1[cite: 245].
NUM_LABELS = 1
# The target classes (1-19) for calculating metrics after rounding predictions.
TARGET_CLASSES = 19

# --- Paths (assuming files are in the root of the project folder) ---
DRIVE_MOUNT_PATH = "/content/drive/MyDrive/"
PROJECT_FOLDER = "BAREC_Competition"
BASE_DIR = os.path.join(DRIVE_MOUNT_PATH, PROJECT_FOLDER)
CHECKPOINT_DIR = os.path.join(BASE_DIR, f"results_training_regression_{MODEL_NAME.split('/')[-1]}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BAREC_TRAIN_PATH = os.path.join(BASE_DIR, 'train.csv')
BAREC_DEV_PATH = os.path.join(BASE_DIR, 'dev.csv')
BLIND_TEST_PATH = os.path.join(BASE_DIR, 'blind_test_data.csv')
SUBMISSION_PATH = os.path.join(BASE_DIR, "submission_regression_final.csv")
ZIPPED_SUBMISSION_PATH = os.path.join(BASE_DIR, "submission_regression_final.zip")


# =====================================================================================
# 3. DATA LOADING AND PREPROCESSING
# =====================================================================================
# MODIFICATION: Initialize CamelTools MLE Disambiguator for D3Tok preprocessing [cite: 232]
print("--- Initializing CAMeL Tools Disambiguator ---")
mle_disambiguator = MLEDisambiguator.pretrained()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_d3tok(text, disambiguator):
    """
    Preprocesses text into the D3Tok format as described in the BAREC paper[cite: 234].
    This involves segmenting words into their base and clitic forms.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    # Disambiguate the sentence to get morphological analyses
    disambiguated_words = disambiguator.disambiguate(text.split())
    # Extract the D3 tokenization for each word
    d3tok_forms = [word.d3tok for word in disambiguated_words]
    return ' '.join(d3tok_forms)

def load_training_and_validation_data():
    """Loads training and validation data and prepares it for the regression task."""
    print("--- Loading BAREC Data from CSV files ---")
    try:
        train_df = pd.read_csv(BAREC_TRAIN_PATH)
        val_df = pd.read_csv(BAREC_DEV_PATH)

        train_df = train_df[['Sentence', 'Readability_Level_19']].rename(columns={'Sentence': 'text', 'Readability_Level_19': 'label'})
        val_df = val_df[['Sentence', 'Readability_Level_19']].rename(columns={'Sentence': 'text', 'Readability_Level_19': 'label'})

        train_df.dropna(subset=['text', 'label'], inplace=True)
        val_df.dropna(subset=['label', 'text'], inplace=True)

        # Convert labels from 1-19 to 0-18 for model processing
        train_df['label'] = train_df['label'].astype(int) - 1
        val_df['label'] = val_df['label'].astype(int) - 1

        # MODIFICATION: Convert labels to float for regression task
        train_df['label'] = train_df['label'].astype(float)
        val_df['label'] = val_df['label'].astype(float)

        print(f"Successfully loaded {len(train_df)} training and {len(val_df)} validation records.")
        return train_df, val_df
    except Exception as e:
        print(f"❗️ ERROR loading data: {e}")
        return None, None

# --- Execute Data Loading and Preprocessing ---
train_df, val_df = load_training_and_validation_data()
if train_df is not None:
    print("\n--- Preprocessing Text to D3Tok format (this may take a moment) ---")
    train_df['text'] = train_df['text'].apply(lambda x: preprocess_d3tok(x, mle_disambiguator))
    val_df['text'] = val_df['text'].apply(lambda x: preprocess_d3tok(x, mle_disambiguator))
    print("Text preprocessing finished.")
else:
    exit()

# =====================================================================================
# 4. DATASET AND METRICS
# =====================================================================================
class ReadabilityDataset(TorchDataset):
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=256)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not a None:
            # MODIFICATION: Ensure labels are floats for regression loss
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.encodings.get('input_ids', []))

def compute_metrics(p):
    """
    Computes metrics for the regression task.
    Predictions are rounded to the nearest integer and clipped to the valid label range[cite: 248].
    """
    # MODIFICATION: For regression, predictions are a 1D array of floats.
    preds = p.predictions.flatten()
    # Round predictions to the nearest integer level
    rounded_preds = np.round(preds)
    # Clip predictions to be within the valid range of [0, 18]
    clipped_preds = np.clip(rounded_preds, 0, TARGET_CLASSES - 1).astype(int)

    labels = p.label_ids.astype(int)
    
    # Calculate Quadratic Weighted Kappa
    qwk = cohen_kappa_score(labels, clipped_preds, weights='quadratic')
    return {"qwk": qwk}

# =====================================================================================
# 5. MODEL TRAINING
# =====================================================================================
print("\n===== ✨ INITIALIZING REGRESSION MODEL AND TRAINER =====\n")

# Initialize the model for sequence classification with 1 label for regression
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# Create Datasets
train_dataset = ReadabilityDataset(train_df['text'].tolist(), train_df['label'].tolist())
val_dataset = ReadabilityDataset(val_df['text'].tolist(), val_df['label'].tolist())

# MODIFICATION: Update Training Arguments to match the paper's hyperparameters [cite: 250]
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=6,
    per_device_train_batch_size=64, # Paper uses batch size of 64
    per_device_eval_batch_size=64,  # Paper uses batch size of 64
    learning_rate=5e-5,             # Paper uses learning rate of 5e-5
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="qwk",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none" # Disables integration with external services like W&B
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

# Start Training
print("Starting training...")
trainer.train()
print("✅ Training finished.")

# =====================================================================================
# 6. FINAL PREDICTION AND SUBMISSION
# =====================================================================================
print("\n===== 🏆 FINAL PREDICTION AND SUBMISSION =====\n")
try:
    test_df_docs = pd.read_csv(BLIND_TEST_PATH)
    test_df = test_df_docs.assign(text=test_df_docs['Sentence'].str.split('\n')).explode('text').reset_index()
    test_df.rename(columns={'ID': 'doc_id', 'text': 'sentence_text'}, inplace=True)
    test_df.dropna(subset=['sentence_text'], inplace=True)
    
    print("Preprocessing blind test text to D3Tok format...")
    test_df['processed_text'] = test_df['sentence_text'].apply(lambda x: preprocess_d3tok(x, mle_disambiguator))
    
    print("Generating predictions on the test set...")
    test_dataset = ReadabilityDataset(test_df['processed_text'].tolist())
    predictions = trainer.predict(test_dataset)
    
    # MODIFICATION: Process regression output
    raw_preds = predictions.predictions.flatten()
    rounded_preds = np.round(raw_preds)
    clipped_preds = np.clip(rounded_preds, 0, TARGET_CLASSES - 1)
    
    # The model predicts 0-18, so add 1 to get back to the 1-19 scale
    test_df['prediction'] = clipped_preds + 1

    print("Aggregating sentence predictions to Sentence-level using MAX rule...")
    doc_level_preds = test_df.groupby('doc_id')['prediction'].max().astype(int)
    
    submission_df = pd.DataFrame({'Sentence ID': doc_level_preds.index, 'Prediction': doc_level_preds.values})

    print(f"Saving prediction file to: {SUBMISSION_PATH}")
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print(f"\nCompressing '{os.path.basename(SUBMISSION_PATH)}' into '{os.path.basename(ZIPPED_SUBMISSION_PATH)}'...")
    with zipfile.ZipFile(ZIPPED_SUBMISSION_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(SUBMISSION_PATH, arcname=os.path.basename(SUBMISSION_PATH))

    print(f"Submission file '{os.path.basename(ZIPPED_SUBMISSION_PATH)}' created successfully.")
except Exception as e:
    print(f"An error occurred during final prediction: {e}")

print("\n--- Script Finished ---")
