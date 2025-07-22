# =====================================================================================
# 1. INSTALLATIONS & IMPORTS
# =====================================================================================
# !pip install transformers[torch] pandas scikit-learn accelerate -q

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
import os
import warnings

warnings.filterwarnings("ignore")

# =====================================================================================
# 2. CONFIGURATION & FILE PATHS
# =====================================================================================
# --- File Paths ---
# Path to the official training data
BAREC_TRAIN_PATH = 'train.csv'
# Path to your external, annotated data
EXTERNAL_DATA_PATH = 'Annotated_Paper_Dataset.csv'
# 🔴 UPDATE THIS when the official blind test file is released
BLIND_TEST_PATH = 'test.csv' 
# Name of the final submission file
SUBMISSION_PATH = 'submission.csv'

# --- Model Configuration ---
MODEL_NAME = 'asafaya/bert-base-arabic'
# The BAREC task has 19 levels, but labels are 1-19.
# We need 20 labels so that label 19 maps to index 19.
NUM_LABELS = 20

# =====================================================================================
# 3. DATA LOADING AND PREPROCESSING FUNCTIONS
# =====================================================================================

def load_and_map_external_data(file_path):
    """
    Loads the external dataset and maps its 'Fine-grained' grade
    levels to the BAREC 1-19 numerical scale.
    """
    print(f"\n--- Loading and Mapping External Data from: {file_path} ---")
    try:
        ext_df = pd.read_csv(file_path)

        if 'Fine-grained' not in ext_df.columns or 'Text' not in ext_df.columns:
            print(f"❗️ Error: Required columns 'Fine-grained' or 'Text' not found in {file_path}.")
            return None

        # Create the 'label' by extracting the number from the 'Fine-grained' column (e.g., 'G1' -> 1)
        ext_df['label'] = ext_df['Fine-grained'].str.replace('G', '', regex=False).astype(int)
        
        # Rename 'Text' column to 'text' for consistency
        ext_df.rename(columns={'Text': 'text'}, inplace=True)

        # Drop invalid rows and keep only the necessary columns
        ext_df.dropna(subset=['text', 'label'], inplace=True)
        ext_df = ext_df[['text', 'label']]

        print(f"✅ Successfully loaded and mapped {len(ext_df)} sentences.")
        return ext_df

    except Exception as e:
        print(f"❗️ An error occurred while processing {file_path}: {e}")
        return None

def load_all_training_data(barec_path, external_df=None):
    """
    Loads the original BAREC training data, renames columns, and merges
    it with the processed external data.
    """
    print("\n--- Loading Original BAREC Training Data ---")
    try:
        # Load the original BAREC training data
        train_df = pd.read_csv(barec_path, sep=',') 
        
        # 🔴 **FIX for KeyError**: Rename the correct columns to 'text' and 'label'
        train_df.rename(columns={
            'Sentence': 'text',
            'Readability_Level_19': 'label'
        }, inplace=True)
        
        train_df = train_df[['text', 'label']]
        print(f"Original BAREC training size: {len(train_df)}")

        # Merge with external data if available
        if external_df is not None:
            train_df = pd.concat([train_df, external_df], ignore_index=True)
            print(f"✅ New combined training size: {len(train_df)}")
            
        # Split the combined data into training and validation sets
        train_df, val_df = train_test_split(
            train_df, test_size=0.1, random_state=42, stratify=train_df['label']
        )
        return train_df, val_df

    except Exception as e:
        print(f"❗️ An error occurred while loading BAREC data from {barec_path}: {e}")
        return None, None

# =====================================================================================
# 4. PYTORCH DATASET CLASS
# =====================================================================================
class ReadabilityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# =====================================================================================
# 5. MODEL TRAINING AND PREDICTION
# =====================================================================================

# --- Load and Prepare Data ---
external_data_df = load_and_map_external_data(EXTERNAL_DATA_PATH)
train_df, val_df = load_all_training_data(BAREC_TRAIN_PATH, external_data_df)

if train_df is not None:
    # --- Initialize Tokenizer and Datasets ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = ReadabilityDataset(
        texts=train_df.text.to_list(),
        labels=train_df.label.to_list(),
        tokenizer=tokenizer
    )
    val_dataset = ReadabilityDataset(
        texts=val_df.text.to_list(),
        labels=val_df.label.to_list(),
        tokenizer=tokenizer
    )

    # --- Initialize Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True
    )

    # --- Define Training Arguments ---
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    # --- Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # --- Start Training ---
    print("\n--- 🚀 Starting Model Training ---")
    trainer.train()
    print("--- ✅ Training Finished Successfully! ---")

    # --- Prediction on the Blind Test Set ---
    print(f"\n--- 🏆 Predicting on the Blind Test Set: {BLIND_TEST_PATH} ---")
    test_df = pd.read_csv(BLIND_TEST_PATH, sep='\t')
    test_texts = test_df['Text'].tolist() # Use 'Text' column from blind test

    test_dataset = ReadabilityDataset(
        texts=test_texts,
        labels=[0] * len(test_texts), # Dummy labels for prediction
        tokenizer=tokenizer
    )

    predictions = trainer.predict(test_dataset)
    predicted_labels = predictions.predictions.argmax(axis=1)

    # --- Create Submission File ---
    submission_df = pd.DataFrame({'id': test_df['id'], 'label': predicted_labels})
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print(f"--- 🎉 Submission file '{SUBMISSION_PATH}' created successfully! ---")

else:
    print("\n--- ❌ Could not proceed due to data loading errors. Please check file paths and formats. ---")
