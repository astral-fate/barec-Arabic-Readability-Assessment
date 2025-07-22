# sent_test_final.py
# This script trains the model using the preprocessed data and generates the submission file.

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
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = 'processed_training_data.csv'
# 🔴 UPDATE THIS for the final competition file. Using the development test set for now.
BLIND_TEST_PATH = 'sentnse_blind_test.csv' 
SUBMISSION_PATH = 'submission.csv'
MODEL_NAME = 'aubmindlab/bert-base-arabertv2'
NUM_LABELS = 20 # For labels 1-19

# --- DATASET CLASS (No changes needed here) ---
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
            text, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- MAIN SCRIPT EXECUTION ---
if __name__ == '__main__':
    print("--- Loading Preprocessed Data ---")
    try:
        full_df = pd.read_csv(PROCESSED_DATA_PATH)
        print(f"✅ Loaded {len(full_df)} rows from {PROCESSED_DATA_PATH}")
        
        train_df, val_df = train_test_split(
            full_df, test_size=0.1, random_state=42, stratify=full_df['label']
        )
        print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")
    except Exception as e:
        print(f"❗️ ERROR: Could not load preprocessed data. Please run preprocess.py first. Details: {e}")
        exit()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = ReadabilityDataset(train_df.text.to_list(), train_df.label.to_list(), tokenizer)
    val_dataset = ReadabilityDataset(val_df.text.to_list(), val_df.label.to_list(), tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
    )

   
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Adjust based on your GPU memory
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="epoch", # This was the line with the error
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset
    )
    
    print("\n--- 🚀 Starting Model Training ---")
    trainer.train()
    print("--- ✅ Training Finished ---")

    print(f"\n--- 🏆 Predicting on Blind Test Set: {BLIND_TEST_PATH} ---")
    # The official blind test might be a CSV, so we'll use read_csv
    test_df = pd.read_csv(BLIND_TEST_PATH, sep='\t')
    test_texts = test_df['Text'].tolist()
    test_dataset = ReadabilityDataset(test_texts, [0] * len(test_texts), tokenizer)

    predictions = trainer.predict(test_dataset)
    predicted_labels = predictions.predictions.argmax(axis=1)

    submission_df = pd.DataFrame({'id': test_df['id'], 'label': predicted_labels})
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"--- 🎉 Submission file '{SUBMISSION_PATH}' created successfully! ---")
