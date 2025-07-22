# sent_test_final.py
# This script correctly saves full checkpoints (including trainer_state.json)
# to Google Drive and can resume from them.

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
from sklearn.metrics import cohen_kappa_score
import warnings
import os
from google.colab import drive

warnings.filterwarnings("ignore")

# --- MOUNT GOOGLE DRIVE ---
drive.mount('/content/drive')

# --- CONFIGURATION ---
DRIVE_PATH = '/content/drive/MyDrive/ArabicReadabilityProject/'
PROCESSED_DATA_PATH = DRIVE_PATH + 'processed_training_data.csv'
BLIND_TEST_PATH = DRIVE_PATH + 'sentnse_blind_test.csv'
SUBMISSION_PATH = DRIVE_PATH + 'submission.csv'
RESULTS_PATH = DRIVE_PATH + 'results_new' # ✨ Using a new folder to ensure clean checkpoints
LOGS_PATH = DRIVE_PATH + 'logs_new'

# Create directories if they don't exist to prevent errors
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)


MODEL_NAME = 'aubmindlab/bert-base-arabertv2'
NUM_LABELS = 20

# --- DATASET CLASS ---
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

# --- METRIC FUNCTION ---
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=-1)
    qwk = cohen_kappa_score(labels, preds, weights='quadratic')
    return {'qwk': qwk}

# --- MAIN SCRIPT EXECUTION ---
if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("❗️ WARNING: GPU not found. Training will be slow.")
    else:
        print(f"✅ GPU found: {torch.cuda.get_device_name(0)}")

    print("\n--- Loading Preprocessed Data ---")
    try:
        full_df = pd.read_csv(PROCESSED_DATA_PATH)
        train_df, val_df = train_test_split(
            full_df, test_size=0.1, random_state=42, stratify=full_df['label']
        )
        print(f"✅ Data loaded. Training set: {len(train_df)}, Validation set: {len(val_df)}")
    except Exception as e:
        print(f"❗️ ERROR: Could not load preprocessed data. Details: {e}")
        exit()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = ReadabilityDataset(train_df.text.to_list(), train_df.label.to_list(), tokenizer)
    val_dataset = ReadabilityDataset(val_df.text.to_list(), val_df.label.to_list(), tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
    )

    # ✨ This setup correctly saves all necessary files, including trainer_state.json
    training_args = TrainingArguments(
        output_dir=RESULTS_PATH,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        fp16=True,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=LOGS_PATH,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch", # This ensures a full save at the end of each epoch
        load_best_model_at_end=True,
        metric_for_best_model="qwk",
        greater_is_better=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print("\n--- 🚀 Starting or Resuming Model Training ---")
    
    # ✨ This logic correctly resumes from a complete checkpoint if one exists.
    # It will start fresh otherwise, creating new, complete checkpoints.
    latest_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        # Find the latest checkpoint in the output directory
        checkpoints = [d for d in os.listdir(training_args.output_dir) if d.startswith('checkpoint-')]
        if checkpoints:
            latest_checkpoint_num = max([int(c.split('-')[1]) for c in checkpoints])
            latest_checkpoint = os.path.join(training_args.output_dir, f'checkpoint-{latest_checkpoint_num}')
    
    if latest_checkpoint:
        print(f"✅ Resuming training from the latest complete checkpoint: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print("✅ No complete checkpoint found. Starting training from scratch...")
        trainer.train()

    print("--- ✅ Training Finished ---")

    print(f"\n--- 🏆 Predicting on Blind Test Set: {BLIND_TEST_PATH} ---")
    try:
        test_df = pd.read_csv(BLIND_TEST_PATH, sep='\t')
        test_texts = test_df['Text'].tolist()
    except Exception as e:
        print(f"❗️ ERROR: Could not load the blind test file. Details: {e}")
        exit()

    test_dataset = ReadabilityDataset(test_texts, [0] * len(test_texts), tokenizer)

    predictions = trainer.predict(test_dataset)
    predicted_labels = predictions.predictions.argmax(axis=1)

    submission_df = pd.DataFrame({'id': test_df['id'], 'label': predicted_labels})
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"--- 🎉 Submission file '{SUBMISSION_PATH}' created successfully! ---")
