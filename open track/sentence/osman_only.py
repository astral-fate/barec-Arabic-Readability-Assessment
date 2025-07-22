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
from sklearn.metrics import cohen_kappa_score # 🟢 ADDED: For QWK calculation
import os
import warnings

warnings.filterwarnings("ignore")

# =====================================================================================
# 2. CONFIGURATION & FILE PATHS
# =====================================================================================
# --- File Paths ---
BAREC_TRAIN_PATH = 'train.csv'
EXTERNAL_DATA_PATH = 'Annotated_Paper_Dataset.csv'
BLIND_TEST_PATH = 'test.csv'
SUBMISSION_PATH = 'submission.csv'
MODEL_OUTPUT_DIR = './results' # 🟢 ADDED: Directory to save model checkpoints

# --- Model Configuration ---
MODEL_NAME = 'asafaya/bert-base-arabic'
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

        ext_df['label'] = ext_df['Fine-grained'].str.replace('G', '', regex=False).astype(int)
        ext_df.rename(columns={'Text': 'text'}, inplace=True)
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
        train_df = pd.read_csv(barec_path, sep=',')
        train_df.rename(columns={
            'Sentence': 'text',
            'Readability_Level_19': 'label'
        }, inplace=True)
        
        train_df = train_df[['text', 'label']]
        print(f"Original BAREC training size: {len(train_df)}")

        if external_df is not None:
            train_df = pd.concat([train_df, external_df], ignore_index=True)
            print(f"✅ New combined training size: {len(train_df)}")
            
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
# 5. METRICS, TRAINING, AND PREDICTION
# =====================================================================================

# 🟢 ADDED: Function to compute QWK metric
def compute_metrics(eval_pred):
    """
    Computes Quadratic Weighted Kappa (QWK) for the predictions.
    """
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=-1)
    
    qwk = cohen_kappa_score(labels, preds, weights='quadratic')
    
    return {
        'qwk': qwk
    }

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
        output_dir=MODEL_OUTPUT_DIR, # Use the defined output directory
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="epoch", # Corrected from eval_strategy
        save_strategy="epoch",       # Corrected from save_strategy
        load_best_model_at_end=True,
        metric_for_best_model="qwk",   # 🟢 CHANGED: Use QWK to find the best model
        greater_is_better=True,      # 🟢 CHANGED: Higher QWK is better
    )

    # --- Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics, # 🟢 ADDED: Pass the metrics function
    )

    # --- Start or Resume Training ---
    # The Trainer automatically looks for the latest checkpoint in `output_dir`.
    # To resume, simply run the script again after it has saved a checkpoint.
    # The `resume_from_checkpoint=True` argument makes this explicit.
    # If a checkpoint exists, it will resume training from there.
    # If not, it will start from epoch 0.
    
    print("\n--- 🚀 Starting or Resuming Model Training ---")
    
    # To explicitly resume from the *last* saved checkpoint:
    # trainer.train(resume_from_checkpoint=True)

    # To resume from a *specific* best checkpoint if you know its path:
    checkpoint_path = None
    if os.path.exists(MODEL_OUTPUT_DIR):
        # Logic to find the best checkpoint folder (e.g., based on trainer_state.json)
        # For simplicity, we'll just check if any checkpoint exists
        checkpoints = [d for d in os.listdir(MODEL_OUTPUT_DIR) if d.startswith('checkpoint-')]
        if checkpoints:
            # A simple approach is to find the checkpoint with the highest number
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
            checkpoint_path = os.path.join(MODEL_OUTPUT_DIR, latest_checkpoint)
            print(f"✅ Resuming training from checkpoint: {checkpoint_path}")

    # If checkpoint_path is None, training starts from scratch.
    # Otherwise, it resumes from the specified checkpoint.
    trainer.train(resume_from_checkpoint=checkpoint_path)
    
    print("--- ✅ Training Finished Successfully! ---")

    # --- Prediction on the Blind Test Set ---
    # The trainer automatically loads the best model at the end of training
    print(f"\n--- 🏆 Predicting on the Blind Test Set: {BLIND_TEST_PATH} ---")
    test_df = pd.read_csv(BLIND_TEST_PATH, sep='\t')
    test_texts = test_df['Text'].tolist()

    test_dataset = ReadabilityDataset(
        texts=test_texts,
        labels=[0] * len(test_texts), # Dummy labels
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
