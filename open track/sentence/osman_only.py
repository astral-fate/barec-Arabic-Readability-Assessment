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
from sklearn.metrics import cohen_kappa_score
import os
import warnings

warnings.filterwarnings("ignore")

# =====================================================================================
# 2. CONFIGURATION & FILE PATHS
# =====================================================================================
# --- File Paths ---
BAREC_TRAIN_PATH = 'train.csv'
EXTERNAL_DATA_PATH = 'Annotated_Paper_Dataset.csv'
BLIND_TEST_PATH = 'sentnse_blind_test.csv'
SUBMISSION_PATH = 'submission.csv'
MODEL_OUTPUT_DIR = './results'

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
        output_dir=MODEL_OUTPUT_DIR,
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
        metric_for_best_model="qwk",
        greater_is_better=True,
    )

    # --- Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("\n--- 🚀 Starting or Resuming Model Training ---")
    
    checkpoint_path = None
    if os.path.exists(MODEL_OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(MODEL_OUTPUT_DIR) if d.startswith('checkpoint-')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
            checkpoint_path = os.path.join(MODEL_OUTPUT_DIR, latest_checkpoint)
            print(f"✅ Resuming training from checkpoint: {checkpoint_path}")

    trainer.train(resume_from_checkpoint=checkpoint_path)
    
    print("--- ✅ Training Finished Successfully! ---")

    # =====================================================================================
    # START OF CORRECTIONS
    # =====================================================================================
    
    # --- Prediction on the Blind Test Set ---
    # The trainer automatically loads the best model at the end of training
    print(f"\n--- 🏆 Predicting on the Blind Test Set: {BLIND_TEST_PATH} ---")
    try:
       
        test_df = pd.read_csv(BLIND_TEST_PATH, sep=',')
        
        # Use 'Sentence' column instead of 'Text'
        if 'Sentence' not in test_df.columns:
            raise KeyError("The test file must contain a 'Sentence' column.")
        test_texts = test_df['Sentence'].tolist()

        test_dataset = ReadabilityDataset(
            texts=test_texts,
            labels=[0] * len(test_texts), # Dummy labels
            tokenizer=tokenizer
        )

        predictions = trainer.predict(test_dataset)
        predicted_labels = predictions.predictions.argmax(axis=1)

        # --- Create Submission File ---
        # Use 'ID' column (uppercase) from test file for submission 'id'
        if 'ID' not in test_df.columns:
            raise KeyError("The test file must contain an 'ID' column.")
        submission_df = pd.DataFrame({'id': test_df['ID'], 'label': predicted_labels})
        submission_df.to_csv(SUBMISSION_PATH, index=False)

        print(f"--- 🎉 Submission file '{SUBMISSION_PATH}' created successfully! ---")
        
    except FileNotFoundError:
        print(f"❗️ Error: The blind test file was not found at {BLIND_TEST_PATH}")
    except KeyError as e:
        print(f"❗️ Error: A required column was not found in the test file. {e}")
    except Exception as e:
        print(f"❗️ An unexpected error occurred during prediction: {e}")

    # =====================================================================================
    # END OF CORRECTIONS
    # =====================================================================================

else:
    print("\n--- ❌ Could not proceed due to data loading errors. Please check file paths and formats. ---")
