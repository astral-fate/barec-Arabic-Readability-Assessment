# =====================================================================================
# 0. INSTALLATIONS
# =====================================================================================
# This will install all necessary libraries quietly.
# !pip install transformers[torch] pandas scikit-learn arabert accelerate -q

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
# REMOVED: The 'load_dataset' function is no longer needed.
# from datasets import load_dataset


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
# MODIFICATION: Added a path for the local blind test file.
BLIND_TEST_PATH = 'blind_test_data.csv'
SAMER_CORPUS_PATH = os.path.join(BASE_DIR, 'samer_train.tsv')
SAMER_LEXICON_PATH = os.path.join(BASE_DIR, 'SAMER-Readability-Lexicon-v2.tsv')

# Submission file paths
SUBMISSION_FILE_NAME = "prediction"
SUBMISSION_PATH = os.path.join(BASE_DIR, SUBMISSION_FILE_NAME)
ZIPPED_SUBMISSION_PATH = os.path.join(BASE_DIR, f"{SUBMISSION_FILE_NAME}.zip")


# =====================================================================================
# 3. DATA LOADING
# =====================================================================================

def load_training_and_validation_data():
    """Loads and prepares training and validation data from local CSVs."""
    print("--- Loading BAREC Data from CSV files ---")
    try:
        train_df = pd.read_csv(BAREC_TRAIN_PATH)
        val_df = pd.read_csv(BAREC_DEV_PATH)
        train_df = train_df[['Sentences', 'Readability_Level_19']].rename(columns={'Sentences': 'text', 'Readability_Level_19': 'label'})
        val_df = val_df[['Sentences', 'Readability_Level_19']].rename(columns={'Sentences': 'text', 'Readability_Level_19': 'label'})
    except (FileNotFoundError, KeyError) as e:
        print(f"❗️ ERROR loading local CSVs: {e}")
        return None, None

    print(f"Loaded {len(train_df)} training documents and {len(val_df)} validation documents.")
    train_df = train_df.assign(text=train_df['text'].str.split('\n')).explode('text').reset_index(drop=True)
    val_df = val_df.assign(text=val_df['text'].str.split('\n')).explode('text').reset_index(drop=True)
    train_df.dropna(subset=['text'], inplace=True)
    val_df.dropna(subset=['text'], inplace=True)
    print(f"Exploded into {len(train_df)} training sentences and {len(val_df)} validation sentences.")
    return train_df, val_df

# MODIFICATION: The function now reads the local CSV file.
def load_blind_test_data(file_path):
    """Loads and prepares the blind test set from a local CSV file."""
    print(f"\n--- Loading Blind Test Data from local file: {file_path} ---")
    try:
        # Load the CSV file from the session storage
        doc_test_df = pd.read_csv(file_path)

        # The local file has 'ID' and 'Document' columns. Rename them.
        doc_test_df = doc_test_df.rename(columns={'ID': 'doc_id', 'Document': 'text'})
        print(f"Loaded {len(doc_test_df)} documents from the blind test file.")

        # Explode documents into sentences for prediction, keeping track of the doc_id
        sentence_test_df = doc_test_df.assign(text=doc_test_df['text'].str.split('\n')).explode('text')
        sentence_test_df.dropna(subset=['text'], inplace=True)
        print(f"Exploded into {len(sentence_test_df)} sentences for prediction.")
        return sentence_test_df

    except Exception as e:
        print(f"❗️ ERROR loading blind test file: {e}")
        return None

# --- Execute Data Loading ---
train_df, val_df = load_training_and_validation_data()
# MODIFICATION: Call the updated function with the local file path.
test_df = load_blind_test_data(BLIND_TEST_PATH)

if train_df is None or val_df is None or test_df is None:
    print("\nScript aborted due to data loading errors.")
    exit()

# =====================================================================================
# 4. FEATURE ENGINEERING (No Changes Needed)
# =====================================================================================
def get_lexical_features(text, lexicon):
    if not lexicon: return [0.0] * 7
    words = str(text).split()
    if not words: return [0.0] * 7
    word_difficulties = [lexicon.get(word, 3.0) for word in words]
    return [
        float(len(text)), float(len(words)), float(np.mean([len(w) for w in words])),
        float(np.mean(word_difficulties)), float(np.max(word_difficulties)),
        float(np.sum(np.array(word_difficulties) > 4)),
        float(len([w for w in words if w not in lexicon]) / len(words))
    ]

print("\n--- Engineering Lexical Features ---")
try:
    samer_lexicon = pd.read_csv(SAMER_LEXICON_PATH, sep='\t')
    samer_lexicon[['lemma', 'pos']] = samer_lexicon['lemma#pos'].str.split('#', expand=True)
    lexicon_dict = samer_lexicon.set_index('lemma')['readability (rounded average)'].to_dict()
    print(f"Loaded {len(lexicon_dict)} lemmas into lexicon dictionary.")
except FileNotFoundError:
    print("SAMER Lexicon not found. Features will be zeros.")
    lexicon_dict = {}

train_df['text'] = train_df['text'].apply(arabert_preprocessor.preprocess)
val_df['text'] = val_df['text'].apply(arabert_preprocessor.preprocess)
test_df['text'] = test_df['text'].apply(arabert_preprocessor.preprocess)

train_df['features'] = train_df['text'].apply(lambda x: get_lexical_features(x, lexicon_dict)).tolist()
val_df['features'] = val_df['text'].apply(lambda x: get_lexical_features(x, lexicon_dict)).tolist()
test_df['features'] = test_df['text'].apply(lambda x: get_lexical_features(x, lexicon_dict)).tolist()

NUM_FEATURES = 7
print(f"Engineered {NUM_FEATURES} features per sentence.")

# =====================================================================================
# 5. HYBRID MODEL, DATASET, AND TRAINER (No Changes Needed)
# =====================================================================================
class HybridReadabilityModel(nn.Module):
    def __init__(self, model_name, num_extra_features, num_labels):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        transformer_output_dim = self.transformer.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(transformer_output_dim + num_extra_features, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, numerical_features, labels=None):
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = transformer_outputs.last_hidden_state[:, 0, :]
        combined_features = torch.cat([cls_embedding, numerical_features], dim=1)
        logits = self.head(combined_features)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return (loss, logits) if loss is not None else logits

class ReadabilityDataset(TorchDataset):
    def __init__(self, texts, features, labels=None):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=256)
        self.features = features
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['numerical_features'] = torch.tensor(self.features[idx], dtype=torch.float)
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx] - 1, dtype=torch.long)
        return item
    def __len__(self):
        return len(self.features)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"qwk": cohen_kappa_score(p.label_ids, preds, weights='quadratic')}

# =====================================================================================
# 6. TRAINING AND PREDICTION
# =====================================================================================
print("\n===== PREPARING FOR TRAINING AND PREDICTION =====\n")

# --- Create Datasets ---
train_dataset = ReadabilityDataset(train_df['text'].tolist(), train_df['features'].tolist(), train_df['label'].tolist())
val_dataset = ReadabilityDataset(val_df['text'].tolist(), val_df['features'].tolist(), val_df['label'].tolist())
test_dataset = ReadabilityDataset(test_df['text'].tolist(), test_df['features'].tolist())

# --- Initialize Model and Trainer ---
model = HybridReadabilityModel(MODEL_NAME, num_extra_features=NUM_FEATURES, num_labels=NUM_LABELS)
training_args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, "results"),
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

# --- Train ---
print("Starting model training...")
trainer.train()
print("Training finished.")

# --- Predict ---
print("\nGenerating predictions on the test set...")
predictions = trainer.predict(test_dataset)
sentence_level_preds = np.argmax(predictions.predictions, axis=1) + 1
test_df['prediction'] = sentence_level_preds

# --- Aggregate and Save ---
print("Aggregating sentence predictions to document-level using MAX rule...")
doc_level_preds = test_df.groupby('doc_id')['prediction'].max()
submission_df = pd.DataFrame({'Document ID': doc_level_preds.index, 'Prediction': doc_level_preds.values})

print(f"Saving prediction file to: {SUBMISSION_PATH}")
submission_df.to_csv(SUBMISSION_PATH, index=False)

print(f"\nCompressing '{SUBMISSION_FILE_NAME}' into '{ZIPPED_SUBMISSION_PATH}'...")
with zipfile.ZipFile(ZIPPED_SUBMISSION_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(SUBMISSION_PATH, arcname=SUBMISSION_FILE_NAME)

print(f"Submission file '{os.path.basename(ZIPPED_SUBMISSION_PATH)}' created successfully.")
print("\n--- Script Finished ---")
