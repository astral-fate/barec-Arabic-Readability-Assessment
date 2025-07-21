
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import zipfile
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, EarlyStoppingCallback
from arabert.preprocess import ArabertPreprocessor
# MODIFICATION: Import FarasaSegmenter to manually set its mode
from farasa.segmenter import FarasaSegmenter

# This setting is still good practice.
os.environ['JAVA_TOOL_OPTIONS'] = '-Dfile.encoding=UTF-8'

# =====================================================================================
# 1. CONFIGURATION
# =====================================================================================
# --- Model & Preprocessing ---
MODEL_NAME = "aubmindlab/bert-large-arabertv2"
arabert_preprocessor = ArabertPreprocessor(model_name=MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- MODIFICATION: Force Farasa to use Standalone mode ---
# This is the key fix for the persistent UnicodeDecodeError.
# It's more stable for long sentences than the default interactive mode.
print("Switching FarasaSegmenter to more stable Standalone mode...")
arabert_preprocessor.farasa_segmenter = FarasaSegmenter(interactive=False)


# --- Training & Classification ---
RANDOM_STATE = 42
NUM_LABELS = 19

# --- File Paths for Local Execution ---
BASE_DIR = "./"
os.makedirs(BASE_DIR, exist_ok=True)

# Data paths
BAREC_TRAIN_PATH = 'train.csv'
BAREC_DEV_PATH = 'dev.csv'
BLIND_TEST_PATH = 'blind_test_data.csv'

# Submission file paths
SUBMISSION_FILE_NAME = "prediction.csv"
SUBMISSION_PATH = os.path.join(BASE_DIR, SUBMISSION_FILE_NAME)
ZIPPED_SUBMISSION_PATH = os.path.join(BASE_DIR, "prediction.zip")

# =====================================================================================
# 2. DATA LOADING
# =====================================================================================

def load_training_and_validation_data():
    """Loads and prepares training and validation data from local CSVs."""
    print("--- Loading BAREC Data from CSV files ---")
    try:
        train_df = pd.read_csv(BAREC_TRAIN_PATH, encoding='utf-8')
        val_df = pd.read_csv(BAREC_DEV_PATH, encoding='utf-8')
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

def load_blind_test_data(file_path):
    """Loads and prepares the blind test set from a local CSV file."""
    print(f"\n--- Loading Blind Test Data from local file: {file_path} ---")
    try:
        doc_test_df = pd.read_csv(file_path, encoding='utf-8')
        doc_test_df = doc_test_df.rename(columns={'ID': 'doc_id', 'Document': 'text'})
        print(f"Loaded {len(doc_test_df)} documents from the blind test file.")
        sentence_test_df = doc_test_df.assign(text=doc_test_df['text'].str.split('\n')).explode('text')
        sentence_test_df.dropna(subset=['text'], inplace=True)
        sentence_test_df['doc_id'] = sentence_test_df['doc_id'].ffill()
        print(f"Exploded into {len(sentence_test_df)} sentences for prediction.")
        return sentence_test_df
    except FileNotFoundError as e:
        print(f"❗️ ERROR loading blind test file: {e}")
        return None

# --- Execute Data Loading and Preprocessing ---
train_df, val_df = load_training_and_validation_data()
test_df = load_blind_test_data(BLIND_TEST_PATH)

if train_df is None or val_df is None or test_df is None:
    print("\nScript aborted due to data loading errors.")
    exit()

print("\n--- Preprocessing Text ---")
train_df['text'] = train_df['text'].apply(arabert_preprocessor.preprocess)
val_df['text'] = val_df['text'].apply(arabert_preprocessor.preprocess)
test_df['text'] = test_df['text'].apply(arabert_preprocessor.preprocess)
print("Text preprocessing finished.")

# =====================================================================================
# 3. MODEL, DATASET, AND TRAINER
# =====================================================================================

class ReadabilityModel(nn.Module):
    """A standard Transformer model for classification, without extra features."""
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        transformer_output_dim = self.transformer.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(transformer_output_dim, num_labels)
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
    """A simplified dataset class that only handles tokenized text and labels."""
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
    """Computes the quadratic weighted kappa score for evaluation."""
    preds = np.argmax(p.predictions, axis=1)
    return {"qwk": cohen_kappa_score(p.label_ids, preds, weights='quadratic')}


# =====================================================================================
# 4. TRAINING AND PREDICTION
# =====================================================================================
print("\n===== PREPARING FOR TRAINING AND PREDICTION =====\n")

# --- Create Datasets ---
train_dataset = ReadabilityDataset(train_df['text'].tolist(), train_df['label'].tolist())
val_dataset = ReadabilityDataset(val_df['text'].tolist(), val_df['label'].tolist())
test_dataset = ReadabilityDataset(test_df['text'].tolist())

# --- Initialize Model and Trainer ---
model = ReadabilityModel(MODEL_NAME, num_labels=NUM_LABELS)

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
