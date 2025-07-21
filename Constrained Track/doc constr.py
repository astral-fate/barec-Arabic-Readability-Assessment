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
SAMER_CORPUS_PATH = os.path.join(BASE_DIR, 'samer_train.tsv')
SAMER_LEXICON_PATH = os.path.join(BASE_DIR, 'SAMER-Readability-Lexicon-v2.tsv')

# Submission file paths
SUBMISSION_FILE_NAME = "prediction"
SUBMISSION_PATH = os.path.join(BASE_DIR, SUBMISSION_FILE_NAME)
ZIPPED_SUBMISSION_PATH = os.path.join(BASE_DIR, f"{SUBMISSION_FILE_NAME}.zip")

# =====================================================================================
# 3. REVISED: DATA LOADING FOR SENTENCE-LEVEL CSVs
# =====================================================================================
def load_data_for_training():
    """
    Loads data from sentence-level CSVs and augments with SAMER corpus.
    """
    print("--- Loading BAREC Data from Sentence-Level CSV files ---")
    try:
        # Load the datasets, which are already one sentence per row
        train_df = pd.read_csv(BAREC_TRAIN_PATH)
        val_df = pd.read_csv(BAREC_DEV_PATH)
        test_df = pd.read_csv(BAREC_TEST_PATH)

        # --- Process Training and Validation Data ---
        # Select the text and label columns, and rename them for consistency
        train_df = train_df[['Sentence', 'Readability_Level_19']].rename(columns={'Sentence': 'text', 'Readability_Level_19': 'label'})
        val_df = val_df[['Sentence', 'Readability_Level_19']].rename(columns={'Sentence': 'text', 'Readability_Level_19': 'label'})
        
        # --- Process Test Data ---
        # The 'Document' column appears to be the document identifier needed for aggregation
        test_df = test_df[['Sentence', 'Document']].rename(columns={'Sentence': 'text', 'Document': 'doc_id'})

    except FileNotFoundError as e:
        print(f"❗️ ERROR: {e}. Make sure train.csv, dev.csv, and test.csv are in your project folder.")
        return None, None, None
    except KeyError as e:
        print(f"❗️ ERROR: A required column was not found: {e}. This means the CSV format is unexpected.")
        return None, None, None

    print(f"Loaded {len(train_df)} sentences from BAREC train.csv.")
    print(f"Loaded {len(val_df)} sentences from BAREC dev.csv (for validation).")
    print(f"Loaded {len(test_df)} sentences from BAREC test.csv.")


    print("\n--- Loading SAMER Corpus for Augmentation ---")
    try:
        samer_level_map = {'L3': 4, 'L4': 10, 'L5': 16}
        samer_df = pd.read_csv(SAMER_CORPUS_PATH, sep='\t')
        samer_records = []
        for level_name, barec_level in samer_level_map.items():
            samer_subset = samer_df[[level_name]].dropna().rename(columns={level_name: 'text'})
            samer_subset['label'] = barec_level
            samer_records.append(samer_subset)
        samer_augmentation_df = pd.concat(samer_records, ignore_index=True)
        print(f"Loaded {len(samer_augmentation_df)} sentences from SAMER Corpus.")

        # Combine BAREC training data with SAMER data
        full_train_df = pd.concat([train_df, samer_augmentation_df], ignore_index=True)
        full_train_df.dropna(subset=['text', 'label'], inplace=True)
        full_train_df['label'] = full_train_df['label'].astype(int)
        full_train_df = full_train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"Total unified training sentences: {len(full_train_df)}")
    except FileNotFoundError:
        print("SAMER Corpus not found, proceeding with BAREC data only.")
        full_train_df = train_df.copy()


    return full_train_df, val_df, test_df

def load_samer_lexicon(file_path):
    """Loads the SAMER Lexicon."""
    print("\n--- Loading SAMER Lexicon for Feature Engineering ---")
    try:
        df = pd.read_csv(file_path, sep='\t')
        df[['lemma', 'pos']] = df['lemma#pos'].str.split('#', expand=True)
        lexicon_dict = df.set_index('lemma')['readability (rounded average)'].to_dict()
        print(f"Loaded {len(lexicon_dict)} lemmas into lexicon dictionary.")
        return lexicon_dict
    except FileNotFoundError:
        print("SAMER Lexicon not found. Features requiring it will be disabled.")
        return {}


# Execute loading functions
train_df, val_df, test_df = load_data_for_training()
if train_df is None:
    exit()
samer_lexicon = load_samer_lexicon(SAMER_LEXICON_PATH)


# =====================================================================================
# 4. FEATURE ENGINEERING
# =====================================================================================
def get_lexical_features(text, lexicon):
    # If lexicon is empty (not found), return zeros.
    if not lexicon:
        return [0.0] * 7
        
    words = str(text).split()
    if not words: return [0.0] * 7
    word_difficulties = [lexicon.get(word, 3.0) for word in words]
    features = [
        float(len(text)), float(len(words)), float(np.mean([len(w) for w in words])),
        float(np.mean(word_difficulties)), float(np.max(word_difficulties)),
        float(np.sum(np.array(word_difficulties) > 4)),
        float(len([w for w in words if w not in lexicon]) / len(words))
    ]
    return features

print("\n--- Engineering Lexical Features for all datasets ---")
train_df['text'] = train_df['text'].apply(arabert_preprocessor.preprocess)
val_df['text'] = val_df['text'].apply(arabert_preprocessor.preprocess)
test_df['text'] = test_df['text'].apply(arabert_preprocessor.preprocess)

train_features = np.array(train_df['text'].apply(lambda x: get_lexical_features(x, samer_lexicon)).tolist())
val_features = np.array(val_df['text'].apply(lambda x: get_lexical_features(x, samer_lexicon)).tolist())
test_features = np.array(test_df['text'].apply(lambda x: get_lexical_features(x, samer_lexicon)).tolist())

train_df['features'] = list(train_features)
val_df['features'] = list(val_features)
test_df['features'] = list(test_features)
NUM_FEATURES = len(train_features[0])
print(f"Engineered {NUM_FEATURES} features per sentence.")


# =====================================================================================
# 5. HYBRID MODEL, DATASET, AND TRAINER FOR CLASSIFICATION
# =====================================================================================
class HybridReadabilityModel(nn.Module):
    def __init__(self, model_name, num_extra_features, num_labels):
        super(HybridReadabilityModel, self).__init__()
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
    logits, labels = p.predictions, p.label_ids
    preds = np.argmax(logits, axis=1)
    return {"qwk": cohen_kappa_score(labels, preds, weights='quadratic')}

# =====================================================================================
# 6. TRAINING AND EVALUATION
# =====================================================================================
print("\n===== PREPARING FOR CLASSIFICATION TRAINING RUN =====\n")

train_dataset = ReadabilityDataset(train_df['text'].tolist(), train_df['features'].tolist(), train_df['label'].tolist())
val_dataset = ReadabilityDataset(val_df['text'].tolist(), val_df['features'].tolist(), val_df['label'].tolist())
test_dataset = ReadabilityDataset(test_df['text'].tolist(), test_df['features'].tolist())

model = HybridReadabilityModel(MODEL_NAME, num_extra_features=NUM_FEATURES, num_labels=NUM_LABELS)
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
# 7. PREDICTION AND SUBMISSION FILE GENERATION
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