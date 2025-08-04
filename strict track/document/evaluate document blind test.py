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
    Trainer
)

from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.dediac import dediac_ar

# --- Configuration ---
MODEL_NAME = "CAMeL-Lab/readability-arabertv2-d3tok-reg"
NUM_LABELS = 1
TARGET_CLASSES = 19
BASE_DIR = '.'
DATA_DIR = os.path.join(BASE_DIR, "data")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# --- File Paths ---
DOC_BLIND_TEST_PATH = os.path.join(DATA_DIR, 'doc_blind_test_data.csv') 
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission_document_regression_final.csv")
ZIPPED_SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission_document_regression_final.zip")


# --- DATA PREPROCESSING (UNCHANGED) ---
def preprocess_d3tok(text, disambiguator):
    if not isinstance(text, str) or not text.strip(): return ""
    tokens = simple_word_tokenize(text)
    disambiguated_sentence = disambiguator.disambiguate(tokens)
    d3tok_forms = []
    for disambig_word in disambiguated_sentence:
        if disambig_word.analyses:
            analysis_dict = disambig_word.analyses[0][1]
            if 'd3tok' in analysis_dict:
                d3tok = dediac_ar(analysis_dict['d3tok']).replace("_+", " +").replace("+_", "+ ")
                d3tok_forms.append(d3tok)
            else: d3tok_forms.append(disambig_word.word)
        else: d3tok_forms.append(disambig_word.word)
    return " ".join(d3tok_forms)

# --- Initialize Tools ---
print("Initializing CAMeL Tools...")
bert_disambiguator = BERTUnfactoredDisambiguator.pretrained('msa')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("✔ Tools initialized.")


# --- DATASET CLASS (UNCHANGED) ---
class ReadabilityDataset(TorchDataset):
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=256)
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None: item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    def __len__(self):
        return len(self.encodings.get('input_ids', []))


# --- MODEL LOADING (UNCHANGED) ---
print("\n===== LOADING PRE-TRAINED SENTENCE-LEVEL MODEL =====\n")
CHECKPOINT_DIR = r"D:\arabic_readability_project\results\regression_readability-arabertv2-d3tok-reg\checkpoint-10284"
if os.path.exists(os.path.join(CHECKPOINT_DIR, "model.safetensors")):
    print(f"✔ Found checkpoint at: {CHECKPOINT_DIR}")
    print("Loading model from checkpoint...")
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
    trainer = Trainer(model=model)
    print("✔ Model loaded successfully.")
else:
    print(f"! ERROR: Checkpoint not found at '{CHECKPOINT_DIR}'. Please check the path.")
    exit()


# --- DOCUMENT-LEVEL FINAL PREDICTION AND SUBMISSION ---
print("\n===== DOCUMENT-LEVEL FINAL PREDICTION AND SUBMISSION =====\n")
try:
    print(f"Loading document test data from {DOC_BLIND_TEST_PATH}...")
    doc_test_df = pd.read_csv(DOC_BLIND_TEST_PATH)
    # --- CHANGE: Drop rows if 'Sentences' column is empty, not 'Document' ---
    doc_test_df.dropna(subset=['ID', 'Sentences'], inplace=True)
    
    print("Processing documents: splitting into sentences by newline characters...")
    all_sentences, doc_ids = [], []
    for _, row in doc_test_df.iterrows():
        doc_id = row['ID']
        # --- THE CRITICAL FIX: Read text from the 'Sentences' column ---
        full_document_text = row['Sentences']
        
        if isinstance(full_document_text, str) and full_document_text.strip():
            sentences_list = full_document_text.split('\n')
            sentences_list = [s.strip() for s in sentences_list if s.strip()]
            if sentences_list:
                all_sentences.extend(sentences_list)
                doc_ids.extend([doc_id] * len(sentences_list))
        else:
            print(f"Warning: Document ID {doc_id} has empty or invalid text in 'Sentences' column. Skipping.")
            continue

    if not all_sentences:
        print("\n! ERROR: No sentences were extracted. Check the 'Sentences' column in your CSV.")
        exit()

    sentence_df = pd.DataFrame({'doc_id': doc_ids, 'sentence_text': all_sentences})
    
    # Save split sentences for review
    review_split_path = 'review_split_sentences.csv'
    sentence_df.to_csv(review_split_path, index=False, encoding='utf-8-sig') # Use utf-8-sig for Excel
    print(f"\n✔ Raw split sentences saved to {review_split_path}")
    print(f"Successfully created {len(sentence_df):,} sentences from {len(doc_test_df):,} documents.")

    print("\nPreprocessing all sentences to D3Tok format (this may take a moment)...")
    sentence_df['processed_text'] = sentence_df['sentence_text'].apply(lambda x: preprocess_d3tok(x, bert_disambiguator))
    
    # Save D3tok output for review
    review_d3tok_path = 'review_d3tok_processed_output.csv'
    sentence_df[['sentence_text', 'processed_text']].to_csv(review_d3tok_path, index=False, encoding='utf-8-sig')
    print(f"✔ D3tok processed output saved to {review_d3tok_path}")
    
    print("\nGenerating predictions for all sentences...")
    test_dataset = ReadabilityDataset(sentence_df['processed_text'].tolist())
    predictions = trainer.predict(test_dataset)
    sentence_df['raw_prediction'] = predictions.predictions.flatten()
    
    print("Aggregating results: finding the max readability score per document...")
    doc_predictions = sentence_df.groupby('doc_id')['raw_prediction'].max()
    
    rounded_preds = np.round(doc_predictions.values)
    clipped_preds = np.clip(rounded_preds, 0, TARGET_CLASSES - 1)
    
    final_submission_df = pd.DataFrame({'Sentence ID': doc_test_df['ID']})
    pred_df = pd.DataFrame({
        'Sentence ID': doc_predictions.index, 
        'Prediction': (clipped_preds + 1).astype(int)
    })
    final_submission_df = final_submission_df.merge(pred_df, on='Sentence ID', how='left')
    final_submission_df['Prediction'].fillna(1, inplace=True)
    final_submission_df['Prediction'] = final_submission_df['Prediction'].astype(int)

    print(f"\nSaving prediction file to: {SUBMISSION_PATH}")
    final_submission_df.to_csv(SUBMISSION_PATH, index=False)
    
    print(f"Compressing {os.path.basename(SUBMISSION_PATH)} into {os.path.basename(ZIPPED_SUBMISSION_PATH)}...")
    with zipfile.ZipFile(ZIPPED_SUBMISSION_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(SUBMISSION_PATH, arcname=os.path.basename(SUBMISSION_PATH))
        
    print(f"✔ Submission file {os.path.basename(ZIPPED_SUBMISSION_PATH)} created successfully.")

except FileNotFoundError:
    print(f"! ERROR: Test file not found at '{DOC_BLIND_TEST_PATH}'.")
except Exception as e:
    print(f"An error occurred during final document prediction: {e}")

print("\n--- Script Finished ---")
