# =====================================================================================
# 0. INSTALLATIONS & IMPORTS
# =====================================================================================
# This will install all necessary libraries quietly.
# !pip install transformers[torch] datasets pandas scikit-learn arabert accelerate pyarrow -q

import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer
from arabert.preprocess import ArabertPreprocessor

print("✅ Libraries imported successfully.")

# =====================================================================================
# 1. CONFIGURATION
# =====================================================================================
# --- Model & Tokenizer ---
# Using the model from your original script for consistency
MODEL_NAME = "aubmindlab/bert-large-arabertv2"
MAX_LENGTH = 256  # Max sequence length for tokenizer

# --- File Paths ---
# Assumes your initial dataset is in the default /kaggle/input/sentses directory
RAW_DATA_DIR = '/kaggle/input/sentses/'
BAREC_TRAIN_PATH = os.path.join(RAW_DATA_DIR, 'train.csv')
BAREC_DEV_PATH = os.path.join(RAW_DATA_DIR, 'dev.csv')
BLIND_TEST_PATH = os.path.join(RAW_DATA_DIR, 'blind_test_data.csv')
# The SAMER files are assumed to be in the same directory for this example
SAMER_CORPUS_PATH = os.path.join(RAW_DATA_DIR, 'samer_train.tsv')
SAMER_LEXICON_PATH = os.path.join(RAW_DATA_DIR, 'SAMER-Readability-Lexicon-v2.tsv')

# --- Output Path ---
# Processed files will be saved here, ready for output
OUTPUT_DIR = '/kaggle/working/'

# --- Initialize Processors ---
try:
    arabert_preprocessor = ArabertPreprocessor(model_name=MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("✅ AraBERT Preprocessor and Tokenizer initialized.")
except Exception as e:
    print(f"❗️ Error initializing processors: {e}")
    # Exit if the core components can't be loaded
    exit()

# =====================================================================================
# 2. DATA LOADING FUNCTIONS
# =====================================================================================

def load_samer_lexicon(file_path):
    """Loads the SAMER Lexicon for feature engineering."""
    print("\n--- Loading SAMER Lexicon ---")
    try:
        df = pd.read_csv(file_path, sep='\t')
        df[['lemma', 'pos']] = df['lemma#pos'].str.split('#', expand=True)
        lexicon_dict = df.set_index('lemma')['readability (rounded average)'].to_dict()
        print(f"Loaded {len(lexicon_dict)} lemmas into lexicon dictionary.")
        return lexicon_dict
    except FileNotFoundError:
        print("❗️ SAMER Lexicon not found. Lexical features will be disabled.")
        return {}

def load_training_and_validation_data(lexicon):
    """Loads and augments training/validation data."""
    print("\n--- Loading BAREC Training & Validation Data ---")
    try:
        train_df = pd.read_csv(BAREC_TRAIN_PATH)[['Sentence', 'Readability_Level_19']].rename(columns={'Sentence': 'text', 'Readability_Level_19': 'label'})
        val_df = pd.read_csv(BAREC_DEV_PATH)[['Sentence', 'Readability_Level_19']].rename(columns={'Sentence': 'text', 'Readability_Level_19': 'label'})
        train_df.dropna(subset=['text', 'label'], inplace=True)
        val_df.dropna(subset=['text', 'label'], inplace=True)
        print(f"Loaded {len(train_df)} BAREC training sentences and {len(val_df)} validation sentences.")
    except Exception as e:
        print(f"❗️ ERROR loading BAREC CSVs: {e}")
        return None, None

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
        print(f"Loaded {len(samer_augmentation_df)} sentences from SAMER.")
        
        full_train_df = pd.concat([train_df, samer_augmentation_df], ignore_index=True)
        full_train_df.dropna(subset=['text', 'label'], inplace=True)
        full_train_df['label'] = full_train_df['label'].astype(int)
        full_train_df = full_train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Total unified training sentences: {len(full_train_df)}")
        return full_train_df, val_df
    except FileNotFoundError:
        print("❗️ SAMER Corpus not found, proceeding with BAREC data only.")
        return train_df, val_df

def load_blind_test_data(file_path):
    """Loads and prepares the blind test set."""
    print(f"\n--- Loading Blind Test Data ---")
    try:
        doc_test_df = pd.read_csv(file_path).rename(columns={'ID': 'doc_id', 'Sentence': 'text'})
        sentence_test_df = doc_test_df.assign(text=doc_test_df['text'].str.split('\n')).explode('text').reset_index(drop=True)
        sentence_test_df.dropna(subset=['text'], inplace=True)
        print(f"Loaded and exploded {len(sentence_test_df)} sentences for prediction.")
        return sentence_test_df
    except Exception as e:
        print(f"❗️ ERROR loading blind test file: {e}")
        return None

# =====================================================================================
# 3. FEATURE ENGINEERING & PREPROCESSING FUNCTION
# =====================================================================================

def get_lexical_features(text, lexicon):
    """Calculates lexical features based on the SAMER lexicon."""
    if not lexicon or not isinstance(text, str):
        return [0.0] * 7

    words = text.split()
    if not words: return [0.0] * 7

    word_difficulties = [lexicon.get(word, 3.0) for word in words]
    
    # Use float() to ensure type consistency for pyarrow
    features = [
        float(len(text)),
        float(len(words)),
        float(np.mean([len(w) for w in words]) if words else 0.0),
        float(np.mean(word_difficulties)),
        float(np.max(word_difficulties)),
        float(np.sum(np.array(word_difficulties) > 4)),
        float(len([w for w in words if w not in lexicon]) / len(words))
    ]
    return features

def process_dataframe(df, lexicon, is_test=False):
    """Applies all preprocessing steps to a dataframe."""
    print(f"\n--- Starting processing for {'Test' if is_test else 'Train/Val'} dataframe ---")
    
    # 1. Clean and preprocess text
    print("Step 1: Applying AraBERT preprocessor...")
    df['text_preprocessed'] = df['text'].apply(arabert_preprocessor.preprocess)
    
    # 2. Engineer lexical features
    print("Step 2: Engineering lexical features...")
    features = np.array(df['text_preprocessed'].apply(lambda x: get_lexical_features(x, lexicon)).tolist())
    df['features'] = list(features)

    # 3. Tokenize text
    print("Step 3: Tokenizing text...")
    encodings = tokenizer(
        df['text_preprocessed'].tolist(),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    df['input_ids'] = encodings['input_ids']
    df['attention_mask'] = encodings['attention_mask']
    
    # 4. Select final columns
    if is_test:
        final_cols = ['doc_id', 'input_ids', 'attention_mask', 'features']
    else:
        final_cols = ['label', 'input_ids', 'attention_mask', 'features']
    
    print("✅ Processing complete.")
    return df[final_cols]

# =====================================================================================
# 4. EXECUTION
# =====================================================================================

# Load all data first
samer_lexicon = load_samer_lexicon(SAMER_LEXICON_PATH)
train_df, val_df = load_training_and_validation_data(samer_lexicon)
test_df = load_blind_test_data(BLIND_TEST_PATH)

# Check if data loading was successful before proceeding
if train_df is not None and val_df is not None and test_df is not None:
    # Process each dataframe
    processed_train_df = process_dataframe(train_df, samer_lexicon)
    processed_val_df = process_dataframe(val_df, samer_lexicon)
    processed_test_df = process_dataframe(test_df, samer_lexicon, is_test=True)

    # Save the processed dataframes to Feather files
    print("\n--- Saving processed dataframes to Feather files ---")
    
    train_save_path = os.path.join(OUTPUT_DIR, 'train_processed.feather')
    val_save_path = os.path.join(OUTPUT_DIR, 'val_processed.feather')
    test_save_path = os.path.join(OUTPUT_DIR, 'test_processed.feather')

    processed_train_df.to_feather(train_save_path)
    print(f"✅ Training data saved to {train_save_path}")
    
    processed_val_df.to_feather(val_save_path)
    print(f"✅ Validation data saved to {val_save_path}")
    
    processed_test_df.to_feather(test_save_path)
    print(f"✅ Test data saved to {test_save_path}")

    print("\n🎉 All preprocessing is complete. You can now save this notebook's output as a new dataset.")
else:
    print("\n❗️ Script aborted due to data loading errors.")
