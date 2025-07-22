# preprocess.py
# This script loads, cleans, maps, and combines all readability datasets
# into a single file for faster training.

import pandas as pd
import numpy as np

# --- CONFIGURATION: Define all file paths ---
BAREC_TRAIN_PATH = 'train.csv'
OSMAN_DATA_PATH = 'Annotated_Paper_Dataset.csv'
SAMER_CORPUS_PATH = 'samer_train.tsv'
SAMER_LEXICON_PATH = 'SAMER-Readability-Lexicon-v2.tsv'
FINAL_OUTPUT_PATH = 'processed_training_data.csv'

def load_barec_data(file_path):
    """Loads the main BAREC competition training data and renames columns."""
    print(f"--- Loading BAREC data from: {file_path} ---")
    try:
        df = pd.read_csv(file_path)
        df.rename(columns={
            'Sentence': 'text',
            'Readability_Level_19': 'label'
        }, inplace=True)
        df = df[['text', 'label']]
        df.dropna(subset=['text', 'label'], inplace=True)
        print(f"✅ Loaded {len(df)} rows from BAREC.")
        return df
    except Exception as e:
        print(f"❗️ ERROR loading BAREC data: {e}")
        return pd.DataFrame()

def load_osman_data(file_path):
    """Loads the 'OSMAN' dataset and maps its fine-grained labels."""
    print(f"--- Loading OSMAN data from: {file_path} ---")
    try:
        df = pd.read_csv(file_path)
        df['label'] = df['Fine-grained'].str.replace('G', '', regex=False).astype(int)
        df.rename(columns={'Text': 'text'}, inplace=True)
        df = df[['text', 'label']]
        df.dropna(subset=['text', 'label'], inplace=True)
        print(f"✅ Loaded and mapped {len(df)} rows from OSMAN data.")
        return df
    except Exception as e:
        print(f"❗️ ERROR loading OSMAN data: {e}")
        return pd.DataFrame()

def load_samer_corpus(file_path):
    """
    Loads the SAMER Corpus from the specified TSV file using the correct column names.
    """
    print(f"--- Loading SAMER Corpus from: {file_path} ---")
    try:
        # 🔴 Read as a Tab-Separated Value (TSV) file and use correct column names
        df = pd.read_csv(file_path, sep='\t')
        
        # L5 is the most complex, L3 is the simplest
        original_df = pd.DataFrame({'text': df['L5'], 'label': 17})  # High difficulty
        simple1_df = pd.DataFrame({'text': df['L4'], 'label': 10}) # Medium difficulty
        simple2_df = pd.DataFrame({'text': df['L3'], 'label': 5})  # Low difficulty
        
        combined_df = pd.concat([original_df, simple1_df, simple2_df], ignore_index=True)
        combined_df.dropna(inplace=True)
        print(f"✅ Loaded and mapped {len(combined_df)} sentences from SAMER Corpus.")
        return combined_df
    except KeyError:
        print(f"❗️ KEY ERROR in SAMER Corpus: Make sure columns 'L5', 'L4', 'L3' exist in {file_path}.")
        return pd.DataFrame()
    except Exception as e:
        print(f"❗️ ERROR loading SAMER Corpus: {e}. Check file path and column names.")
        return pd.DataFrame()

def create_samer_lexicon_sentences(file_path):
    """
    Loads the SAMER Lexicon from the specified TSV file using correct column names.
    """
    print(f"--- Creating sentences from SAMER Lexicon: {file_path} ---")
    try:
        # 🔴 Read as a Tab-Separated Value (TSV) file
        df = pd.read_csv(file_path, sep='\t')
        
        # 🔴 Rename correct columns and create the 'word' column
        df.rename(columns={'readability (rounded average)': 'samer_level'}, inplace=True)
        # Extract the word (lemma) by splitting the string before the '#'
        df['word'] = df['lemma#pos'].apply(lambda x: str(x).split('#')[0])

        # Map the 5 levels of SAMER to the 19-point BAREC scale
        level_map = {1: 3, 2: 7, 3: 11, 4: 15, 5: 18}
        df['label'] = df['samer_level'].map(level_map)
        
        # Create simple template sentences from the words
        df['text'] = df['word'].apply(lambda x: f"هذه كلمة بسيطة: {x}.")
        
        df = df[['text', 'label']]
        df.dropna(inplace=True)
        print(f"✅ Created {len(df)} example sentences from SAMER Lexicon.")
        return df
    except KeyError:
        print(f"❗️ KEY ERROR in SAMER Lexicon: Make sure 'lemma#pos' and 'readability (rounded average)' columns exist in {file_path}.")
        return pd.DataFrame()
    except Exception as e:
        print(f"❗️ ERROR processing SAMER Lexicon: {e}. Check file path and column names.")
        return pd.DataFrame()


if __name__ == '__main__':
    print("--- Starting Offline Data Preprocessing ---")
    
    barec_df = load_barec_data(BAREC_TRAIN_PATH)
    osman_df = load_osman_data(OSMAN_DATA_PATH)
    samer_corpus_df = load_samer_corpus(SAMER_CORPUS_PATH)
    samer_lexicon_df = create_samer_lexicon_sentences(SAMER_LEXICON_PATH)
    
    all_data = pd.concat([barec_df, osman_df, samer_corpus_df, samer_lexicon_df], ignore_index=True)
    
    all_data.dropna(subset=['text', 'label'], inplace=True)
    all_data = all_data[all_data['text'].str.len() > 5]
    all_data['label'] = all_data['label'].astype(int)
    
    final_df = all_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    final_df.to_csv(FINAL_OUTPUT_PATH, index=False)
    
    print("\n--- Preprocessing Complete! ---")
    print(f"✅ Final combined dataset has {len(final_df)} rows.")
    print(f"Saved to: {FINAL_OUTPUT_PATH}")
