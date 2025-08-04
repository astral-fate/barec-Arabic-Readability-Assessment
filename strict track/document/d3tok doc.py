import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm

from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.dediac import dediac_ar

# =====================================================================================
# 1. CONFIGURATION
# =====================================================================================
BASE_DIR = '.'
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --- INPUT FILES ---
# The source file with the original document text
DOC_TEST_PATH = os.path.join(DATA_DIR, 'doc_test.csv')
# The lexicon needed for hybrid feature calculation
SAMER_LEXICON_PATH = os.path.join(DATA_DIR, 'samer_lexicon.tsv')

# --- OUTPUT FILE ---
# The destination for our pre-processed data
PREPROCESSED_OUTPUT_PATH = os.path.join(DATA_DIR, 'doc_test_preprocessed.csv')

# This is the number of features your hybrid model expects
NUM_FEATURES = 7

# =====================================================================================
# 2. PRE-PROCESSING FUNCTION (Handles both d3tok and hybrid features)
# =====================================================================================
def calculate_features_and_d3tok(sentence_text, disambiguator, lexicon_map):
    """
    Processes a single sentence to get its d3tok form and a vector of numeric features.
    """
    if not isinstance(sentence_text, str) or not sentence_text.strip():
        return ([0.0] * NUM_FEATURES, "", "")

    try:
        tokens = simple_word_tokenize(sentence_text)
        disambiguated_sentence = disambiguator.disambiguate(tokens)

        # --- D3tok processing ---
        d3tok_forms = []
        for da in disambiguated_sentence:
            if da.analyses and 'd3tok' in da.analyses[0][1]:
                d3tok_value = da.analyses[0][1]['d3tok']
                d3tok_forms.append(dediac_ar(d3tok_value).replace("_+", " +").replace("+_", "+ "))
            else:
                d3tok_forms.append(da.word)
        d3tok_text = " ".join(d3tok_forms)

        # --- Feature calculation (from your hybrid script) ---
        scores = []
        for dw in disambiguated_sentence:
            if dw.analyses:
                analysis = dw.analyses[0][1]
                lemma, pos = analysis.get('lex'), analysis.get('pos')
                if pos and isinstance(lemma, str):
                    score = lexicon_map.get(f"{dediac_ar(lemma)}#{pos}")
                    if score is not None:
                        scores.append(score)

        avg_readability = np.mean(scores) if scores else 0.0
        max_readability = np.max(scores) if scores else 0.0
        # Placeholder for other 5 features as in your script
        feature_3, feature_4, feature_5, feature_6, feature_7 = 0.0, 0.0, 0.0, 0.0, 0.0
        feature_vector = [avg_readability, max_readability, feature_3, feature_4, feature_5, feature_6, feature_7]

        # Return the feature vector string representation and the d3tok text
        return str(feature_vector), d3tok_text, sentence_text

    except Exception as e:
        print(f"Warning: An error '{e}' occurred on sentence. Skipping.")
        return str([0.0] * NUM_FEATURES), "", sentence_text

# =====================================================================================
# 3. MAIN EXECUTION
# =====================================================================================
if __name__ == "__main__":
    print("--- Starting One-Time Pre-processing Script ---")
    
    # --- Load source data and tools ---
    print(f"Loading documents from: {DOC_TEST_PATH}")
    doc_df = pd.read_csv(DOC_TEST_PATH)
    doc_df.dropna(subset=['ID', 'Sentences'], inplace=True)
    
    print(f"Loading SAMER Lexicon from: {SAMER_LEXICON_PATH}")
    lexicon_df = pd.read_csv(SAMER_LEXICON_PATH, sep='\t')
    lexicon_map = lexicon_df.set_index('lemma#pos')['readability (rounded average)'].to_dict()

    print("Initializing CAMeL Tools Disambiguator (this may take a moment)...")
    disambiguator = BERTUnfactoredDisambiguator.pretrained('msa')
    print("✔ Tools initialized.")

    # --- Process all documents and sentences ---
    processed_rows = []
    print("\nProcessing all documents. This is the slow part and will only be done once.")
    for _, row in tqdm(doc_df.iterrows(), total=len(doc_df), desc="Processing Documents"):
        doc_id = row['ID']
        full_text = row['Sentences']

        if isinstance(full_text, str) and full_text.strip():
            sentences_list = [s.strip() for s in full_text.split('\n') if s.strip()]
            for sentence in sentences_list:
                # Calculate features and d3tok text for each sentence
                features_str, processed_text, original_sentence = calculate_features_and_d3tok(sentence, disambiguator, lexicon_map)
                processed_rows.append({
                    'doc_id': doc_id,
                    'original_sentence': original_sentence,
                    'processed_text': processed_text,
                    'features': features_str # Stored as a string '[0.0, 0.0, ...]'
                })

    # --- Save to a new CSV file ---
    if not processed_rows:
        print("❌ Error: No sentences were processed. Please check your input file.")
    else:
        processed_df = pd.DataFrame(processed_rows)
        print(f"\nSaving pre-processed data to: {PREPROCESSED_OUTPUT_PATH}")
        processed_df.to_csv(PREPROCESSED_OUTPUT_PATH, index=False, encoding='utf-8-sig')
        print(f"✔ Success! Pre-processing complete. You can now use the evaluation scripts.")

    print("\n--- Script Finished ---")
