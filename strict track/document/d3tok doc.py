import pandas as pd
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

# --- INPUT FILE ---
# The source file with the original document text
DOC_TEST_PATH = os.path.join(DATA_DIR, 'doc_test.csv')

# --- OUTPUT FILE ---
# The destination for our d3tok-only pre-processed data
PREPROCESSED_OUTPUT_PATH = os.path.join(DATA_DIR, 'doc_test_d3tok_only_preprocessed.csv')

# =====================================================================================
# 2. PRE-PROCESSING FUNCTION (d3tok only)
# =====================================================================================
def preprocess_d3tok_only(sentence_text, disambiguator):
    """
    Processes a single sentence to get its d3tok form.
    """
    if not isinstance(sentence_text, str) or not sentence_text.strip():
        return "", sentence_text

    try:
        tokens = simple_word_tokenize(sentence_text)
        disambiguated_sentence = disambiguator.disambiguate(tokens)

        d3tok_forms = []
        for da in disambiguated_sentence:
            if da.analyses and 'd3tok' in da.analyses[0][1]:
                d3tok_value = da.analyses[0][1]['d3tok']
                d3tok_forms.append(dediac_ar(d3tok_value).replace("_+", " +").replace("+_", "+ "))
            else:
                d3tok_forms.append(da.word)
        d3tok_text = " ".join(d3tok_forms)

        return d3tok_text, sentence_text

    except Exception as e:
        print(f"Warning: An error '{e}' occurred on sentence. Skipping.")
        return "", sentence_text

# =====================================================================================
# 3. MAIN EXECUTION
# =====================================================================================
if __name__ == "__main__":
    print("--- Starting One-Time d3tok-Only Pre-processing Script ---")

    # --- Load source data and tools ---
    print(f"Loading documents from: {DOC_TEST_PATH}")
    doc_df = pd.read_csv(DOC_TEST_PATH)
    doc_df.dropna(subset=['ID', 'Sentences'], inplace=True)

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
                # Calculate d3tok text for each sentence
                processed_text, original_sentence = preprocess_d3tok_only(sentence, disambiguator)
                processed_rows.append({
                    'doc_id': doc_id,
                    'original_sentence': original_sentence,
                    'processed_text': processed_text,
                })

    # --- Save to a new CSV file ---
    if not processed_rows:
        print("❌ Error: No sentences were processed. Please check your input file.")
    else:
        processed_df = pd.DataFrame(processed_rows)
        print(f"\nSaving pre-processed data to: {PREPROCESSED_OUTPUT_PATH}")
        processed_df.to_csv(PREPROCESSED_OUTPUT_PATH, index=False, encoding='utf-8-sig')
        print(f"✔ Success! Pre-processing complete.")
        print(f"You can now use '{PREPROCESSED_OUTPUT_PATH}' with your fast evaluation script.")

    print("\n--- Script Finished ---")
