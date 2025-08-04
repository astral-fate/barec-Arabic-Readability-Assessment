import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import zipfile
import gc

from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
)
from safetensors.torch import load_file
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.dediac import dediac_ar
from tqdm.auto import tqdm

# =====================================================================================
# 1. CONFIGURATION
# =====================================================================================
MODEL_NAME = "CAMeL-Lab/readability-arabertv2-d3tok-reg"
TARGET_CLASSES = 19
NUM_FEATURES = 7 

CHECKPOINT_PATH = r"D:\arabic_readability_project\results\hybrid_constrained_samer_regression_v2_readability-arabertv2-d3tok-reg\checkpoint-24472"
BASE_DIR = r"D:\arabic_readability_project" 
DATA_DIR = os.path.join(BASE_DIR, "data")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")

DOC_BLIND_TEST_PATH = os.path.join(DATA_DIR, 'doc_blind_test_data.csv')
SAMER_LEXICON_PATH = os.path.join(DATA_DIR, 'samer_lexicon.tsv') 

SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission_hybrid_document_v2.csv")
ZIPPED_SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission_hybrid_document_v2.zip")

os.makedirs(SUBMISSION_DIR, exist_ok=True)
print(f"✔️ Configuration loaded. Using checkpoint: {CHECKPOINT_PATH}")


# =====================================================================================
# 2. FEATURE CALCULATION - WITH THE FINAL API FIX
# =====================================================================================
def calculate_features_and_d3tok(sentence_text, disambiguator, lexicon_map):
    if not isinstance(sentence_text, str) or not sentence_text.strip():
        return ([0.0] * NUM_FEATURES, "")

    try:
        tokens = simple_word_tokenize(sentence_text)
        disambiguated_sentence = disambiguator.disambiguate(tokens)

        d3tok_forms = []
        for da in disambiguated_sentence:
            if da.analyses and 'd3tok' in da.analyses[0][1]:
                d3tok_value = da.analyses[0][1]['d3tok']
                if isinstance(d3tok_value, str):
                    d3tok_forms.append(dediac_ar(d3tok_value).replace("_+", " +").replace("+_", "+ "))
            elif isinstance(da.word, str): d3tok_forms.append(da.word)
        d3tok_text = " ".join(d3tok_forms)

        scores = []
        for dw in disambiguated_sentence:
            if dw.analyses:
                analysis = dw.analyses[0][1]
                lemma, pos = analysis.get('lex'), analysis.get('pos')
                if pos and isinstance(lemma, str):
                    score = lexicon_map.get(f"{dediac_ar(lemma)}#{pos}")
                    if score is not None: scores.append(score)
        
        avg_readability = np.mean(scores) if scores else 0.0
        max_readability = np.max(scores) if scores else 0.0

        # !!! ACTION REQUIRED: Add your other 5 features here !!!
        feature_3, feature_4, feature_5, feature_6, feature_7 = 0.0, 0.0, 0.0, 0.0, 0.0
        feature_vector = [avg_readability, max_readability, feature_3, feature_4, feature_5, feature_6, feature_7]
        
        return feature_vector, d3tok_text

    except TypeError as e:
        error_message = f"A TypeError occurred processing sentence: >>>{sentence_text}<<< Original error: {e}"
        raise TypeError(error_message)
    except Exception as e:
        print(f"Warning: An error '{e}' occurred on sentence: '{sentence_text}'. Skipping.")
        return ([0.0] * NUM_FEATURES, "")


# =====================================================================================
# 3. HYBRID MODEL AND DATASET CLASSES (FIXED)
# =====================================================================================

# FIXED: This class now matches the architecture from your training script.
class HybridRegressionModel(nn.Module):
    """
    A hybrid model that combines a transformer base with additional numerical features.
    The output is a single regression value.
    """
    def __init__(self, model_name, num_extra_features):
        super(HybridRegressionModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        # This layer `self.regressor` matches the keys in your checkpoint file.
        self.regressor = nn.Linear(self.transformer.config.hidden_size + num_extra_features, 1)

    # FIXED: The forward pass now expects 'extra_features' to match the dataset.
    def forward(self, input_ids, attention_mask, extra_features, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Using pooler_output as per the training script
        pooler_output = outputs.pooler_output
        
        combined_features = torch.cat((pooler_output, extra_features), dim=1)
        logits = self.regressor(combined_features).squeeze(-1) # Ensure output shape is correct
        
        if labels is not None:
            loss = nn.MSELoss()(logits, labels.float())
            return (loss, logits)
        return logits

# FIXED: This class now provides a dictionary key that matches the model's forward() signature.
class ReadabilityDataset(TorchDataset):
    def __init__(self, texts, features, labels=None, tokenizer_obj=None, max_len=256):
        self.texts=texts; self.features=features; self.labels=labels
        self.tokenizer=tokenizer_obj; self.max_len=max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        feature_vec = torch.tensor(self.features[idx], dtype=torch.float)
        
        inputs = self.tokenizer.encode_plus(
            text, 
            None, 
            add_special_tokens=True, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True
        )
        
        item = {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            # The key is now 'extra_features', which the new model's forward pass expects.
            'extra_features': feature_vec  
        }
        
        if self.labels is not None: 
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            
        return item

# =====================================================================================
# 4. PREDICTION LOGIC
# =====================================================================================

def generate_document_predictions(checkpoint_path):
    print("\n===== 🚀 STARTING HYBRID DOCUMENT PREDICTION PIPELINE =====\n")
    try:
        print("Initializing Tokenizer and Disambiguator...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        disambiguator = BERTUnfactoredDisambiguator.pretrained('msa')
        
        print(f"Loading SAMER Lexicon from: {SAMER_LEXICON_PATH}")
        lexicon_df = pd.read_csv(SAMER_LEXICON_PATH, sep='\t')
        lexicon_map = lexicon_df.set_index('lemma#pos')['readability (rounded average)'].to_dict()
        
        print(f"Loading model weights from checkpoint: {checkpoint_path}")
        # Initialize the CORRECT model class
        model = HybridRegressionModel(MODEL_NAME, num_extra_features=NUM_FEATURES)
        
        # Load the state dict. This should now work without errors.
        model.load_state_dict(load_file(os.path.join(checkpoint_path, "model.safetensors")))
        print("✔ All models and data loaded successfully.")
        
        doc_test_df = pd.read_csv(DOC_BLIND_TEST_PATH)
        doc_test_df.dropna(subset=['ID', 'Sentences'], inplace=True)
        
        print("\nProcessing documents: this will take time...")
        rows_for_df = []
        for _, row in tqdm(doc_test_df.iterrows(), total=len(doc_test_df), desc="Processing Documents"):
            doc_id = row['ID']
            full_text = row['Sentences']
            if isinstance(full_text, str) and full_text.strip():
                sentences_list = [s.strip() for s in full_text.split('\n') if s.strip()]
                for sentence in sentences_list:
                    features, processed_text = calculate_features_and_d3tok(sentence, disambiguator, lexicon_map)
                    rows_for_df.append({'doc_id': doc_id, 'features': features, 'processed_text': processed_text})

        if not rows_for_df: raise ValueError("No sentences could be extracted.")
        sentence_df = pd.DataFrame(rows_for_df)
        print(f"✔ Successfully created {len(sentence_df)} sentences with features.")

        trainer = Trainer(model=model, args=TrainingArguments(output_dir="./temp_results", per_device_eval_batch_size=32, report_to="none"))
        
        print("\nGenerating predictions for all sentences...")
        # Initialize the CORRECT dataset class
        test_dataset = ReadabilityDataset(texts=sentence_df['processed_text'].tolist(), features=sentence_df['features'].tolist(), tokenizer_obj=tokenizer)
        raw_predictions = trainer.predict(test_dataset)
        sentence_df['raw_prediction'] = raw_predictions.predictions.flatten()

        print("Aggregating results...")
        doc_predictions = sentence_df.groupby('doc_id')['raw_prediction'].max()
        
        clipped_preds = np.clip(np.round(doc_predictions.values), 0, TARGET_CLASSES - 1)
        final_labels = (clipped_preds + 1).astype(int)
        
        submission_df = pd.DataFrame({'Sentence ID': doc_predictions.index, 'Prediction': final_labels})
        final_submission_df = pd.DataFrame({'Sentence ID': doc_test_df['ID']}).merge(submission_df, on='Sentence ID', how='left')
        final_submission_df['Prediction'] = final_submission_df['Prediction'].fillna(1).astype(int)

        print(f"\nSaving prediction file to: {SUBMISSION_PATH}")
        final_submission_df.to_csv(SUBMISSION_PATH, index=False)
        
        with zipfile.ZipFile(ZIPPED_SUBMISSION_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(SUBMISSION_PATH, arcname=os.path.basename(SUBMISSION_PATH))
        
        print(f"\n--- ✅ SUCCESS! Submission file '{os.path.basename(ZIPPED_SUBMISSION_PATH)}' created. ---")

    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
    finally:
        gc.collect()
        if 'model' in locals(): del model
        if 'trainer' in locals(): del trainer
        torch.cuda.empty_cache()

# =====================================================================================
# 5. EXECUTE THE SCRIPT
# =====================================================================================
if __name__ == "__main__":
    generate_document_predictions(CHECKPOINT_PATH)
