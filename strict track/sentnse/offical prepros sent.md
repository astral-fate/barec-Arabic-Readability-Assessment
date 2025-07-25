```python
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
    Trainer,
    EarlyStoppingCallback
)
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.dediac import dediac_ar
```


```python

# --- Configuration ---
MODEL_NAME = "CAMeL-Lab/readability-arabertv2-d3tok-reg"
NUM_LABELS = 1
TARGET_CLASSES = 19
BASE_DIR = '.'
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "results", f"regression_{MODEL_NAME.split('/')[-1]}")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# --- File Paths ---
BAREC_TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
BAREC_DEV_PATH = os.path.join(DATA_DIR, 'dev.csv')
BLIND_TEST_PATH = os.path.join(DATA_DIR, 'blind_test_data.csv')
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission_regression_final.csv")
ZIPPED_SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission_regression_final.zip")
TRAIN_PREPROCESSED_PATH = os.path.join(DATA_DIR, 'train_preprocessedv2.csv')
DEV_PREPROCESSED_PATH = os.path.join(DATA_DIR, 'dev_preprocessedv2.csv')
```


```python
# --- DATA LOADING AND PREPROCESSING ---

def preprocess_d3tok(text, disambiguator):
    """
    Preprocesses text into the D3Tok format using BERTUnfactoredDisambiguator.
    This version includes robust error handling for missing 'd3tok' keys.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    tokens = simple_word_tokenize(text)
    disambiguated_sentence = disambiguator.disambiguate(tokens)
    d3tok_forms = []
    for disambig_word in disambiguated_sentence:
        if disambig_word.analyses:
            analysis_dict = disambig_word.analyses[0][1]
            # MODIFICATION: Safely check if the 'd3tok' key exists.
            if 'd3tok' in analysis_dict:
                d3tok = dediac_ar(analysis_dict['d3tok']).replace("_+", " +").replace("+_", "+ ")
                d3tok_forms.append(d3tok)
            else:
                # Fallback for analyses that don't have a 'd3tok' key (e.g., punctuation)
                d3tok_forms.append(disambig_word.word)
        else:
            # Fallback for words with no analysis at all
            d3tok_forms.append(disambig_word.word)
    return " ".join(d3tok_forms)
```


```python


def load_or_preprocess_data(disambiguator):
    """
    Loads preprocessed data if it exists, otherwise, it runs preprocessing.
    """
    print("--- Loading BAREC Data ---")
    if os.path.exists(TRAIN_PREPROCESSED_PATH) and os.path.exists(DEV_PREPROCESSED_PATH):
        print("✔ Found preprocessed files. Loading them directly...")
        train_df = pd.read_csv(TRAIN_PREPROCESSED_PATH)
        val_df = pd.read_csv(DEV_PREPROCESSED_PATH)
        train_df['text'] = train_df['text'].astype(str)
        val_df['text'] = val_df['text'].astype(str)
        print(f"Successfully loaded {len(train_df)} training and {len(val_df)} validation records.")
        return train_df, val_df
    else:
        print("Preprocessed files not found. Starting one-time preprocessing...")
        try:
            train_df = pd.read_csv(BAREC_TRAIN_PATH)
            val_df = pd.read_csv(BAREC_DEV_PATH)
            train_df = train_df[['Sentence', 'Readability_Level_19']].rename(
                columns={'Sentence': 'text', 'Readability_Level_19': 'label'})
            val_df = val_df[['Sentence', 'Readability_Level_19']].rename(
                columns={'Sentence': 'text', 'Readability_Level_19': 'label'})
            train_df.dropna(subset=['text', 'label'], inplace=True)
            val_df.dropna(subset=['label', 'text'], inplace=True)
            train_df['text'] = train_df['text'].astype(str)
            val_df['text'] = val_df['text'].astype(str)
            train_df['label'] = train_df['label'].astype(int) - 1
            val_df['label'] = val_df['label'].astype(int) - 1
            train_df['label'] = train_df['label'].astype(float)
            val_df['label'] = val_df['label'].astype(float)
            print(f"Successfully loaded raw data: {len(train_df)} training and {len(val_df)} validation records.")
            print("\n--- Preprocessing Text to D3Tok format (this will only run once) ---")
            train_df['text'] = train_df['text'].apply(lambda x: preprocess_d3tok(x, disambiguator))
            val_df['text'] = val_df['text'].apply(lambda x: preprocess_d3tok(x, disambiguator))
            print("✔ Text preprocessing finished.")
            print("\n--- Saving preprocessed data for future use... ---")
            train_df.to_csv(TRAIN_PREPROCESSED_PATH, index=False)
            val_df.to_csv(DEV_PREPROCESSED_PATH, index=False)
            print(f"** Saved preprocessed files to {TRAIN_PREPROCESSED_PATH} and {DEV_PREPROCESSED_PATH} **")
            return train_df, val_df
        except FileNotFoundError:
            print(f"! ERROR: Raw file not found. Make sure 'train.csv' and 'dev.csv' are in the '{DATA_DIR}' directory.")
            return None, None
        except Exception as e:
            print(f"! ERROR during initial processing: {e}")
            return None, None
```


```python

print("Initializing BERT Disambiguator for preprocessing...")
bert_disambiguator = BERTUnfactoredDisambiguator.pretrained('msa')

train_df, val_df = load_or_preprocess_data(bert_disambiguator)

if train_df is not None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
else:
    print("Stopping script due to data loading failure.")
    exit()

# --- DATASET AND METRICS ---
class ReadabilityDataset(TorchDataset):
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=256)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.encodings.get('input_ids', []))

def compute_metrics(p):
    preds = p.predictions.flatten()
    rounded_preds = np.round(preds)
    clipped_preds = np.clip(rounded_preds, 0, TARGET_CLASSES - 1).astype(int)
    labels = p.label_ids.astype(int)
    qwk = cohen_kappa_score(labels, clipped_preds, weights='quadratic')
    return {"qwk": qwk}

```

    Initializing BERT Disambiguator for preprocessing...
    

    Some weights of the model checkpoint at C:\Users\Fatima\AppData\Roaming\camel_tools\data\disambig_bert_unfactored\msa were not used when initializing BertForTokenClassification: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight']
    - This IS expected if you are initializing BertForTokenClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForTokenClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    

    --- Loading BAREC Data ---
    Preprocessed files not found. Starting one-time preprocessing...
    Successfully loaded raw data: 54845 training and 7310 validation records.
    
    --- Preprocessing Text to D3Tok format (this will only run once) ---
    ✔ Text preprocessing finished.
    
    --- Saving preprocessed data for future use... ---
    ** Saved preprocessed files to .\data\train_preprocessedv2.csv and .\data\dev_preprocessedv2.csv **
    


```python

# --- MODEL TRAINING ---
print("\n===== INITIALIZING REGRESSION MODEL AND TRAINER =====\n")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
train_dataset = ReadabilityDataset(train_df['text'].tolist(), train_df['label'].tolist())
val_dataset = ReadabilityDataset(val_df['text'].tolist(), val_df['label'].tolist())

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="qwk",
    greater_is_better=True,
    save_total_limit=2,
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

print("Starting training...")
trainer.train()
print("✔ Training finished.")


```

    
    ===== INITIALIZING REGRESSION MODEL AND TRAINER =====
    
    

    C:\ProgramData\anaconda3\envs\barec_env\lib\site-packages\transformers\training_args.py:1525: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
      warnings.warn(
    

    Starting training...
    



    <div>

      <progress value='17140' max='20568' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [17140/20568 3:29:39 < 41:56, 1.36 it/s, Epoch 5/6]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Qwk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>1.915000</td>
      <td>3.771299</td>
      <td>0.818270</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.632400</td>
      <td>3.843556</td>
      <td>0.820582</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.016600</td>
      <td>3.922288</td>
      <td>0.822806</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.699200</td>
      <td>3.913797</td>
      <td>0.820187</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.577400</td>
      <td>3.959563</td>
      <td>0.818270</td>
    </tr>
  </tbody>
</table><p>


    ✔ Training finished.
    
    ===== FINAL PREDICTION AND SUBMISSION =====
    
    Preprocessing blind test text to D3Tok format...
    Generating predictions on the test set...
    





    An error occurred during final prediction: "['Sentence ID'] not in index"
    
    --- Script Finished ---
    


```python
# --- FINAL PREDICTION AND SUBMISSION ---
print("\n===== FINAL PREDICTION AND SUBMISSION =====\n")
try:
    test_df = pd.read_csv(BLIND_TEST_PATH)
    test_df.dropna(subset=['Sentence'], inplace=True)
    
    print("Preprocessing blind test text to D3Tok format...")
    # This part is correct because bert_disambiguator was defined in the global scope
    test_df['processed_text'] = test_df['Sentence'].apply(lambda x: preprocess_d3tok(x, bert_disambiguator))
    
    print("Generating predictions on the test set...")
    test_dataset = ReadabilityDataset(test_df['processed_text'].tolist())
    predictions = trainer.predict(test_dataset)
    
    raw_preds = predictions.predictions.flatten()
    rounded_preds = np.round(raw_preds)
    clipped_preds = np.clip(rounded_preds, 0, TARGET_CLASSES - 1)
    
    test_df['Prediction'] = (clipped_preds + 1).astype(int)

    # --- FIX: Use the 'ID' column and rename it to 'Sentence ID' ---
    # The blind test CSV has a column 'ID', not 'Sentence ID'.
    submission_df = test_df[['ID', 'Prediction']]
    # Rename the column to match the required submission format.
    submission_df = submission_df.rename(columns={'ID': 'Sentence ID'})

    print(f"Saving prediction file to: {SUBMISSION_PATH}")
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    
    print(f"\nCompressing {os.path.basename(SUBMISSION_PATH)} into {os.path.basename(ZIPPED_SUBMISSION_PATH)}...")
    with zipfile.ZipFile(ZIPPED_SUBMISSION_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(SUBMISSION_PATH, arcname=os.path.basename(SUBMISSION_PATH))
        
    print(f"✔ Submission file {os.path.basename(ZIPPED_SUBMISSION_PATH)} created successfully.")

except FileNotFoundError:
    print(f"! ERROR: Test file not found. Make sure 'blind_test_data.csv' is in the '{DATA_DIR}' directory.")
except KeyError:
    print("! KEY ERROR: Could not find the 'ID' column in the test data. Please check the blind_test_data.csv file.")
except Exception as e:
    print(f"An error occurred during final prediction: {e}")

print("\n--- Script Finished ---")
```

    
    ===== FINAL PREDICTION AND SUBMISSION =====
    
    Preprocessing blind test text to D3Tok format...
    Generating predictions on the test set...
    





    Saving prediction file to: .\submission\submission_regression_final.csv
    
    Compressing submission_regression_final.csv into submission_regression_final.zip...
    ✔ Submission file submission_regression_final.zip created successfully.
    
    --- Script Finished ---
    
