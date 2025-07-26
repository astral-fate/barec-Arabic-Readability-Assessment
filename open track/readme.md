# summary of old custume pre-prossesing method
<details>
   The Core Strategy: A Hybrid "Two-Brain" Approach

The most important thing to understand is that this isn't a standard, simple model. It uses a hybrid architecture, which is like giving the AI two different ways to "think" about a sentence.

Brain #1: The Deep Language Expert (The Transformer)

This is the aubmindlab/araelectra-base-discriminator model. Think of it as an expert that has read a massive amount of Arabic text. It understands grammar, context, sentence structure, and the subtle nuances of the language.

Its Job: To read the raw sentence and produce a rich numerical representation (an "embedding") that captures its deep meaning.

Brain #2: The Data Scientist (The Feature Engineer)

This part of the code uses the SAMER Lexicon (the .tsv file with word difficulties) to calculate simple, powerful facts about the sentence.

Its Job: To answer questions like: How many words are in this sentence? What's the average difficulty of these words? How many "hard" words (level 4 or 5) does it contain? How many words are so rare they aren't even in the lexicon? These facts are turned into a list of numbers.

The Magic: The code then combines the outputs of both brains. It takes the deep understanding from the Transformer and adds the hard facts from the Feature Engineer. This combined information is then fed into a final decision-making layer to produce the readability score. This hybrid approach is often much more powerful than using either method alone.

Step-by-Step Breakdown of the Code's Actions

Here is the entire process in the order the script executes it:

Step 1: Setup and Configuration

It installs all the necessary Python libraries (transformers, pandas, etc.).

It connects to the Google Drive.

It defines all the important variables, like the model name (araelectra), the number of training folds (N_SPLITS = 5), and the file paths for all the datasets inside the BAREC_Competition folder.

Step 2: Data Loading and Unification

This is one of the most critical parts for the Open Track.

Load BAREC Data (from the CSVs): It reads the train.csv and dev.csv files. Since these files are at the document level, the code intelligently splits the text in the "Sentences" column into individual sentences. Each sentence is given the readability score of the document it came from. This creates the primary, high-quality training data.

Load SAMER Corpus Data (for augmentation): It reads the samer_train.tsv file. This data has levels L3, L4, and L5. The code performs a clever trick called data augmentation: it maps these simple levels to the 1-19 scale (e.g., L3 becomes level 4, L5 becomes level 16). This gives the model thousands of extra sentences to learn from, even if their labels are approximate.

Combine and Shuffle: It mixes the BAREC sentences and the SAMER sentences into one giant training dataset and shuffles it randomly.

Step 3: Feature Engineering with the SAMER Lexicon

The code loads the SAMER-Readability-Lexicon-v2.tsv and turns it into a fast lookup dictionary.

For every single sentence in both the training data and the test data, it calculates 7 numerical features:

Character count

Word count

Average word length

Average word readability score (from the lexicon)

The readability score of the hardest word in the sentence

The number of "difficult" words (score > 4)

The percentage of words not found in the lexicon (a measure of rarity)

Step 4: The K-Fold Cross-Validation Loop

Instead of training just one model, the script uses a robust technique called K-Fold Cross-Validation to build 5 different models.

It splits the entire training dataset into 5 "folds" (or chunks).

It then runs a loop 5 times. In each loop:

It uses 4 of the folds for training and 1 fold for validation.

It initializes a brand new hybrid model.

It trains this model, evaluating it after each epoch and saving only the best version (the one with the highest QWK score on the validation set).

Once the model for that fold is trained, it uses it to make predictions on the official test set. These predictions are stored.

It then cleans up the memory and starts the next fold with a different chunk of data for validation.

Step 5: Ensembling and Creating the Submission File

This is the final step where it combines the results.

Ensembling: You now have 5 different sets of predictions for the test data (one from each of the 5 models). Instead of just picking one, the code averages the predictions for each test sentence. This "wisdom of the crowd" approach almost always produces a more accurate and stable final result.

Final Touches: The averaged predictions (which are decimals) are rounded to the nearest whole number and clipped to ensure they are between 1 and 19.



___

# data pre-processing  

1. Text Normalization (Cleaning)

This is the first and most fundamental step. Raw text from the internet or different sources is often "messy". The goal here is to make it clean and consistent. The script uses the ArabertPreprocessor library, which is specifically designed for Arabic text and performs several cleaning actions:

Removes Diacritics/Tashkeel (التشكيل): It removes vowels and other markings like fatha (ـَ), damma (ـُ), kasra (ـِ), and shadda (ـّ). For most modern Transformer models, this helps reduce the complexity of the vocabulary without losing too much meaning.

Example: الْعَرَبِيَّةُ becomes العربية

Normalizes Alef Variants: It standardizes different forms of the letter Alef (أ, إ, آ) into a single form (ا). This prevents the model from treating words like "أحمد" and "احمد" as completely different words.

Example: أحمد إبراهيم becomes احمد ابراهيم

Normalizes Yaa and Taa Marbuta: It converts the final Yaa (ى) to (ي) and the Taa Marbuta (ة) to Haa (ه). This is a common normalization step.

Example: مَدْرَسَةٌ فِي القَرْيَةِ becomes مدرسه في القريه

Removes Repetitive Characters: It reduces elongated characters used for emphasis.

Example: جمييييييل becomes جميل

Removes Punctuation and Special Characters: It strips out commas, periods, question marks, etc., leaving only the core words.

Why is this done? To simplify the text and reduce the vocabulary size the model has to learn. It makes the model more robust by treating slightly different writings of the same word as identical.

2. Feature Engineering (Creating New Information)

This is where the script acts like a data scientist. Instead of just giving the model the text, it extracts explicit numerical facts (features) about each sentence using the SAMER Lexicon. This gives the model extra clues about readability that might not be obvious from the text alone.

For every single sentence, it calculates these 7 features:

len(text) - Character Count: A simple count of the total number of characters. Longer sentences are often harder to read.

len(words) - Word Count: The total number of words. This is a classic readability metric.

np.mean([len(w) for w in words]) - Average Word Length: The average number of characters per word. Sentences with longer, more complex words (e.g., "استنتاجات") are generally harder than sentences with short words (e.g., "بيت").

np.mean(word_difficulties) - Average Word Readability Score: This is a very powerful feature. The script looks up every word in the SAMER Lexicon, gets its 1-5 difficulty score, and then calculates the average score for the entire sentence. A higher average indicates a more difficult sentence.

np.max(word_difficulties) - Maximum Word Readability Score: This feature captures the difficulty of the single hardest word in the sentence. A sentence might be simple overall but contain one very difficult word (e.g., a scientific term) that makes it hard to understand.

np.sum(np.array(word_difficulties) > 4) - Count of "Hard" Words: This counts how many words in the sentence have a difficulty score of 5 (the highest). This helps the model identify sentences with a lot of advanced vocabulary.

len([w for w in words if w not in lexicon]) / len(words) - Out-of-Vocabulary (OOV) Rate: This calculates the percentage of words in the sentence that are so rare they don't even appear in the 40k-word SAMER Lexicon. A high OOV rate is a strong signal that the sentence contains very specialized or uncommon terminology, making it difficult.

Why is this done? Transformers are great at understanding context, but they can sometimes miss these simple, powerful signals. By feeding these numbers directly into the model, we are explicitly telling it: "Pay attention! This sentence has long words and a high average difficulty."

3. Tokenization (Translating for the AI)

This is the final step to prepare the data for the Transformer model (araelectra). A Transformer doesn't read words; it reads numbers. Tokenization is the process of converting the cleaned text into a sequence of numbers.

WordPiece Tokenization: The script uses a Tokenizer that breaks words down into common sub-word units. For example, a complex word might be broken into a stem and a suffix.

Example: The word المكتبات (libraries) might be tokenized into [ال, مكتب, ات].

Converting to IDs: Each of these sub-word pieces has a unique number (ID) in the model's vocabulary. The tokenizer converts the sequence of pieces into a sequence of numbers.

Example: [ال, مكتب, ات] might become [4, 2590, 778].

Padding and Truncation: All sentences must be the same length to be processed in batches. The tokenizer ensures this by:

Padding: Adding a special [PAD] token (usually with ID 0) to the end of shorter sentences.

Truncation: Cutting off sentences that are longer than the maximum length (set to 256).

Adding Special Tokens: It adds [CLS] at the beginning (a token the model uses to understand the whole sentence) and [SEP] at the end (a separator token).

Creating an Attention Mask: This is a list of 1s and 0s that tells the model which tokens are real words (pay attention to these) and which are just padding (ignore these).

Why is this done? This is the mandatory final step to format the text into the exact numerical input that the Transformer model was designed to accept.
Saving the File: The final predictions are saved into a file named submission_..._hybrid_csv.csv in the Google Drive, perfectly formatted with the required "Sentence ID" and "Prediction" columns.

In summary, the script executes a complete, professional-level machine learning pipeline that intelligently combines multiple data sources, engineers custom features, and uses robust training and ensembling techniques to create the best possible submission file for the competition.



______________




### Data Preprocessing Steps

the model performs a sophisticated, three-stage preprocessing pipeline designed to clean, enrich, and format the Arabic text for the model.

1.  **Text Normalization (Cleaning)**: The first stage cleans the raw text to ensure consistency. The `ArabertPreprocessor` library handles this automatically by:
    * Removing diacritics (Tashkeel) like fatha and damma.
    * Standardizing different forms of the letter Alef (أ, إ, آ) into a single form (ا).
    * Normalizing the final Yaa (ى) to (ي) and Taa Marbuta (ة) to Haa (ه).
    * Reducing elongated characters (e.g., جمييييل becomes جميل).

2.  **Feature Engineering**: The script then extracts 7 explicit numerical features from each sentence using the SAMER Lexicon to give the model extra clues about readability. These features are:
    * Total character count.
    * Total word count.
    * Average word length.
    * Average word readability score (looking up each word in the lexicon).
    * The maximum readability score of the single hardest word in the sentence.
    * A count of "difficult" words (those with a readability score greater than 4).
    * The percentage of words so uncommon they are not found in the SAMER lexicon.

3.  **Tokenization**: The final step translates the cleaned text and engineered features into a numerical format the AI can understand. This involves:
    * Using a WordPiece tokenizer to break words into common sub-word units.
    * Converting these sub-word pieces into unique numerical IDs from the model's vocabulary.
    * Ensuring all sentences have a uniform length by padding shorter sentences and truncating longer ones.
    * Adding special tokens like `[CLS]` (start of sentence) and `[SEP]` (end of sentence).
    * Creating an "attention mask" to tell the model to focus on real tokens and ignore padding.

***

### Implementation Verification



* ✅ **Text Normalization**: This is implemented. The script initializes the preprocessor with `arabert_preprocessor = ArabertPreprocessor(model_name=MODEL_NAME)` and then applies it to the 'text' column of the train, validation, and test dataframes with the line: `df['text'] = df['text'].apply(arabert_preprocessor.preprocess)`.

* ✅ **Feature Engineering**: This is fully implemented in the `get_lexical_features` function. The code in that function calculates the exact 7 features described in the documentation, which are then added as a new 'features' column to each dataframe.

* ✅ **Tokenization**: This is fully implemented within the `ReadabilityDataset` class. The line `self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=256)` uses the Hugging Face tokenizer to perform all the described tokenization steps—converting text to IDs, padding, truncating, adding special tokens, and creating an attention mask—in a single, efficient operation.

</details>



# Data distubution os dares dataset

merges train dev test =  13868 DARES records

## experment 1: unbalanced  data distubution

- train: barec + 100% samer = 68715
- dev: 100 barec, 0% dares = 7310
  ### result
  - dev: 63
  - test: 50
 
## experment 2: balanced  data distubution
 
 
13868 DARES records in the merged file.
- Performing stratified split on the DARES data to select 15.0% for the dev set...
✔ DARES data split successfully!
  - 2081 records will be added to the dev set.
  - 11787 records will remain in the training set.

- New training set (66634 records)
- New development set (9391 records)

   ## training params
  ```
  training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="qwk",
    greater_is_better=True,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none"
  ```

  ## results
  ```
   Scores:
   {'accuracy': 8.2, 'accuracy+-1': 32.7, 'avg_abs_dist': 2.7, 'qwk': 55.6, 'accuracy_7': 20.7, 'accuracy_5': 37.7, 'accuracy_3': 53.0}
   ```


## experment 3: using only offoical train and dev  data distubution

1. Loading BAREC data...
  - Loaded 54845 BAREC training records.
  - Loaded 7310 BAREC validation records.

2. Loading and mapping dares data...
  - Loaded and mapped 9703 Osman training records.
  - Loaded and mapped 1380 Osman validation records.

3. Combining BAREC and dares datasets...
  - Combined training data size: 64548 records.
  - Combined validation data size: 8690 records.

### result

- dev: 81
- test: 82.9
