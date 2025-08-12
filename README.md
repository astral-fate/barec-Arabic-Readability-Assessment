#  MorphoArabia at BAREC 2025 Shared Task: A Hybrid Architecture with Morphological Analysis for Arabic Readability Assessmen

<p align="center">
<img src="https://placehold.co/800x200/dbeafe/3b82f6?text=Barec-Readability-Assessment" alt="Barec Readability Assessment">
</p>


This repository contains the official models and results for **MorphoArabia**, the submission to the **[BAREC 2025 Shared Task](https://www.google.com/search?q=https://sites.google.com/view/barec-2025/home)** on Arabic Readability Assessment.

#### By: [Fatimah Mohamed Emad Elden](https://scholar.google.com/citations?user=CfX6eA8AAAAJ&hl=ar)

#### *Cairo University*


[![Paper](https://img.shields.io/badge/arXiv-25XX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/25XX.XXXXX)
[![Code](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/astral-fate/barec-Arabic-Readability-Assessment)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Page-F9D371)](https://huggingface.co/collections/FatimahEmadEldin/barec-shared-task-2025-689195853f581b9a60f9bd6c)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](https://github.com/astral-fate/mentalqa2025/blob/main/LICENSE)

---

## Model Description

This project introduces a **morphologically-aware approach** for assessing the readability of Arabic text. The system is built around a fine-tuned regression model designed to process morphologically analyzed text. For the **Constrained** and **Open** tracks of the shared task, this core model is extended into a hybrid architecture that incorporates seven engineered lexical features.

A key element of this system is its deep morphological preprocessing pipeline, which uses the **CAMEL Tools d3tok analyzer**. This allows the model to capture linguistic complexities that are often missed by surface-level tokenization methods. This approach proved to be highly effective, achieving a peak **Quadratic Weighted Kappa (QWK) score of 84.2** on the strict sentence-level test set.

The model predicts a readability score on a **19-level scale**, from 1 (easiest) to 19 (hardest), for a given Arabic sentence or document.

-----

## 🚀 How to Use

You can use the fine-tuned models directly with the `transformers` library pipeline for `text-regression`. The following example uses the best-performing model from the **Strict** track.

```python
from transformers import pipeline

# Load the regression pipeline
# This model is the best performer for the Strict track
# It's also the base model for the other tracks.
regressor = pipeline(
    "text-regression",
    model="FatimahEmadEldin/MorphoArabia-CAMEL-BERT-BAREC-Strict-Sentence"
)

# Example sentence in Arabic
sentence = "أليست هذه العاطفة التي نخافها ونرتجف لمرورها في صدورنا جزءا من الناموس الكلي"
# (Translation: "Isn't this emotion, which we fear and tremble at its passing in the chests, a part of the universal law?")

# Get the readability score
results = regressor(sentence)

# The output is a score between 1 and 19
predicted_score = results[0]['score']

print(f"Sentence: {sentence}")
print(f"Predicted Readability Score: {predicted_score:.2f}")

```

-----

## ⚙️ Training Procedure

The system employs two distinct architectures based on the track's constraints:

  * **Strict Track**: This track uses a base regression model, `CAMeL-Lab/readability-arabertv2-d3tok-reg`, fine-tuned directly on the BAREC dataset.
  * **Constrained and Open Tracks**: These tracks utilize a hybrid model. This architecture combines the deep contextual understanding of the Transformer with explicit numerical features. The final representation for a sentence is created by concatenating the Transformer's `[CLS]` token embedding with a 7-dimensional vector of engineered lexical features derived from the SAMER lexicon.

A critical component of the system is its preprocessing pipeline, which leverages the CAMEL Tools `d3tok` format. The `d3tok` analyzer performs a deep morphological analysis by disambiguating words in context and then segmenting them into their constituent morphemes.

### Frameworks

  * PyTorch
  * Hugging Face Transformers

-----

### 📊 Evaluation Results

The models were evaluated on the blind test set provided by the BAREC organizers. The primary metric for evaluation is the **Quadratic Weighted Kappa (QWK)**, which penalizes larger disagreements more severely.

#### Final Test Set Scores (QWK)

| Track | Task | Dev (QWK) | Test (QWK) |
| :--- | :--- | :---: | :---: |
| **Strict** | Sentence | 0.823 | **84.2** |
| | Document | 0.823\* | 79.9 |
| **Constrained** | Sentence | 0.810 | 82.9 |
| | Document | 0.835\* | 75.5 |
| **Open** | Sentence | 0.827 | 83.6 |
| | Document | 0.827\* | **79.2** |

\*Document-level dev scores are based on the performance of the sentence-level model on the validation set.

-----

## 📜 Citation

If you use the work, please cite the paper:

```
@inproceedings{eldin2025morphoarabia,
    title={{MorphoArabia at BAREC 2025 Shared Task: A Hybrid Architecture with Morphological Analysis for Arabic Readability Assessmen}},
    author={Eldin, Fatimah Mohamed Emad},
    year={2025},
    booktitle={Proceedings of the BAREC 2025 Shared Task},
    eprint={25XX.XXXXX},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
