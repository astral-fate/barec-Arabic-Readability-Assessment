# barec-Arabic-Readability-Assessment

BAREC Shared Task 2025
Arabic Readability Assessment
The Third Arabic Natural Language Processing Conference (ArabicNLP 2025) @ EMNLP 2025
https://barec.camel-lab.com/sharedtask2025



# Strict Track
## Task 1: Sentence-level Readability Assessment
Given an Arabic sentence, predict its readability level on a scale from 1 to 19, indicating the degree of reading difficulty. Here are some examples of sentences 19 BAREC readability levels:
Given an Arabic sentence, predict its readability level on a scale from 1 to 19, indicating the degree of reading difficulty.


For the strict track, models must be trained exclusively on the BAREC Corpus.

Data:
BAREC Corpus: The BAREC Corpus (Elmadani et al., 2025) consists of 1,922 documents and 69,441 sentences classified into 19 readability levels.

Phases:
Development Phase: This phase will run until July 20, 2025. Participants will build their models and submit predictions on the BAREC Test set, which is publicly available (i.e, Open Test). Submitting predictions in this phase does not require registration for the shared task. However, please note that doing so does not make you an official participant in the shared task. To be officially considered, you must register and submit your predictions during the Testing Phase.

Testing Phase: This phase will run from July 20, 2025 to July 25, 2025. Participants will upload their predictions on the Official Blind Test set (henceforth Blind Test). The Blind Test set will only be available to participants who registered to participate in the shared task.

Shared Task Registration Form:
By registering to participate in the shared task and receiving access to the Official Blind Test set, you commit to submitting a description paper. Participants who register but fail to submit a paper may be disqualified from future shared tasks. Checkout the paper submission guidelines for more information.

Contact:

dataset: https://huggingface.co/datasets/CAMeL-Lab/BAREC-Shared-Task-2025-doc/tree/main


BAREC Shared Task 2025 on Arabic Readability Assessment
The BAREC Shared Task 2025 focuses on fine-grained readability classification across 19 levels using the Balanced Arabic Readability Evaluation Corpus (BAREC), a dataset of over 1 million words.

## Task 2: Document-level Readability Assessment
Given a document consisting of multiple sentences, predict its readability level on a scale from 1 to 19, where the hardest (i.e., highest readability) sentence in the document determines the overall document readability level.

Strict Track
For the strict track, models must be trained exclusively on the BAREC Corpus.

Data:
BAREC Corpus: The BAREC Corpus (Elmadani et al., 2025) consists of 1,922 documents and 69,441 sentences classified into 19 readability levels.

Phases:
Development Phase: This phase will run until July 20, 2025. Participants will build their models and submit predictions on the BAREC Test set, which is publicly available (i.e, Open Test). Submitting predictions in this phase does not require registration for the shared task. However, please note that doing so does not make you an official participant in the shared task. To be officially considered, you must register and submit your predictions during the Testing Phase.

Testing Phase: This phase will run from July 20, 2025 to July 25, 2025. Participants will upload their predictions on the Official Blind Test set (henceforth Blind Test). The Blind Test set will only be available to participants who registered to participate in the shared task.

Shared Task Registration Form:
By registering to participate in the shared task and receiving access to the Official Blind Test set, you commit to submitting a description paper. Participants who register but fail to submit a paper may be disqualified from future shared tasks. Checkout the paper submission guidelines for more information.

codabench: https://www.codabench.org/competitions/9086/

dataset: https://huggingface.co/datasets/CAMeL-Lab/BAREC-Shared-Task-2025-doc/tree/main

# Constrained Track

## Task 1: Sentence-level Readability Assessment 
The BAREC Shared Task 2025 focuses on fine-grained readability classification across 19 levels using the Balanced Arabic Readability Evaluation Corpus (BAREC), a dataset of over 1 million words.

Task 1: Sentence-level Readability Assessment
Given an Arabic sentence, predict its readability level on a scale from 1 to 19, indicating the degree of reading difficulty.

Constrained Track
For the constrained track, participants may use the BAREC Corpus, SAMER Corpus (including document, fragment, and word-level annotations), and the SAMER Lexicon.

https://www.codabench.org/competitions/9083/

dataset: https://huggingface.co/collections/CAMeL-Lab/readability-6846ed8acb652c8d82aecd2a


## task 2 Sentence-level Readability Assessment:

BAREC Shared Task 2025 on Arabic Readability Assessment
The BAREC Shared Task 2025 focuses on fine-grained readability classification across 19 levels using the Balanced Arabic Readability Evaluation Corpus (BAREC), a dataset of over 1 million words.

Task 2: Document-level Readability Assessment
Given a document consisting of multiple sentences, predict its readability level on a scale from 1 to 19, where the hardest (i.e., highest readability) sentence in the document determines the overall document readability level.

Strict Track
For the strict track, models must be trained exclusively on the BAREC Corpus.

Data:
BAREC Corpus: The BAREC Corpus (Elmadani et al., 2025) consists of 1,922 documents and 69,441 sentences classified into 19 readability levels.


https://www.codabench.org/competitions/9083/

dataset: https://huggingface.co/collections/CAMeL-Lab/readability-6846ed8acb652c8d82aecd2a



# Open Track
No restrictions on external resources, allowing the use of any publicly available data.

## Sentence-level Readability Assessment:
Given an Arabic sentence, predict its readability level on a scale from 1 to 19, indicating the degree of reading difficulty.

Open Track
For the Open track, participants may use any publicly available data.

Data:
BAREC Corpus: The BAREC Corpus (Elmadani et al., 2025) consists of 1,922 documents and 69,441 sentences classified into 19 readability levels.

The SAMER Corpus: The SAMER Corpus (Alhafni et al., 2024) consists of 4,289 documents and 20,358 fragments classified into three readability levels.

The SAMER Lexicon: The SAMER Lexicon (Al Khalil et al., 2020) is a 40K-lemma leveled readability lexicon. The lexicon consists of 40K lemma and part-of-speech pairs annotated into five readability levels.

Any other publicly available resource.


https://www.codabench.org/competitions/9085/


dataset: https://huggingface.co/collections/CAMeL-Lab/readability-6846ed8acb652c8d82aecd2a
o https://camel.abudhabi.nyu.edu/samer-simplification-corpus/
https://camel.abudhabi.nyu.edu/samer-readability-lexicon/

## Document-level Readability Assessment:

Given a document consisting of multiple sentences, predict its readability level on a scale from 1 to 19, where the hardest (i.e., highest readability) sentence in the document determines the overall document readability level.

dataset: https://huggingface.co/collections/CAMeL-Lab/readability-6846ed8acb652c8d82aecd2a
o https://camel.abudhabi.nyu.edu/samer-simplification-corpus/
https://camel.abudhabi.nyu.edu/samer-readability-lexicon/


# refrences

https://arbml.github.io/masader/search

https://github.com/drelhaj/OsmanReadability?tab=readme-ov-file 

https://huggingface.co/aubmindlab/bert-base-arabertv2

https://github.com/CAMeL-Lab/barec_analyzer/tree/main
