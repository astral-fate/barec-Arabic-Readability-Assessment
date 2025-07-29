# barec-Arabic-Readability-Assessment

BAREC Shared Task 2025
Arabic Readability Assessment
The Third Arabic Natural Language Processing Conference (ArabicNLP 2025) @ EMNLP 2025
https://barec.camel-lab.com/sharedtask2025


## results

Please find the information from the image organized into a Markdown table below:

| Model                | Strict (Sentence) | Strict (Sentence) | Constrained (Sentence) | Constrained (Sentence) | Open (Sentence) | Open (Sentence) | Strict (Document) | Strict (Document) | Constrained (Document) | Constrained (Document) | Open (Document) | Open (Document) |
|----------------------|-------------------|-------------------|------------------------|------------------------|-----------------|-----------------|-------------------|-------------------|------------------------|------------------------|-----------------|-----------------|
|                      | dev               | test              | dev                    | test                   | dev             | test            | dev               | test              | dev                    | test                   | dev             | test            |
| **Official progress Regression** | 0.822806          | 84.2              | 0.8103                 | 82.9                   | 82.7            | 83.6            | model as trained on senses 0.822806 | 79.9              | Same as train (83.5)   | 75.50                  | 82.7 | 79.2            |




# abilation study


| Model | Strict - Sentence (dev) | Strict - Sentence (test) | Constrained - Sentence (dev) | Constrained - Sentence (test) | Open - Sentence (dev) | Open - Sentence (test) | Strict - Document (dev) | Strict - Document (test) | Constrained - Document (dev) | Constrained - Document (test) | Open - Document (dev) | Open - Document (test) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CAMeL-Lab/readability-araberty02-word-CE** | 0.749514 | 78.20, {'accuracy': 53.0, 'accuracy+1': 68.2, 'avg_abs_dist': 1.2, 'wae': 68.2, 'accuracy_7': 63.3, 'accuracy_5': 67.8, 'accuracy_3': 74.2} | 69 | 72.20 | test | test | test | test | test | test | test | test |
| **CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment** | test | **82.70** | 78.50 | 79.60 | test | current | test | test | test | test | test | test |
| **Arabert base v2** | 81 | 82.60 {'accuracy': 59.6, 'accuracy+1': 78.6, 'avg_abs_dist': 1.8, 'wae': 68.5, 'accuracy_7': 68.5, 'accuracy_5': 72.2, 'accuracy_3': 77.8} | test | 78 | 78 | 44 | Osman 79.6 {'accuracy': 53.8, 'accuracy+1': 68.0, 'avg_abs_dist': 1.4, 'wae': 79.6, 'accuracy_7': 63.8, 'accuracy_5': 68.8, 'accuracy_3': 73.7} | test | Samer @ Osman 73.1, Suares-{'accuracy': 45.8, 'accuracy+1': 59.3, 'avg_abs_dist': 1.5, 'wae': 73.1, 'accuracy_7': 56.2, 'accuracy_5': 62.1, 'accuracy_3': 70.2} | test | test | test |



# refrences

https://arbml.github.io/masader/search

https://github.com/drelhaj/OsmanReadability?tab=readme-ov-file 

https://huggingface.co/aubmindlab/bert-base-arabertv2

https://github.com/CAMeL-Lab/barec_analyzer/tree/main
