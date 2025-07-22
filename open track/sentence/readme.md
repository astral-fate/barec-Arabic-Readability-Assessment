
[cite\_start]Here is the correct grade mapping, according to the `dataset_card.md` file from the repository you provided[cite: 1]:

| Grade Label | Meaning | Corresponding School Grades |
| :--- | :--- | :--- |
| **BE** | Beginner Elementary | Grades 1-2 |
| **IE** | Intermediate Elementary | Grades 3-4 |
| **AE** | Advanced Elementary | Grades 5-6 |
| **BM** | Beginner Middle | Grades 7-8 |
| **IM** | Intermediate Middle | Grade 9 |
| **AM** | Advanced Middle | Grade 10 |
| **BH** | Beginner High | Grade 11 |
| **IH** | Intermediate High | Grade 12 |

### How to Convert This to BAREC's 1-19 Scale

The BAREC competition uses a 1-19 scale that corresponds to grade levels. This makes the conversion from the new dataset straightforward. You can now create a precise mapping. When a label corresponds to a range of grades (like "Grades 1-2"), the standard practice is to take the average.

Here is the recommended mapping to use in your Python script:

```python
# This is the correct, evidence-based mapping derived from the dataset card.
grade_to_barec_map = {
    'BE': 1.5,  # Average of grades 1 and 2
    'IE': 3.5,  # Average of grades 3 and 4
    'AE': 5.5,  # Average of grades 5 and 6
    'BM': 7.5,  # Average of grades 7 and 8
    'IM': 9.0,
    'AM': 10.0,
    'BH': 11.0,
    'IH': 12.0
}

# In your data loading code, you would then apply this map.
# After mapping, you should round the labels to the nearest integer
# to match the BAREC format (1, 2, 3, ... 19).

# Example:
# external_df['label'] = external_df['Grade'].map(grade_to_barec_map)
# external_df.dropna(subset=['label'], inplace=True)
# external_df['label'] = external_df['label'].round().astype(int)
```

By using this evidence-based mapping, you are ensuring that the labels from your external dataset are accurately aligned with the BAREC task, which should significantly improve your model's performance.

# resouse

https://github.com/DamithDR/arabic-readability-assessment
