# 1`st attempt
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


# 2nd attempt

Of course. Let's walk through the mapping process step-by-step, showing exactly how the computer transforms the data. It's a two-stage process: first, a **human** decides the mapping logic, and second, the **computer** executes that logic on every row.

-----

### \#\# Stage 1: The Human Logic (The "Recipe")

Before writing any code, we first look at the data and create a clear rule.

1.  **Identify the Goal**: We want to create a new column called **`label`** that contains a number from 1 to 19, which represents the BAREC readability level.

2.  **Find the Source Information**: We look at the columns in our extra dataset (`Annotated_Paper_Dataset.csv`) to find the best source for this number. We see two options:

      * **`Coarse-grained` column**: Has values like `EE`, `IE`, `ME`. This is complicated. We would need a separate lookup table (like the image you showed) to convert `EE` to a number.
      * **`Fine-grained` column**: Has values like `G1`, `G2`, `G3`, `G11`, `G12`. This is much simpler\! The grade number is already right there in the text.

3.  **Define the Rule**: We decide on the most direct rule possible: *"To get the label, take the string from the `Fine-grained` column, remove the 'G' from the beginning, and use the number that's left."*

This simple rule is the "recipe" we will give to the computer.

-----

### \#\# Stage 2: The Computer's Execution (Following the Recipe)

Now, we translate our human rule into a single line of Python code that uses the **pandas** library. The computer will apply this line to every single one of the 13,870 rows in the file.

Let's trace the execution for a few example rows:

#### **The Code:**

```python
# This is the line that does all the work
ext_df['label'] = ext_df['Fine-grained'].str.replace('G', '', regex=False).astype(int)
```

Let's break down this line into its three parts:

| Code Part | What It Does |
| :--- | :--- |
| `ext_df['Fine-grained']` | **Selects the Column**: Tells the computer to look only at the `Fine-grained` column. |
| `.str.replace('G', '', regex=False)` | **Replaces Text**: The `.str` accessor treats the column like text. The `replace()` function finds every occurrence of the letter 'G' and replaces it with nothing (`''`), effectively deleting it. |
| `.astype(int)` | **Converts to Number**: The result from the replacement is still text (e.g., "5"). This function converts that text into a whole number (an integer), so the computer can use it for math and model training. |

-----

#### **Row-by-Row Example Walkthrough:**

Imagine the computer processing the first few rows of your dataset:

**Row 1:**

  * **1. Select**: The computer reads the value in the `Fine-grained` column: `"G1"`.
  * **2. Replace**: It applies `.str.replace('G', '')`. The string `"G1"` becomes `"1"`.
  * **3. Convert**: It applies `.astype(int)`. The text `"1"` becomes the number `1`.
  * **4. Assign**: It creates the new `label` column for this row and puts the number `1` into it.

**Row 4 (Example with a different grade):**

  * **1. Select**: It reads the `Fine-grained` value: `"G3"`.
  * **2. Replace**: `"G3"` becomes `"3"`.
  * **3. Convert**: The text `"3"` becomes the number `3`.
  * **4. Assign**: It puts the number `3` into the `label` column for this row.

**Row with a higher grade (e.g., Grade 12):**

  * **1. Select**: It reads the `Fine-grained` value: `"G12"`.
  * **2. Replace**: `"G12"` becomes `"12"`.
  * **3. Convert**: The text `"12"` becomes the number `12`.
  * **4. Assign**: It puts the number `12` into the `label` column.

This process is repeated automatically for all 13,870 rows, creating a clean, numerical `label` column that is perfectly formatted for training your readability model.
# resouse

https://github.com/DamithDR/arabic-readability-assessment
