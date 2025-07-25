# Hybrid Model 
### 7 numerical features 
 It was trained on data with 7 features per sentence.


- features based on **samer_lexicon**:
1. avg_readability
2. max_readability are based on



# Converting samer scales to barec scale 

The line:

```python
samer_level_map = {'L3': 4, 'L4': 10, 'L5': 16}
```

**What does this mean?**
- This creates a mapping between the SAMER readability corpus levels (`L3`, `L4`, `L5`) and specific numerical values used in the BAREC dataset.
- `L3` is mapped to 4
- `L4` is mapped to 10
- `L5` is mapped to 16

---

**How does this relate to the BAREC scale (1-19)?**

- The BAREC dataset uses a fine-grained readability scale from 1 to 19, where each number represents a different readability level.
- The SAMER corpus uses broader level names: `L3`, `L4`, `L5`. These are less granular than BAREC’s scale.
- The mapping essentially places each SAMER level at a representative point within the BAREC scale:
  - `L3` is considered similar to BAREC level **4**
  - `L4` is considered similar to BAREC level **10**
  - `L5` is considered similar to BAREC level **16**

**Why do this?**
- When augmenting the BAREC data with sentences from SAMER, you need to assign each SAMER sentence a label compatible with BAREC’s system.
- Since the SAMER levels are coarse, the code picks a number in the BAREC range that roughly represents each SAMER level.

**How was this mapping chosen?**
- The choice of 4, 10, and 16 is a design decision. Typically, it’s based on:
  - The distribution or meaning of levels in SAMER versus BAREC.
  - Placing L3, L4, L5 at intervals in BAREC’s 1–19 scale to roughly match their difficulty.

**Summary:**  
This mapping allows data from the SAMER corpus (which uses `L3`, `L4`, `L5`) to be “translated” into the BAREC scale (1–19) so the two datasets can be combined and used together. The numbers 4, 10, and 16 are chosen representative points in the BAREC scale for these SAMER levels.
