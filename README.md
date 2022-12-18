## RUS-EN Neural machine translation.

- Goal: try different approaches to improve RUS-EN translation.
- Dataset: Provided by YSDA. Parsed from *LetsBookHotel.com*.
    - This was part 2 of Lab № 2 for the **Applied Machine Learning** course.


<br></br>

**MIPT (spring 2021).**

---

Tried 4 approaches (each in its own notebook).

### Notebook 1
- Baseline implementation: LSTM (in `network.py`)

### Notebook 2
- Implementation: LSTM + Luong's attention (in `network_luong_attn.py`)

### Notebook 3
- Simple learning rate tuning.

### Notebook 4
- Some word segmentation.

And comparison between the approaches in `Conclusions.ipynb`.
