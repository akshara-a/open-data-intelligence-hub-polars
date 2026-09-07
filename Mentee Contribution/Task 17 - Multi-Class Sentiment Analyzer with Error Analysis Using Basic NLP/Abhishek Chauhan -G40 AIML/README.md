# Multi-Class Sentiment Analyzer with Error Analysis

NLP project that classifies airline-related tweets into three
sentiment classes — **positive**, **neutral**, and **negative** — using TF-IDF features and
Logistic Regression, followed by a manual error analysis.

---

## Project Structure

```text
.
├── Datasets/
│   └── airline_tweets_dataset.csv     # Raw dataset (Twitter US Airline Sentiment)
├── Notebooks/
│   └── Sentiment_Analysis_Notebook.ipynb   # Full pipeline, code + outputs
├── Plots/
│   ├── data_distribution_bar_plot.png # Class distribution before training
│   └── confusion_matrix.png           # Model performance on the test set
├── REPORT.md                          # Full write-up with embedded plots and results
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## Dataset

- **Name:** Twitter US Airline Sentiment
- **File:** `Datasets/airline_tweets_dataset.csv`
- **Size:** 14,640 tweets, 15 original columns
- **Columns used:** `text` (tweet content), `airline_sentiment` (renamed to `sentiment`)
- **Classes:** `negative` (9,178), `neutral` (3,099), `positive` (2,363) — imbalanced, so
  `class_weight="balanced"` is used in the model.

---

## Pipeline Overview

1. **Load** the dataset and keep only the `text` and `sentiment` columns.
2. **Clean** the text: lowercase, strip URLs, `@mentions`, hashtag symbols, punctuation,
   numbers, and extra whitespace.
3. **Split** into 80% train / 20% test with stratification on `sentiment`.
4. **Vectorize** with `TfidfVectorizer` (unigrams + bigrams, `max_features=5000`, English
   stop words removed).
5. **Train** a `LogisticRegression` classifier (`class_weight="balanced"`).
6. **Evaluate** with accuracy, precision, recall, F1-score, and a confusion matrix.
7. **Error analysis**: isolate misclassified tweets, group by (actual, predicted) pair, and
   manually read examples to understand *why* the model gets them wrong (negation, sarcasm,
   mixed opinions, weak/neutral wording).

See `REPORT.md` for the full results, plots, and written observations.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the dataset

Place `airline_tweets_dataset.csv` in the `Datasets/` folder (already included in this
project; if you're starting fresh, any copy of the Twitter US Airline Sentiment dataset with
`text` and `airline_sentiment` columns will work).

### 3. Run the notebook

```bash
jupyter notebook Notebooks/Sentiment_Analysis_Notebook.ipynb
```

Run all cells top to bottom. The notebook loads the CSV using a relative path
(`../Datasets/airline_tweets_dataset.csv`), so keep the folder structure above intact, or
open the notebook from inside `Notebooks/`.

---

## Results Summary

| Metric | Score |
|---|---|
| Accuracy | **74.25%** |
| Macro F1-score | 0.70 |
| Weighted F1-score | 0.75 |

The model performs best on the `negative` class (largest, most linguistically distinct
class) and struggles most with `neutral` tweets, which tend to lack strong sentiment words
and get pulled toward `negative` or `positive`. Full breakdown, plots, and example errors
are in `REPORT.md`.

---

## Known Limitations

- TF-IDF is a bag-of-words style representation — it doesn't capture word order or deep
  context.
- Negation (`"not bad"`), sarcasm (`"great, another delay"`), and mixed-opinion sentences
  are the model's main failure modes.
- The dataset is imbalanced toward negative tweets, which biases the model even with
  class weighting.

## Possible Improvements

- Balance or augment the `neutral` and `positive` classes.
- Try Multinomial Naive Bayes or a linear SVM for comparison.
- Add a rule-based or lexicon-based negation handler before vectorizing.
- Move to contextual embeddings (e.g. a Transformer-based model) for better handling of
  sarcasm and negation, once basic NLP techniques have been demonstrated.
