# Customer Feedback Analysis System Using NLP

A beginner-friendly NLP project that analyzes customer feedback to determine:

1. **Customer sentiment** — positive / neutral / negative
2. **Main complaint / category** — payment, login, performance, support, ui, bug, feature_request, general
3. **Important keywords / phrases** — via TF-IDF and n-grams
4. **Multiple categories per feedback** — multi-label classification
5. **Model evaluation** — honest, reproducible metrics

---

## 1. Project Title

**Customer Feedback Analysis System Using NLP**

---

## 2. Project Objective

Given a customer feedback message (e.g. *"The application is very slow and payment keeps failing."*), the system should report:

- **Sentiment** (negative)
- **Categories** (performance, payment)
- **Important keywords/phrases** (application slow, payment failing)

We build this step by step, from simple classical NLP to modern transformer embeddings.

---

## 3. Problem Statement

Companies receive thousands of feedback messages daily. Reading each one by hand is slow and inconsistent. A system that automatically:

- knows whether the feedback is positive or negative,
- identifies *which part* of the product the feedback is about,
- and extracts the key reason for the feedback,

helps teams triage issues and understand customer pain points at scale. Our task is to build such a system and do it properly — with clean evaluation, careful handling of data leakage, and honest reporting.

---

## 4. Dataset

We use the **Twitter US Airline Sentiment Dataset** (Crowdflower). It contains ~14,640 tweets about six major US airlines, with crowd-sourced sentiment labels (positive / neutral / negative).

The raw file (`Tweets.csv`) is placed in `data/raw/`.

### Labeling strategy (documented)

- **Sentiment:** we use the real `airline_sentiment` column directly. These are real crowd labels — no fabrication.
- **Categories:** the dataset only labels *negative* tweets with `negativereason` (e.g. `Customer Service Issue`, `Late Flight`). These are airline-specific and do *not* match our app-feedback categories (payment, login, ...). We therefore map each `negativereason` to the closest project category:
  - `Customer Service Issue`, `Flight Attendant Complaints` → **support**
  - `Flight Booking Problems` → **payment**
  - `Lost Luggage`, `Damaged Luggage` → **bug**
  - `Late Flight`, `Cancelled Flight`, `Bad Flight`, `longlines` → **performance**
  - `Can't Tell` → **general**
- **Multi-label categories:** the original dataset does not contain reliable multi-label app-category annotations. To demonstrate multi-label classification *without fabricating labels*, we provide a small **manually labeled** supplementary dataset (`data/processed/manual_categories.csv` — ~285 rows, hand-labeled). This is clearly documented as manually labeled.

**We do not fabricate evaluation results.** Every metric reported comes from actually training and evaluating the models in this repository.

---

## 5. Dataset Columns

| Original column | Meaning | Our use |
|---|---|---|
| `text` | The tweet / feedback text | the `feedback` field |
| `airline_sentiment` | positive / neutral / negative | the `sentiment` label |
| `negativereason` | complaint reason (negative tweets only) | mapped to project categories |

Processed outputs:

- `feedback_sentiment.csv` → `feedback`, `sentiment` (14,640 rows)
- `feedback_sentiment_cat.csv` → `feedback`, `sentiment`, `category` (single label)
- `feedback_airline_multilabel.csv` → `feedback`, `sentiment`, `categories`
- `manual_categories.csv` → `feedback`, `sentiment`, `categories` (**manually labeled** multi-label supplement)

---

## 6. NLP Concepts Used

- Text cleaning and normalization
- Tokenization
- Stop-word removal (with negation words kept)
- Lemmatization (optional, via spaCy)
- TF-IDF (Term Frequency – Inverse Document Frequency)
- n-grams (unigrams & bigrams)
- Bag-of-words representation
- Sentence embeddings (dense vectors)
- Cosine similarity
- Transformers / attention (DistilBERT)
- Multi-label classification
- Precision / Recall / F1 / Confusion matrix

---

## 7. Architecture

```
                       Customer Feedback (text)
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │      Text cleaning      │
                     └─────────────────────────┘
                                  │
            ┌───────────────┬─────┴───────────────┐
            │               │                     │
            ▼               ▼                     ▼
     TF-IDF vectorizer  Sentence embeddings   Tokenization
     (Bag of words)     (MiniLM-L6-v2)        (DistilBERT)
            │               │                     │
            ▼               ▼                     ▼
   Logistic Regression  Cosine similarity    Transformer
   (one-vs-rest)        search               classifier
            │               │                     │
            └───────┬───────┴──────────┬──────────┘
                    │                  │
                    ▼                  ▼
            Sentiment +           Semantic
            Category +            similarity
            Keywords              search
```

The **classical** path (TF-IDF + Logistic Regression) is our interpretable baseline. The **transformer** path is a modern experiment for comparison.

---

## 8. Project Structure

```
customer-feedback-nlp/
│
├── data/
│   ├── raw/            # Tweets.csv (download/place here)
│   └── processed/      # built processed CSVs + manual labels
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_text_preprocessing.ipynb
│   ├── 03_tfidf_sentiment.ipynb
│   ├── 04_multilabel_classification.ipynb
│   ├── 05_keyword_extraction.ipynb
│   ├── 06_sentence_embeddings.ipynb
│   └── 07_transformer_model.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── sentiment.py
│   ├── category_classifier.py
│   ├── multilabel_classifier.py
│   ├── keyword_extractor.py
│   ├── embeddings.py
│   ├── evaluation.py
│   └── transformer_model.py
│
├── tests/
│   └── test_preprocessing.py
│
├── make_dataset.py       # builds processed CSVs from raw data
├── train.py              # trains + saves all classical models
├── predict.py            # interactive prediction interface
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 9. Installation

Requires **Python 3.10+**.

```bash
# 1. Clone / enter the project directory
cd customer-feedback-nlp

# 2. (Recommended) create a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install core requirements
pip install -r requirements.txt

# 4. (Optional) spaCy model for lemmatization
pip install spacy
python -m spacy download en_core_web_sm

# 5. (Optional, for Stage 3 & 4) heavier NLP libs
pip install sentence-transformers transformers torch datasets
```

### Get the dataset

Place the Twitter US Airline `Tweets.csv` in `data/raw/`. Then build the processed datasets:

```bash
python make_dataset.py
```

This writes the cleaned CSVs into `data/processed/`.

---

## 10. How to Run

### Train the models

```bash
python train.py
```

This trains and saves (into `models/`):

- `sentiment_model.joblib` — TF-IDF + Logistic Regression sentiment classifier
- `multilabel_model.joblib` + `multilabel_binarizer.joblib` — multi-label category classifier
- TF-IDF keyword index files

```bash
# Optional: also fine-tune DistilBERT (slow, needs transformers/torch)
python train.py --transformer
```

### Run the prediction interface

```bash
python predict.py
```

Type a feedback message and press Enter. Use `--multi` for multiple one-per-line inputs, or `--file <path>` to read from a file.

---

## 11. Example Input

```
The application is very slow and payment keeps failing.
```

## 12. Example Output

```
========================================
CUSTOMER FEEDBACK ANALYSIS
========================================

Feedback:
The application is very slow and payment keeps failing.

Sentiment:
Negative

Categories:
- Payment
- Performance

Important Keywords/Phrases:
- payment
- slow
- application slow
- slow payment
- payment keeps

========================================
```

---

## 13. TF-IDF Explanation

**TF-IDF** (Term Frequency – Inverse Document Frequency) converts text into numbers a machine learning model can use.

- **TF (Term Frequency):** how often a word appears in one document. Words that appear often are more relevant *within* that document.
- **IDF (Inverse Document Frequency):** how rare a word is across all documents. Rare words are more informative. A word like *the* appears in everything, so its IDF is near zero.

```
TF-IDF = TF × IDF
```

The final value is high for words that are **frequent in this document AND rare across the corpus** — exactly the words that summarize a piece of feedback.

We use `TfidfVectorizer` with `ngram_range=(1,2)` (unigrams and bigrams) and `sublinear_tf=True` (which uses `1 + log(TF)` to damp down very frequent words).

---

## 14. Logistic Regression Explanation

**Logistic Regression** is a linear classifier. It learns a weight for each word (feature) and combines them:

```
score(feedback) = w1·word1 + w2·word2 + ... + bias
```

The score is passed through a **sigmoid** to produce the probability of each class. The model learns the weights so that, for example, the word *failing* gets a large weight for the *negative* class.

For multiple classes, scikit-learn trains **one-vs-rest**: a separate binary model per class, choosing the highest-scoring one.

**Why it's a good baseline:** it's fast, interpretable (we can inspect word weights), requires little data, and works surprisingly well with TF-IDF features.

---

## 15. Multi-label Classification Explanation

Sometimes one feedback is about *two things*:

> "The app is very slow and payment keeps failing."

It's about **performance** *and* **payment** — we must not force it into a single category.

Multi-label classification decomposes this into **one binary question per category**:

- "Is this about payment?" → yes/no
- "Is this about login?" → yes/no
- "Is this about performance?" → yes/no
- ... and so on

Tools used:

- **`MultiLabelBinarizer`** — turns a list of label-sets into a `(samples × categories)` binary matrix.
- **`OneVsRestClassifier(LogisticRegression)`** — trains one binary classifier per category; a sample can get *many* `yes` answers.

**Metrics:** because multiple labels are involved, we use **Micro F1** and **Macro F1** — a sample is not "right/wrong" in a simple sense, so we measure how well the model recovers each individual label.

---

## 16. Keyword Extraction Explanation

We extract important keywords and phrases using **TF-IDF + n-grams**:

1. Fit a TF-IDF model on the **whole corpus**.
2. For a given feedback, find which corpus-rare terms it contains → its most informative single words.
3. Extract **bigrams** (two-word phrases) like `payment gateway`, `app slow` from the text.
4. Merge and deduplicate.

Because we fit TF-IDF on the corpus, a rare-but-meaningful word like *gateway* ranks highly, while filler words like *the* and *is* are dropped (they appear everywhere).

**Stop words:** we only remove boring function words. Negation words (`not`, `never`, `no`) are **kept**, because they change meaning.

---

## 17. Sentence Embeddings Explanation

TF-IDF represents a sentence by which words appear. But "Payment failed" and "Couldn't complete my card transaction" share few words even though they mean the same thing.

**Sentence embeddings** solve this. A model like `all-MiniLM-L6-v2` maps each sentence to a 384-number vector in a way that places *similar-meaning* sentences close together.

We measure closeness with **cosine similarity** (the cosine of the angle between two vectors, from −1 to 1).

```
"Payment failed during checkout"      → vector A
"Unable to complete my card transaction" → vector B
cosine(A, B) ≈ 0.7  (similar)
```

This powers **similar feedback search**: given one feedback, return the most semantically similar others.

---

## 18. Transformer Explanation

A **transformer** is a deep neural network that processes whole sentences using **attention**. Unlike a bag-of-words, it considers each word *in context* of the others.

**DistilBERT** is a distilled version of BERT:

- ~40% smaller and 60% faster than BERT
- retains ~97% of BERT's language understanding
- pre-trained on huge text corpora, then **fine-tuned** on our task

In Stage 4 we fine-tune `distilbert-base-uncased` for sentiment classification and compare it to the classical model.

### TF-IDF + LogisticRegression vs DistilBERT

| Aspect | TF-IDF + LogReg | DistilBERT |
|---|---|---|
| **Accuracy / F1** | good baseline | often higher on complex text |
| **Training complexity** | seconds, CPU only | minutes, GPU helps |
| **Inference complexity** | instant, tiny | slower, more memory |
| **Context understanding** | none (bag of words) | full (attention) |
| **Interpretability** | high (word weights) | low (black box) |

**We do not claim the transformer is better unless the evaluation actually shows it.** On simple, keyword-heavy sentiment data the classical model often matches the transformer at a fraction of the cost.

---

## 19. Evaluation Metrics

We report standard classification metrics on a **held-out test set** (never the training data):

- **Accuracy** — fraction of correct predictions
- **Precision** — of items predicted positive, how many were right
- **Recall** — of actual positives, how many were found
- **F1-score** — harmonic mean of precision and recall
- **Confusion matrix** — shows exactly which classes get confused
- **Micro F1 / Macro F1** — used for multi-label

**Data leakage is prevented** by using sklearn `Pipeline`, so the TF-IDF vectorizer is fit **only on training data** and never sees the test set before prediction.

**Notably**, we accurately report both: the **honest airline-only baseline** and the **deployed model** (which also uses the small manual supplement). We never fake numbers.

---

## 20. Results

Run `python train.py` to reproduce locally. Typical results (may vary with sklearn version):

### Sentiment (TF-IDF + Logistic Regression)

- **Airline-only baseline:** Accuracy ≈ 0.79, F1 ≈ 0.78
- **Deployed (with manual supplement):** Accuracy ≈ 0.78, F1 ≈ 0.76

Per-class, *negative* sentiment is recovered very well (recall > 0.9). *neutral* and *positive* are harder and show more confusion.

### Multi-label category (manual supplement)

- Micro F1 ≈ 0.60–0.67, Macro F1 ≈ 0.57–0.64

The flagship example `"The application is very slow and payment keeps failing."` returns **both** `performance` and `payment`.

### Keyword extraction

- `"The payment gateway failed during checkout."` → `payment`, `failed`, `payment gateway`, `gateway failed`, `failed checkout`

### Sentence embeddings (Stage 3)

- `"Payment failed during checkout."` and `"Unable to complete my card transaction."` have high cosine similarity (> 0.6).
- Similar-feedback search returns top-3 relevant messages.

---

## 21. Limitations

1. **Domain gap:** the sentiment model is trained on airline tweets. Generic app feedback (payments, logins) is somewhat less accurate, which is why we add the small manually-labeled supplement.
2. **Small multi-label supplement:** the manual dataset is only ~285 rows. Multi-label metrics are a *demonstration*, not production-grade. A real deployment needs more labeled data.
3. **Bag-of-words limits:** TF-IDF can't handle sarcasm, word order, or complex negation. ("Great job!" said sarcastically is hard.)
4. **Manual labels:** the category supplement is hand-labeled by a single annotator and may contain biases.
5. **Transformers need data & compute:** DistilBERT needs GPU or long CPU training and lots of data to beat the classical baseline.
6. **No spelling-correction:** tweets contain typos that hurt all models somewhat.

---

## 22. Future Improvements

1. **More labeled multi-label data** — larger, multi-annotator supplement to make category metrics trustworthy.
2. **Sarcasm handling** — more training data + larger transformer.
3. **Confidence thresholds** — only report a category when the model is confident, otherwise ask the user.
4. **Spelling correction** — e.g. a small edit-distance corrector before TF-IDF.
5. **Cross-validation** — use `cross_val_score` for more stable metric estimates.
6. **Class weighting on the airline sentiment** — handle the imbalance more aggressively if it matters for the use-case.
7. **Larger transformer** (BERT/RoBERTa) and **hyperparameter tuning**.
8. **Feedback clustering** with sentence embeddings to auto-discover emerging themes.
9. **Active learning** — route low-confidence feedback to a human for labeling, improving the model over time.
10. **Multilingual support** using a multilingual embedding/transformer model.

---

## Reproducibility

- `random_state=42` used throughout.
- Fixed `train_test_split` ratios and stratify where appropriate.
- No hard-coded predictions; every number comes from actually running the code.
