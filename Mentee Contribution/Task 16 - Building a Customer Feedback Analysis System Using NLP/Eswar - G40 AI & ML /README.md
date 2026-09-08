# Customer Feedback Analysis System (NLP)

A simple NLP project that analyzes customer feedback to identify:

- **Sentiment** (positive / negative / neutral)
- **Category** (payment, login, performance, support, ui, bug, feature_request)
- **Important keywords / phrases**
- **Similar feedback** (via sentence embeddings)

The project focuses purely on NLP concepts — no databases, APIs, or deployment.

## Example

Input:

```text
"The latest update is good, but payment is failing frequently."
```

Output:

```text
Sentiment: Negative
Categories: Payment, Application Update
Important Words: latest update, payment, failing
```

## Project Structure

```text
customer-feedback-nlp/
├── data/
│   └── feedback.csv
├── notebooks/
│   ├── 01_text_preprocessing.ipynb
│   ├── 02_tfidf.ipynb
│   ├── 03_sentiment.ipynb
│   ├── 04_category_classification.ipynb
│   └── 05_transformers.ipynb
├── src/
│   ├── preprocessing.py
│   ├── sentiment.py
│   ├── category_classifier.py
│   ├── keyword_extractor.py
│   └── embeddings.py
├── requirements.txt
└── README.md
```

## Pipeline (Classical NLP)

```text
Customer Feedback
    -> Text Cleaning
    -> Tokenization
    -> Careful Stop-word Removal (negations kept)
    -> TF-IDF (unigrams + bigrams)
    -> Logistic Regression
    -> Sentiment / Category
```

## Advanced Extension

```text
Customer Feedback
    -> Transformer Tokenizer
    -> Transformer Model (DistilBERT)
    -> Sentiment / Categories

    -> Sentence Embeddings (sentence-transformers)
    -> Cosine Similarity
    -> Similar Feedback Detection
```

## NLP Concepts Covered

Text Cleaning, Tokenization, Stop Words, Lemmatization, Bag of Words,
TF-IDF, N-grams, Text Classification, Sentiment Analysis, Multi-label
Classification, Word Embeddings, Sentence Embeddings, Transformers,
Keyword Extraction, Model Evaluation.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # optional, for lemmatization
```

## Usage

Run each `src/` module directly for a quick demo, e.g.:

```bash
cd src
python sentiment.py
python category_classifier.py
python keyword_extractor.py
python embeddings.py
```

Or step through the notebooks in order (`01` -> `05`) for the full
walkthrough with explanations.

## Modules

| File | Purpose |
|---|---|
| `src/preprocessing.py` | Text cleaning, tokenization, stop-word removal (keeps negations), optional lemmatization |
| `src/sentiment.py` | TF-IDF + Logistic Regression sentiment classifier |
| `src/category_classifier.py` | Single-label and multi-label feedback category classification |
| `src/keyword_extractor.py` | TF-IDF based keyword / key-phrase extraction |
| `src/embeddings.py` | Sentence embeddings + cosine similarity for similar-feedback detection |

## Evaluation

Models are evaluated with a train/test split (80/20) using precision,
recall, F1-score, and a confusion matrix (see `src/sentiment.py` and
`src/category_classifier.py`, or notebooks `03` and `04`).

The included `data/feedback.csv` is intentionally small for learning
purposes — accuracy improves significantly with a larger, more
balanced dataset.

## Notes

- `sentiment-transformers` and `transformers` are only required for the
  advanced Stage 3 (notebook `05`). If they are not installed, the
  classical TF-IDF pipeline (notebooks `01`-`04`) still runs fully, and
  `src/embeddings.py` falls back to a TF-IDF based similarity measure.
- Stop-word removal deliberately preserves negation words (e.g. "not",
  "never") so that sentiment-bearing meaning is not lost.
