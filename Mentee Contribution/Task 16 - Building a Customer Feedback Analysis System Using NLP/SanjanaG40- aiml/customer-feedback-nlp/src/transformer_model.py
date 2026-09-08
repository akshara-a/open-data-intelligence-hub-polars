"""
Transformer model module for Customer Feedback Analysis System.

Implements sentiment and category classification using DistilBERT
(a lightweight, distilled version of BERT).

DistilBERT:
- 40% smaller than BERT
- 60% faster than BERT
- Retains 97% of BERT's language understanding capability
- Pre-trained on a massive corpus (Wikipedia + BookCorpus)

This module provides a comparison point against the classical
TF-IDF + Logistic Regression approach.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
except ImportError:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

try:
    from datasets import Dataset
except ImportError:
    Dataset = None


# Label encoding maps
SENTIMENT_MAP = {"negative": 0, "neutral": 1, "positive": 2}
SENTIMENT_MAP_INV = {v: k for k, v in SENTIMENT_MAP.items()}

MODEL_NAME = "distilbert-base-uncased"


class TransformerSentimentClassifier:
    """
    Sentiment classifier using DistilBERT.

    Fine-tunes a pre-trained DistilBERT model on the sentiment dataset.
    This is a more complex approach than TF-IDF + Logistic Regression
    but can capture contextual nuances in language.

    Trade-offs:
    + Captures word context (e.g., "not good" vs "good")
    + Better at handling negation, sarcasm, complex sentences
    - Slower training and inference
    - Requires more memory
    - Less interpretable than TF-IDF
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_labels: int = 3,
        max_length: int = 128,
    ):
        if AutoTokenizer is None:
            raise ImportError(
                "transformers is required. "
                "Install with: pip install transformers"
            )

        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.trainer = None

    def _tokenize(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
    ) -> "Dataset":
        """
        Tokenize texts for the Transformer model.

        DistilBERT tokenization:
        1. Split text into subword tokens (WordPiece)
        2. Add [CLS] token at start and [SEP] token at end
        3. Pad/truncate to max_length
        4. Create attention masks (1 for real tokens, 0 for padding)
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        data = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }
        if labels is not None:
            data["labels"] = labels

        return Dataset.from_dict(data)

    def prepare_data(
        self,
        texts: List[str],
        labels: List[str],
        label_map: Dict[str, int],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple["Dataset", "Dataset"]:
        """
        Prepare train/test datasets for the Transformer.

        Returns HuggingFace Dataset objects ready for the Trainer.
        """
        # Convert string labels to integers
        int_labels = [label_map[l.lower()] for l in labels]

        X_train, X_test, y_train, y_test = train_test_split(
            texts, int_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=int_labels,
        )

        train_dataset = self._tokenize(X_train, y_train)
        test_dataset = self._tokenize(X_test, y_test)

        return train_dataset, test_dataset

    def train(
        self,
        train_dataset: "Dataset",
        output_dir: str = "./models/transformer_sentiment",
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ) -> Dict[str, float]:
        """
        Fine-tune DistilBERT on the sentiment dataset.

        Training process:
        1. Forward pass: text -> DistilBERT -> logits
        2. Compute cross-entropy loss
        3. Backpropagation: update model weights
        4. Repeat for specified epochs

        Parameters
        ----------
        train_dataset : Dataset
            Training data with input_ids, attention_mask, and labels.
        output_dir : str
            Directory to save the fine-tuned model.
        epochs : int
            Number of training epochs.
        batch_size : int
            Training batch size.
        learning_rate : float
            Learning rate (typically 2e-5 to 5e-5 for BERT models).

        Returns
        -------
        dict
            Training metrics.
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=50,
            save_strategy="epoch",
            report_to="none",  # Disable wandb/tensorboard logging
            remove_unused_columns=False,
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
            return {"accuracy": acc, "f1": f1}

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=compute_metrics,
        )

        train_result = self.trainer.train()
        return train_result.metrics

    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict sentiment for a list of texts.

        Returns human-readable labels (positive, negative, neutral).
        """
        dataset = self._tokenize(texts)
        output = self.trainer.predict(dataset)
        predictions = np.argmax(output.predictions, axis=-1)
        return [SENTIMENT_MAP_INV[p] for p in predictions]

    def evaluate(self, test_dataset: "Dataset") -> Dict[str, float]:
        """
        Evaluate the fine-tuned model on a test dataset.
        """
        if self.trainer is None:
            raise RuntimeError("Model not trained. Call train() first.")
        result = self.trainer.evaluate(test_dataset)
        return result


def compare_classical_vs_transformer(
    classical_results: Dict[str, float],
    transformer_results: Dict[str, float],
) -> None:
    """
    Print a side-by-side comparison of classical vs transformer models.

    Discusses:
    - Accuracy, Precision, Recall, F1
    - Training complexity
    - Inference complexity
    - Context understanding
    - Interpretability
    """
    print("\n" + "=" * 70)
    print("  Classical (TF-IDF + LR) vs Transformer (DistilBERT)")
    print("=" * 70)

    metrics = ["accuracy", "precision", "recall", "f1"]
    print(f"\n  {'Metric':<20} {'Classical':>12} {'Transformer':>12}")
    print("  " + "-" * 44)

    for m in metrics:
        c_val = classical_results.get(m, float("nan"))
        t_val = transformer_results.get(m, float("nan"))
        winner = " <--" if t_val > c_val else ""
        print(f"  {m:<20} {c_val:>12.4f} {t_val:>12.4f}{winner}")

    print("\n  Complexity Comparison:")
    print("  " + "-" * 44)
    print("  Training time:      Classical << Transformer")
    print("  Inference speed:    Classical >> Transformer")
    print("  Memory usage:       Classical << Transformer")
    print("  Interpretability:   Classical >> Transformer")
    print("  Context handling:   Classical << Transformer")
    print("  Data requirements:  Classical << Transformer")

    print("\n  Interpretation:")
    if transformer_results.get("f1", 0) > classical_results.get("f1", 0):
        diff = transformer_results["f1"] - classical_results["f1"]
        if diff > 0.05:
            print("  The Transformer shows a meaningful improvement over the classical approach.")
            print("  This suggests the dataset benefits from contextual understanding.")
        elif diff > 0.01:
            print("  The Transformer shows a slight improvement.")
            print("  The marginal gain may not justify the added complexity for simple tasks.")
        else:
            print("  Performance is similar. For this task, the classical approach")
            print("  may be preferred due to lower complexity and better interpretability.")
    else:
        print("  The classical approach outperforms or matches the Transformer.")
        print("  This can happen when:")
        print("  - The dataset is small (Transformers need more data)")
        print("  - The task is simple enough for TF-IDF features")
        print("  - The Transformer was not fine-tuned long enough")
    print()
