"""
TF-IDF & Linear Regression Classifier

Task:
    Classify each text as:
        0 = Not a movie/TV review
        1 = Positive movie/TV review
        2 = Negative movie/TV review

This script reproduces the notebook workflow

Run from the project root:
    python scripts/baseline_classifier.py
"""

from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SUBMISSION_FILE = OUTPUT_DIR / "submission.csv"

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def load_data(train_file: Path = TRAIN_FILE, test_file: Path = TEST_FILE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train and test CSV files."""
    if not train_file.exists():
        raise FileNotFoundError(f"Missing train file: {train_file}")

    if not test_file.exists():
        raise FileNotFoundError(f"Missing test file: {test_file}")

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    return train, test


def clean_text_columns(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill missing TEXT values so sklearn vectorizers do not fail."""
    train = train.copy()
    test = test.copy()

    train["TEXT"] = train["TEXT"].fillna("")
    test["TEXT"] = test["TEXT"].fillna("")

    return train, test


def print_data_overview(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print dataset structure, missing values, and class distribution."""
    print("\nDATA OVERVIEW\n")

    print("Train columns:")
    print(train.columns.tolist())

    print("\nTest columns:")
    print(test.columns.tolist())

    print("\nMissing values in train:")
    print(train.isna().sum())

    print("\nMissing values in test:")
    print(test.isna().sum())

    print("\nLabel distribution:")
    print(train["LABEL"].value_counts().sort_index())

    print("\nLabel distribution (%):")
    print((train["LABEL"].value_counts(normalize=True).sort_index() * 100).round(2))


def print_sample_texts(train: pd.DataFrame, n_examples: int = 3, max_chars: int = 500) -> None:
    """Print sample texts from each class."""
    print("\n SAMPLE TEXTS \n")

    for label in [0, 1, 2]:
        print(f"\n CLASS {label} EXAMPLES \n")
        samples = train[train["LABEL"] == label]["TEXT"].head(n_examples)

        for i, text in enumerate(samples, 1):
            print(f"Example {i}:\n{text[:max_chars]}\n")


def build_baseline_model() -> Pipeline:
    """Build the baseline TF-IDF & Logistic Regression model."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])


def build_trigram_model() -> Pipeline:
    """Build the comparison model with extended n-grams."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.9
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])


def evaluate_model(model: Pipeline, X_val: pd.Series, y_val: pd.Series, model_name: str) -> tuple[float, pd.Series]:
    """Evaluate a trained model with macro F1 and a classification report."""
    print(f"\n {model_name.upper()} EVALUATION \n")

    preds = model.predict(X_val)
    macro_f1 = f1_score(y_val, preds, average="macro")

    print(f"{model_name} Macro F1: {macro_f1:.6f}\n")
    print("Classification Report:")
    print(classification_report(y_val, preds))

    print("Confusion Matrix:")
    print(confusion_matrix(y_val, preds))

    return macro_f1, preds


def print_error_examples(
    X_val: pd.Series,
    y_val: pd.Series,
    preds,
    n_examples: int = 5,
    max_chars: int = 500
) -> pd.DataFrame:
    """Print and return a dataframe of misclassified validation examples."""
    print("\n ERROR ANALYSIS \n")

    errors = X_val[preds != y_val]
    true_labels = y_val[preds != y_val]
    pred_labels = preds[preds != y_val]

    error_df = pd.DataFrame({
        "text": errors,
        "true": true_labels,
        "pred": pred_labels
    })

    for _, row in error_df.head(n_examples).iterrows():
        print(f"\nTRUE: {row['true']} | PRED: {row['pred']}\n")
        print(row["text"][:max_chars])

    return error_df


def create_submission(model: Pipeline, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Train the selected final model on full data and create a Kaggle submission file."""
    print("\n TRAIN FINAL MODEL \n")

    X_full = train["TEXT"]
    y_full = train["LABEL"]
    X_test = test["TEXT"]

    model.fit(X_full, y_full)

    test_predictions = model.predict(X_test)

    submission = pd.DataFrame({
        "ID": test["ID"],
        "LABEL": test_predictions
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_FILE, index=False)

    print(f"\nSubmission saved to: {SUBMISSION_FILE}")
    print("\nSubmission preview:")
    print(submission.head())

    return submission


# ---------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------

def main() -> None:
    """Run the full baseline classification pipeline."""
    train, test = load_data()
    print_data_overview(train, test)

    train, test = clean_text_columns(train, test)
    print_sample_texts(train)

    X = train["TEXT"]
    y = train["LABEL"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("\nTRAIN/VALIDATION SPLIT \n")
    print("Train size:", X_train.shape)
    print("Validation size:", X_val.shape)

    # Baseline model
    baseline_model = build_baseline_model()
    baseline_model.fit(X_train, y_train)
    baseline_f1, baseline_preds = evaluate_model(
        baseline_model,
        X_val,
        y_val,
        model_name="Baseline Model"
    )

    print_error_examples(X_val, y_val, baseline_preds)

    # Trigram comparison model
    trigram_model = build_trigram_model()
    trigram_model.fit(X_train, y_train)
    trigram_f1, _ = evaluate_model(
        trigram_model,
        X_val,
        y_val,
        model_name="Trigram Model"
    )

    print("\n MODEL COMPARISON \n")
    print(f"Baseline Macro F1: {baseline_f1:.6f}")
    print(f"Trigram Macro F1:  {trigram_f1:.6f}")
    print(f"Difference:        {trigram_f1 - baseline_f1:.6f}")

    # Final model selection:

    final_model = baseline_model

    create_submission(final_model, train, test)


if __name__ == "__main__":
    main()
