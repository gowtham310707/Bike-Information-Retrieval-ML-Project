import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MODELS = {
    "KNN": KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1500, random_state=42),
}

def build_pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, stop_words="english",
            max_features=30000, ngram_range=(1, 2), sublinear_tf=True
        )),
        ("classifier", classifier)
    ])

def evaluate(dataset_name, texts, labels, class_names):
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    rows = []
    for model_name, classifier in MODELS.items():
        model = build_pipeline(classifier)

        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        predictions = model.predict(X_test)
        predict_seconds = time.perf_counter() - start

        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="weighted", zero_division=0
        )
        rows.append({
            "dataset": dataset_name,
            "algorithm": model_name,
            "accuracy_percent": round(accuracy * 100, 2),
            "weighted_precision_percent": round(precision * 100, 2),
            "weighted_recall_percent": round(recall * 100, 2),
            "weighted_f1_percent": round(f1 * 100, 2),
            "training_time_seconds": round(train_seconds, 3),
            "prediction_time_seconds": round(predict_seconds, 3),
        })

        print(f"\n--- {dataset_name}: {model_name} ---")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Weighted precision: {precision * 100:.2f}%")
        print(f"Weighted recall: {recall * 100:.2f}%")
        print(f"Weighted F1: {f1 * 100:.2f}%")
        print(f"Training time: {train_seconds:.3f} seconds")
        print(classification_report(
            y_test, predictions, target_names=class_names, zero_division=0
        ))

        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test, predictions, display_labels=class_names,
            xticks_rotation=35, cmap="Blues", colorbar=False, ax=ax
        )
        ax.set_title(f"{dataset_name} — {model_name}")
        fig.tight_layout()
        safe = dataset_name.lower().replace(" ", "_") + "_" + model_name.lower().replace(" ", "_")
        fig.savefig(RESULTS / f"{safe}_confusion_matrix.png", dpi=160)
        plt.close(fig)

    result = pd.DataFrame(rows).sort_values(
        ["accuracy_percent", "weighted_f1_percent"], ascending=False
    ).reset_index(drop=True)
    result.insert(0, "rank_by_accuracy", range(1, len(result) + 1))
    safe = dataset_name.lower().replace(" ", "_")
    result.to_csv(RESULTS / f"{safe}_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    result.sort_values("accuracy_percent").plot(
        x="algorithm", y="accuracy_percent", kind="barh", legend=False, ax=ax
    )
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Algorithm")
    ax.set_xlim(0, 100)
    ax.set_title(f"Bike project — accuracy comparison: {dataset_name}")
    fig.tight_layout()
    fig.savefig(RESULTS / f"{safe}_accuracy.png", dpi=160)
    plt.close(fig)

    print(f"\nRanking for {dataset_name}:")
    print(result[["rank_by_accuracy", "algorithm", "accuracy_percent",
                  "weighted_f1_percent", "training_time_seconds"]].to_string(index=False))
    return result

def run_bike_catalog():
    path = DATA / "bike_catalog_texts.csv"
    df = pd.read_csv(path).dropna(subset=["category", "text"])
    print("\nExperiment 1: Built-in bike catalogue teaching dataset")
    print("Note: This is a small illustrative corpus included with the project.")
    return evaluate(
        "Bike Catalogue", df["text"].astype(str).tolist(),
        df["category"].astype(str).tolist(), sorted(df["category"].unique())
    )

def run_20newsgroups():
    print("\nExperiment 2: 20 Newsgroups motorcycle vs automobile discussions")
    print("First run may require internet access to download the dataset.")
    categories = ["rec.motorcycles", "rec.autos"]
    train = fetch_20newsgroups(
        subset="train", categories=categories, shuffle=True, random_state=42,
        remove=("headers", "footers", "quotes")
    )
    test = fetch_20newsgroups(
        subset="test", categories=categories, shuffle=True, random_state=42,
        remove=("headers", "footers", "quotes")
    )
    texts = list(train.data) + list(test.data)
    labels = list(train.target) + list(test.target)
    return evaluate("20 Newsgroups Bike vs Auto", texts, labels, list(train.target_names))

def search_bike_documents():
    df = pd.read_csv(DATA / "bike_catalog_texts.csv").dropna(subset=["text"])
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(df["text"].astype(str))
    print("\nBike information retrieval demo")
    print("Example queries: electric bike battery | motorcycle engine service | mountain bicycle")
    query = input("Enter your bike-related search query (Enter to skip): ").strip()
    if not query:
        return
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    top_indices = scores.argsort()[-5:][::-1]
    print("\nTop 5 retrieved bike documents:")
    for rank, idx in enumerate(top_indices, 1):
        row = df.iloc[idx]
        print(f"\n{rank}. Category: {row['category']} | Similarity: {scores[idx]:.3f}")
        print(row["text"])

if __name__ == "__main__":
    print("=" * 70)
    print("BIKE INFORMATION RETRIEVAL + MACHINE LEARNING COMPARISON")
    print("=" * 70)
    run_bike_catalog()
    run_20newsgroups()
    search_bike_documents()
    print("\nCompleted. Check results/ for CSV tables, graphs and confusion matrices.")
    print("Use actual measured values from your run in your report.")
