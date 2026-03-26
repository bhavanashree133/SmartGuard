import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

SEED = 42


def main():
    print("Loading dataset...")
    df = pd.read_csv("Data/smartguard_trackB_dataset.csv")

    X = df["prompt"].astype(str)
    y = df["label"].astype(str)

    # -----------------------------
    # Learning Curve
    # -----------------------------
    print("Generating learning curve...")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, lowercase=True)),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED))
    ])

    train_sizes, train_scores, val_scores = learning_curve(
        pipeline,
        X,
        y,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=None
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    learning_df = pd.DataFrame({
        "train_size": train_sizes,
        "train_accuracy": train_mean,
        "validation_accuracy": val_mean
    })
    learning_df.to_csv("Data/learning_curve_results.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, marker="o", label="Training Accuracy")
    plt.plot(train_sizes, val_mean, marker="o", label="Validation Accuracy")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve - Logistic Regression (Track B)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=300)
    plt.close()

    # -----------------------------
    # Validation over different C values
    # -----------------------------
    print("Generating regularization curve...")

    train_df, val_df = train_test_split(
        df,
        test_size=0.15,
        stratify=df["label"],
        random_state=SEED
    )

    X_train = train_df["prompt"].astype(str)
    y_train = train_df["label"].astype(str)
    X_val = val_df["prompt"].astype(str)
    y_val = val_df["label"].astype(str)

    c_values = [0.01, 0.1, 1, 5, 10]
    rows = []

    for c in c_values:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, lowercase=True)),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED, C=c))
        ])

        pipe.fit(X_train, y_train)

        train_acc = pipe.score(X_train, y_train)
        val_acc = pipe.score(X_val, y_val)

        rows.append({
            "C": c,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc
        })

    c_df = pd.DataFrame(rows)
    c_df.to_csv("Data/regularization_curve_results.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(c_df["C"], c_df["train_accuracy"], marker="o", label="Training Accuracy")
    plt.plot(c_df["C"], c_df["validation_accuracy"], marker="o", label="Validation Accuracy")
    plt.xscale("log")
    plt.xlabel("C (Inverse Regularization Strength)")
    plt.ylabel("Accuracy")
    plt.title("Validation Curve - Logistic Regression (Track B)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("regularization_curve.png", dpi=300)
    plt.close()

    print("Saved:")
    print("- learning_curve_results.csv")
    print("- learning_curve.png")
    print("- regularization_curve_results.csv")
    print("- regularization_curve.png")


if __name__ == "__main__":
    main()