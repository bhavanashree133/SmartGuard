import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

SEED = 42


def main():
    print("Loading dataset...")
    df = pd.read_csv("Data/smartguard_trackB_dataset.csv")

    X = df["prompt"].astype(str)
    y = df["label"].astype(str)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        stratify=y_encoded,
        random_state=SEED
    )

    # TF-IDF features
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        lowercase=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # SGD-based logistic model for epoch-wise analysis
    model = SGDClassifier(
        loss="log_loss",
        random_state=SEED
    )

    classes = np.unique(y_train)

    epochs = 15
    train_losses = []
    val_losses = []

    print("Training epoch-wise analysis model...")

    for epoch in range(epochs):
        model.partial_fit(X_train_vec, y_train, classes=classes)

        train_probs = model.predict_proba(X_train_vec)
        val_probs = model.predict_proba(X_val_vec)

        train_loss = log_loss(y_train, train_probs, labels=classes)
        val_loss = log_loss(y_val, val_probs, labels=classes)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss = {train_loss:.4f}, "
            f"Validation Loss = {val_loss:.4f}"
        )

    # Save results CSV
    loss_df = pd.DataFrame({
        "epoch": list(range(1, epochs + 1)),
        "train_loss": train_losses,
        "validation_loss": val_losses
    })

    loss_df.to_csv("Data/epoch_loss_results.csv", index=False)

    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(loss_df["epoch"], loss_df["train_loss"], marker="o", label="Train Loss")
    plt.plot(loss_df["epoch"], loss_df["validation_loss"], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("Epoch-wise Loss Curve (SGD Logistic Model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("epoch_loss_curve.png", dpi=300)
    plt.close()

    print("\nSaved:")
    print("- epoch_loss_results.csv")
    print("- epoch_loss_curve.png")


if __name__ == "__main__":
    main()