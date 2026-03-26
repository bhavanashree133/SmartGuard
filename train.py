import time
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def main():
    start_time = time.time()

    print("Loading dataset...")
    df = pd.read_csv("Data/smartguard_trackB_dataset.csv")

    print(f"Dataset size: {len(df)}")
    print("\nClass distribution:")
    print(df["label"].value_counts())

    # 70/15/15 split
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=SEED
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=SEED
    )

    print("\nSplit sizes:")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["label"])
    y_val = label_encoder.transform(val_df["label"])
    y_test = label_encoder.transform(test_df["label"])

    # TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        lowercase=True
    )

    X_train = vectorizer.fit_transform(train_df["prompt"])
    X_val = vectorizer.transform(val_df["prompt"])
    X_test = vectorizer.transform(test_df["prompt"])

    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter=2000,
        random_state=SEED
    )
    model.fit(X_train, y_train)

    # Validation
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    print("\nValidation Accuracy:", round(val_acc, 4))
    print("\nValidation Classification Report:")
    print(classification_report(y_val, val_pred, target_names=label_encoder.classes_))

    # Test
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    print("\nTest Accuracy:", round(test_acc, 4))
    print("\nTest Classification Report:")
    print(classification_report(y_test, test_pred, target_names=label_encoder.classes_))

    # Save artifacts
    joblib.dump(model, "Models/smartguard_final_model.pkl")
    joblib.dump(vectorizer, "Models/smartguard_vectorizer.pkl")
    joblib.dump(label_encoder, "Models/smartguard_label_encoder.pkl")

    print("\nSaved artifacts:")
    print("- smartguard_final_model.pkl")
    print("- smartguard_vectorizer.pkl")
    print("- smartguard_label_encoder.pkl")

    # Save split files for reproducibility
    train_df.to_csv("Data/trackb_train_split.csv", index=False)
    val_df.to_csv("Data/trackb_val_split.csv", index=False)
    test_df.to_csv("Data/trackb_test_split.csv", index=False)

    print("\nSaved split files:")
    print("- trackb_train_split.csv")
    print("- trackb_val_split.csv")
    print("- trackb_test_split.csv")

    elapsed = time.time() - start_time
    print(f"\nTraining pipeline completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()