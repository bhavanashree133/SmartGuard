import time
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report


def map_category_to_binary(label):
    """
    Convert category labels into safe/unsafe binary labels.
    safe   -> safe
    others -> unsafe
    """
    if str(label).strip().lower() == "safe":
        return "safe"
    return "unsafe"


def map_binary_to_decision(label):
    """
    Convert binary label into final firewall decision.
    safe   -> ALLOW
    unsafe -> BLOCK
    """
    if str(label).strip().lower() == "safe":
        return "ALLOW"
    return "BLOCK"


def main():
    print("Running Track B evaluation on official red-team suite...")

    # -----------------------------
    # Load trained artifacts
    # -----------------------------
    model = joblib.load("Models/smartguard_final_model.pkl")
    vectorizer = joblib.load("Models/smartguard_vectorizer.pkl")
    label_encoder = joblib.load("Models/smartguard_label_encoder.pkl")

    # -----------------------------
    # Load baseline red-team results
    # This file should contain:
    # prompt, category, true_label, decision, etc.
    # -----------------------------
    baseline_df = pd.read_csv("Data/red_team_results.csv")

    if "prompt" not in baseline_df.columns:
        raise ValueError("Data/red_team_results.csv must contain a 'prompt' column.")

    if "true_label" not in baseline_df.columns:
        raise ValueError("Data/red_team_results.csv must contain a 'true_label' column.")

    if "decision" not in baseline_df.columns:
        raise ValueError("Data/red_team_results.csv must contain a 'decision' column.")

    prompts = baseline_df["prompt"].astype(str).tolist()

    # -----------------------------
    # Predict with Track B model
    # -----------------------------
    X = vectorizer.transform(prompts)

    latencies = []
    predicted_categories = []
    predicted_binary_labels = []
    predicted_decisions = []
    confidences = []

    for i in range(X.shape[0]):
        single_x = X[i]

        start = time.perf_counter()
        pred_encoded = model.predict(single_x)[0]
        end = time.perf_counter()

        latency = end - start
        latencies.append(latency)

        pred_category = label_encoder.inverse_transform([pred_encoded])[0]
        pred_binary = map_category_to_binary(pred_category)
        pred_decision = map_binary_to_decision(pred_binary)

        # Confidence
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(single_x)[0]
            confidence = float(np.max(probs))
        else:
            confidence = np.nan

        predicted_categories.append(pred_category)
        predicted_binary_labels.append(pred_binary)
        predicted_decisions.append(pred_decision)
        confidences.append(confidence)

    # -----------------------------
    # True labels
    # -----------------------------
    true_binary_labels = [
        map_category_to_binary(lbl) for lbl in baseline_df["true_label"]
    ]

    true_decisions = [
        map_binary_to_decision(lbl) for lbl in true_binary_labels
    ]

    # -----------------------------
    # Save Track B prediction results
    # -----------------------------
    results_df = pd.DataFrame({
        "prompt": prompts,
        "true_label": baseline_df["true_label"],
        "true_binary_label": true_binary_labels,
        "predicted_category": predicted_categories,
        "predicted_binary_label": predicted_binary_labels,
        "decision": predicted_decisions,
        "confidence": confidences,
        "latency_sec": latencies
    })

    results_df.to_csv("Data/trackb_model_results.csv", index=False)

    print("\nSaved:")
    print("- trackb_model_results.csv")

    # -----------------------------
    # Track B binary safe/unsafe evaluation
    # -----------------------------
    print("\n=== TRACK B MODEL METRICS (Safe vs Unsafe) ===")
    print("Accuracy:", round(accuracy_score(true_binary_labels, predicted_binary_labels), 4))
    print(
        classification_report(
            true_binary_labels,
            predicted_binary_labels,
            digits=4,
            zero_division=0
        )
    )

    # -----------------------------
    # Latency metrics
    # -----------------------------
    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    print("\n=== LATENCY ===")
    print("Average latency (seconds):", round(avg_latency, 6))
    print("P95 latency (seconds):", round(p95_latency, 6))

    # -----------------------------
    # Baseline decision-based evaluation
    # -----------------------------
    y_pred_baseline = baseline_df["decision"].astype(str).tolist()
    y_pred_trackb = results_df["decision"].astype(str).tolist()

    print("\n=== BASELINE MODEL (Decision-based) ===")
    print("Accuracy:", round(accuracy_score(true_decisions, y_pred_baseline), 4))
    print(
        classification_report(
            true_decisions,
            y_pred_baseline,
            digits=4,
            zero_division=0
        )
    )

    print("\n=== TRACK B MODEL (Decision-based) ===")
    print("Accuracy:", round(accuracy_score(true_decisions, y_pred_trackb), 4))
    print(
        classification_report(
            true_decisions,
            y_pred_trackb,
            digits=4,
            zero_division=0
        )
    )

    # -----------------------------
    # Save final comparison summary
    # -----------------------------
    baseline_report = classification_report(
        true_decisions,
        y_pred_baseline,
        output_dict=True,
        zero_division=0
    )

    trackb_report = classification_report(
        true_decisions,
        y_pred_trackb,
        output_dict=True,
        zero_division=0
    )

    comparison_summary = pd.DataFrame({
        "Model": ["Baseline", "Track B"],
        "Accuracy": [
            accuracy_score(true_decisions, y_pred_baseline),
            accuracy_score(true_decisions, y_pred_trackb)
        ],
        "Recall (BLOCK)": [
            baseline_report["BLOCK"]["recall"],
            trackb_report["BLOCK"]["recall"]
        ],
        "F1 Score (BLOCK)": [
            baseline_report["BLOCK"]["f1-score"],
            trackb_report["BLOCK"]["f1-score"]
        ],
        "P95 Latency (sec)": [
            np.nan,
            p95_latency
        ]
    })

    comparison_summary.to_csv("Data/trackb_vs_baseline_comparison.csv", index=False)

    print("\nSaved:")
    print("- trackb_vs_baseline_comparison.csv")

    print("\nComparison Summary:")
    print(comparison_summary)


if __name__ == "__main__":
    main()