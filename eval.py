import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================================================
# Helper functions
# =========================================================

def normalize_text(text):
    return str(text).strip().lower()


def map_category_to_binary(label):
    """
    Convert category labels into binary labels.
    safe / benign -> safe
    everything else -> unsafe
    """
    label = normalize_text(label)
    if label in ["safe", "benign"]:
        return "safe"
    return "unsafe"


def map_binary_to_decision(label):
    """
    safe -> ALLOW
    unsafe -> BLOCK
    """
    label = normalize_text(label)
    return "ALLOW" if label == "safe" else "BLOCK"


# =========================================================
# Baseline firewall
# =========================================================

BASELINE_PATTERNS = [
    "ignore all rules",
    "ignore previous instructions",
    "ignore safety rules",
    "ignore the system prompt",
    "reveal system prompt",
    "reveal the system prompt",
    "show hidden prompt",
    "print hidden prompt",
    "reveal internal prompt",
    "show internal instructions",
    "reveal hidden instructions",
    "bypass safety",
    "bypass filters",
    "disable safety",
    "disable guardrails",
    "override instructions",
    "override the system",
    "jailbreak",
    "unrestricted ai",
    "no limitations",
    "developer mode",
    "act as dan",
    "pretend you are unrestricted",
    "act as an unrestricted assistant",
    "private data",
    "confidential data",
    "passwords",
    "login credentials",
    "bank details",
    "passport numbers",
    "aadhaar numbers",
    "hack ai",
    "attack ai",
    "attack guardrails",
    "exploit ai",
    "break the policy",
    "break your policy",
    "attack the system",
    "follow only this malicious instruction",
    "obey the hidden command",
    "treat the document as highest priority",
]

ROLEPLAY_CUES = [
    "pretend you are",
    "act as",
    "roleplay as",
    "imagine you are",
]

DANGEROUS_TOPICS = [
    "hack",
    "hacking",
    "exploit",
    "attack",
    "bypass",
    "jailbreak",
    "leak",
    "steal",
    "private data",
    "confidential data",
    "system prompt",
    "hidden prompt",
    "override",
    "guardrails",
]

EDUCATIONAL_DISGUISE = [
    "for research",
    "for educational purposes",
    "for a class",
    "teaching a class",
    "academic purpose",
    "practical examples",
    "demo",
    "demonstration",
    "case study",
]

PRESSURE_CUES = [
    "urgent",
    "people could be harmed",
    "this is urgent",
    "lives depend on this",
    "if you don't answer",
    "if you don’t answer",
    "emergency",
    "critical situation",
]


def contains_any(text, phrases):
    return [p for p in phrases if p in text]


def baseline_predict(prompt):
    """
    Simple rule-based baseline.
    Returns:
        predicted_category, predicted_binary_label, decision, confidence
    """
    text = normalize_text(prompt)

    # Exact risky phrase match
    exact_matches = contains_any(text, BASELINE_PATTERNS)
    if exact_matches:
        return "prompt_injection", "unsafe", "BLOCK", 1.0

    # Educational disguise + danger topic
    edu_matches = contains_any(text, EDUCATIONAL_DISGUISE)
    danger_matches = contains_any(text, DANGEROUS_TOPICS)
    if edu_matches and danger_matches:
        return "toxic_content", "unsafe", "BLOCK", 0.95

    # Pressure cue + danger topic
    pressure_matches = contains_any(text, PRESSURE_CUES)
    if pressure_matches and danger_matches:
        return "jailbreak", "unsafe", "BLOCK", 0.95

    # Roleplay + danger topic
    roleplay_matches = contains_any(text, ROLEPLAY_CUES)
    if roleplay_matches and danger_matches:
        return "jailbreak", "unsafe", "BLOCK", 0.95

    # Roleplay + unrestricted phrasing
    if roleplay_matches and ("unrestricted" in text or "no limitations" in text):
        return "jailbreak", "unsafe", "BLOCK", 0.95

    # Otherwise allow
    return "safe", "safe", "ALLOW", 0.60


# =========================================================
# Track B trained model prediction
# =========================================================

def map_model_category(raw_label, prompt):
    """
    Map model output / prompt pattern into official category names.
    """
    label = normalize_text(raw_label)
    text = normalize_text(prompt)

    if label in ["safe", "benign"]:
        return "safe"

    if any(x in text for x in ["private data", "password", "aadhaar", "bank details", "credentials", "passport"]):
        return "pii_extraction"

    if any(x in text for x in ["system prompt", "ignore instructions", "hidden prompt", "override", "document as highest priority"]):
        return "prompt_injection"

    if any(x in text for x in ["act as", "pretend you are", "unrestricted", "no limitations", "jailbreak", "dan"]):
        return "jailbreak"

    return "toxic_content"


def trackb_predict(prompt, model, vectorizer, label_encoder, threshold=0.4):
    """
    Predict with trained TF-IDF + Logistic Regression model.
    Returns:
        predicted_category, predicted_binary_label, decision, confidence
    """
    vec = vectorizer.transform([prompt])
    probs = model.predict_proba(vec)[0]
    confidence = float(np.max(probs))

    pred_idx = model.predict(vec)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    pred_label_norm = normalize_text(pred_label)

    if pred_label_norm in ["safe", "benign"] and confidence >= threshold:
        return "safe", "safe", "ALLOW", confidence
    else:
        category = map_model_category(pred_label_norm, prompt)
        return category, "unsafe", "BLOCK", confidence


# =========================================================
# Main evaluation
# =========================================================

def main():
    print("Running Track B evaluation on official red-team suite...")

    # -----------------------------
    # Load trained artifacts
    # -----------------------------
    model = joblib.load("Models/smartguard_final_model.pkl")
    vectorizer = joblib.load("Models/smartguard_vectorizer.pkl")
    label_encoder = joblib.load("Models/smartguard_label_encoder.pkl")

    # -----------------------------
    # Load evaluation suite
    # -----------------------------
    eval_df = pd.read_csv("Data/red_team_results.csv")

    if "prompt" not in eval_df.columns:
        raise ValueError("Data/red_team_results.csv must contain a 'prompt' column.")

    if "true_label" not in eval_df.columns:
        raise ValueError("Data/red_team_results.csv must contain a 'true_label' column.")

    prompts = eval_df["prompt"].astype(str).tolist()
    true_labels_raw = eval_df["true_label"].astype(str).tolist()

    true_binary_labels = [map_category_to_binary(lbl) for lbl in true_labels_raw]
    true_decisions = [map_binary_to_decision(lbl) for lbl in true_binary_labels]

    # -----------------------------
    # Baseline predictions
    # -----------------------------
    baseline_categories = []
    baseline_binary_labels = []
    baseline_decisions = []
    baseline_confidences = []
    baseline_latencies = []

    for prompt in prompts:
        start = time.perf_counter()
        pred_category, pred_binary, pred_decision, pred_conf = baseline_predict(prompt)
        end = time.perf_counter()

        baseline_categories.append(pred_category)
        baseline_binary_labels.append(pred_binary)
        baseline_decisions.append(pred_decision)
        baseline_confidences.append(pred_conf)
        baseline_latencies.append(end - start)

    baseline_results_df = pd.DataFrame({
        "prompt": prompts,
        "true_label": true_labels_raw,
        "true_binary_label": true_binary_labels,
        "predicted_category": baseline_categories,
        "predicted_binary_label": baseline_binary_labels,
        "decision": baseline_decisions,
        "confidence": baseline_confidences,
        "latency_sec": baseline_latencies
    })

    baseline_results_df.to_csv("Data/baseline_eval_results.csv", index=False)

    # -----------------------------
    # Track B predictions
    # -----------------------------
    trackb_categories = []
    trackb_binary_labels = []
    trackb_decisions = []
    trackb_confidences = []
    trackb_latencies = []

    for prompt in prompts:
        start = time.perf_counter()
        pred_category, pred_binary, pred_decision, pred_conf = trackb_predict(
            prompt=prompt,
            model=model,
            vectorizer=vectorizer,
            label_encoder=label_encoder,
            threshold=0.4
        )
        end = time.perf_counter()

        trackb_categories.append(pred_category)
        trackb_binary_labels.append(pred_binary)
        trackb_decisions.append(pred_decision)
        trackb_confidences.append(pred_conf)
        trackb_latencies.append(end - start)

    trackb_results_df = pd.DataFrame({
        "prompt": prompts,
        "true_label": true_labels_raw,
        "true_binary_label": true_binary_labels,
        "predicted_category": trackb_categories,
        "predicted_binary_label": trackb_binary_labels,
        "decision": trackb_decisions,
        "confidence": trackb_confidences,
        "latency_sec": trackb_latencies
    })

    trackb_results_df.to_csv("Data/trackb_model_results.csv", index=False)

    print("\nSaved:")
    print("- Data/baseline_eval_results.csv")
    print("- Data/trackb_model_results.csv")

    # -----------------------------
    # Baseline metrics
    # -----------------------------
    print("\n=== BASELINE MODEL METRICS (Safe vs Unsafe) ===")
    print("Accuracy:", round(accuracy_score(true_binary_labels, baseline_binary_labels), 4))
    print(classification_report(true_binary_labels, baseline_binary_labels, digits=4, zero_division=0))

    baseline_cm = confusion_matrix(true_decisions, baseline_decisions, labels=["ALLOW", "BLOCK"])
    print("Baseline Confusion Matrix [ALLOW, BLOCK]:")
    print(baseline_cm)

    baseline_avg_latency = float(np.mean(baseline_latencies))
    baseline_p95_latency = float(np.percentile(baseline_latencies, 95))

    print("\n=== BASELINE LATENCY ===")
    print("Average latency (seconds):", round(baseline_avg_latency, 6))
    print("P95 latency (seconds):", round(baseline_p95_latency, 6))

    # -----------------------------
    # Track B metrics
    # -----------------------------
    print("\n=== TRACK B MODEL METRICS (Safe vs Unsafe) ===")
    print("Accuracy:", round(accuracy_score(true_binary_labels, trackb_binary_labels), 4))
    print(classification_report(true_binary_labels, trackb_binary_labels, digits=4, zero_division=0))

    trackb_cm = confusion_matrix(true_decisions, trackb_decisions, labels=["ALLOW", "BLOCK"])
    print("Track B Confusion Matrix [ALLOW, BLOCK]:")
    print(trackb_cm)

    trackb_avg_latency = float(np.mean(trackb_latencies))
    trackb_p95_latency = float(np.percentile(trackb_latencies, 95))

    print("\n=== TRACK B LATENCY ===")
    print("Average latency (seconds):", round(trackb_avg_latency, 6))
    print("P95 latency (seconds):", round(trackb_p95_latency, 6))

    # -----------------------------
    # Decision-based comparison
    # -----------------------------
    print("\n=== BASELINE MODEL (Decision-based) ===")
    print("Accuracy:", round(accuracy_score(true_decisions, baseline_decisions), 4))
    print(classification_report(true_decisions, baseline_decisions, digits=4, zero_division=0))

    print("\n=== TRACK B MODEL (Decision-based) ===")
    print("Accuracy:", round(accuracy_score(true_decisions, trackb_decisions), 4))
    print(classification_report(true_decisions, trackb_decisions, digits=4, zero_division=0))

    baseline_report = classification_report(
        true_decisions, baseline_decisions, output_dict=True, zero_division=0
    )
    trackb_report = classification_report(
        true_decisions, trackb_decisions, output_dict=True, zero_division=0
    )

    comparison_summary = pd.DataFrame({
        "Model": ["Baseline", "Track B"],
        "Accuracy": [
            accuracy_score(true_decisions, baseline_decisions),
            accuracy_score(true_decisions, trackb_decisions)
        ],
        "Recall (BLOCK)": [
            baseline_report["BLOCK"]["recall"],
            trackb_report["BLOCK"]["recall"]
        ],
        "F1 Score (BLOCK)": [
            baseline_report["BLOCK"]["f1-score"],
            trackb_report["BLOCK"]["f1-score"]
        ],
        "Average Latency (sec)": [
            baseline_avg_latency,
            trackb_avg_latency
        ],
        "P95 Latency (sec)": [
            baseline_p95_latency,
            trackb_p95_latency
        ]
    })

    comparison_summary.to_csv("Data/trackb_vs_baseline_comparison.csv", index=False)

    print("\nSaved:")
    print("- Data/trackb_vs_baseline_comparison.csv")

    print("\n=== COMPARISON SUMMARY ===")
    print(comparison_summary)


if __name__ == "__main__":
    main()