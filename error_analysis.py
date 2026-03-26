import pandas as pd

def main():
    df = pd.read_csv("Data/trackb_model_results.csv")

    # Normalize
    df["true_binary_label"] = df["true_binary_label"].astype(str).str.lower()
    df["predicted_binary_label"] = df["predicted_binary_label"].astype(str).str.lower()

    # Create decision columns
    df["true_decision"] = df["true_binary_label"].apply(lambda x: "ALLOW" if x == "safe" else "BLOCK")
    df["predicted_decision"] = df["predicted_binary_label"].apply(lambda x: "ALLOW" if x == "safe" else "BLOCK")

    # Error type
    def get_error_type(row):
        if row["true_decision"] == "ALLOW" and row["predicted_decision"] == "BLOCK":
            return "False Positive"
        elif row["true_decision"] == "BLOCK" and row["predicted_decision"] == "ALLOW":
            return "False Negative"
        elif row["true_decision"] == "ALLOW" and row["predicted_decision"] == "ALLOW":
            return "True Negative"
        else:
            return "True Positive"

    df["error_type"] = df.apply(get_error_type, axis=1)

    # Add simple pattern notes
    def add_pattern_note(prompt):
        text = str(prompt).lower()

        if "ignore" in text or "override" in text or "bypass" in text:
            return "Direct instruction override / jailbreak style"
        elif "pretend" in text or "act as" in text or "roleplay" in text:
            return "Roleplay-based attack phrasing"
        elif "for research" in text or "educational" in text or "for a class" in text:
            return "Educational disguise phrasing"
        elif "private" in text or "password" in text or "confidential" in text:
            return "Sensitive data / privacy-related wording"
        elif "urgent" in text or "emergency" in text:
            return "Pressure / urgency framing"
        elif "system prompt" in text or "hidden instructions" in text:
            return "System prompt extraction attempt"
        else:
            return "General phrasing / no obvious pattern"

    df["pattern_note"] = df["prompt"].apply(add_pattern_note)

    # Save all results
    df.to_csv("Data/trackb_error_analysis_full.csv", index=False)

    # Save only mistakes
    errors_df = df[df["error_type"].isin(["False Positive", "False Negative"])].copy()
    errors_df.to_csv("Data/trackb_error_cases_only.csv", index=False)

    # Print summary
    print("Saved:")
    print("- trackb_error_analysis_full.csv")
    print("- trackb_error_cases_only.csv")

    print("\nError Summary:")
    print(df["error_type"].value_counts())

    print("\nFalse Positives:")
    print((df["error_type"] == "False Positive").sum())

    print("False Negatives:")
    print((df["error_type"] == "False Negative").sum())

if __name__ == "__main__":
    main()