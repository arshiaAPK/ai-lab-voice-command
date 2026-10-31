import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABELS = ["روشن", "خاموش", "شروع", "توقف", "کمک"]

def main():
    p = argparse.ArgumentParser(description="Evaluate predictions on held-out test audio.")
    p.add_argument("--input", required=True, help="CSV with true_label,predicted_label")
    args = p.parse_args()
    df = pd.read_csv(args.input)
    required = {"true_label", "predicted_label"}
    if not required.issubset(df.columns):
        raise SystemExit("CSV باید ستون‌های true_label و predicted_label داشته باشد.")
    if df.empty:
        raise SystemExit("فایل پیش‌بینی خالی است.")
    unknown = (set(df.true_label.dropna()) | set(df.predicted_label.dropna())) - set(LABELS)
    if unknown:
        raise SystemExit(f"برچسب ناشناخته: {sorted(unknown)}")
    if df[["true_label", "predicted_label"]].isna().any().any():
        raise SystemExit("مقادیر خالی در برچسب‌ها وجود دارد.")
    print(f"تعداد نمونه آزمون: {len(df)}")
    print(f"Accuracy: {accuracy_score(df.true_label, df.predicted_label):.4f}")
    print("\nگزارش هر کلاس:")
    print(classification_report(df.true_label, df.predicted_label,
                                labels=LABELS, zero_division=0, digits=4))
    print("Confusion matrix (ردیف=واقعی، ستون=پیش‌بینی‌شده):")
    print(pd.DataFrame(confusion_matrix(df.true_label, df.predicted_label,
                                        labels=LABELS), index=LABELS, columns=LABELS))

if __name__ == "__main__":
    main()
