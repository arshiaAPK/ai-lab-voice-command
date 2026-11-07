from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CSV = Path("data/processed/manifest.csv")
OUT = Path("reports/eda")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate file paths:", df["file_path"].duplicated().sum())
print("\nCategorical summaries:")
for col in ["speaker_id", "label", "split", "sample_rate"]:
    print(f"\n{col}:\n", df[col].value_counts(dropna=False))
print("\nDuration summary:\n", df["duration_seconds"].describe())
print("\nDuration by command:\n",
      df.groupby("label")["duration_seconds"].agg(
          ["count", "min", "mean", "median", "max", "std"]
      ).round(3))

# Class distribution
df["label"].value_counts().sort_index().plot(kind="bar", title="Samples per command")
plt.xlabel("Command"); plt.ylabel("Count"); plt.tight_layout()
plt.savefig(OUT / "class_distribution.png", dpi=160); plt.close()

# Train/test by class
pd.crosstab(df["label"], df["split"]).plot(kind="bar", title="Train/test by command")
plt.xlabel("Command"); plt.ylabel("Count"); plt.tight_layout()
plt.savefig(OUT / "train_test_distribution.png", dpi=160); plt.close()

# Duration histogram
df["duration_seconds"].plot(kind="hist", bins=12, title="Audio duration distribution")
plt.xlabel("Duration (seconds)"); plt.tight_layout()
plt.savefig(OUT / "duration_histogram.png", dpi=160); plt.close()

# Duration by class
df.boxplot(column="duration_seconds", by="label")
plt.title("Duration by command"); plt.suptitle("")
plt.xlabel("Command"); plt.ylabel("Duration (seconds)")
plt.tight_layout(); plt.savefig(OUT / "duration_by_class_boxplot.png", dpi=160); plt.close()

print(f"\nCharts saved in: {OUT}")
