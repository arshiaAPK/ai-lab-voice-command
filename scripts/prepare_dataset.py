from pathlib import Path
import pandas as pd
from scipy.io import wavfile
import numpy as np

ROOT = Path("data/raw")
OUT = Path("data/processed/manifest.csv")
LABELS = ["روشن", "خاموش", "شروع", "توقف", "کمک"]
MIN_SEC, MAX_SEC, EXPECTED_RATE = 0.3, 3.0, 16000

def read_ids(path):
    return {x.strip() for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip() and not x.strip().startswith("#")}

def main():
    train = read_ids(Path("splits/train_speakers.txt"))
    test = read_ids(Path("splits/test_speakers.txt"))
    if not train or not test:
        raise SystemExit("فایل‌های split نباید خالی باشند.")
    if train & test:
        raise SystemExit(f"خطا: گوینده مشترک در train و test: {sorted(train & test)}")

    rows, errors = [], []
    for path in ROOT.glob("*/*/*.wav"):
        speaker, label = path.parent.parent.name, path.parent.name
        if label not in LABELS:
            errors.append(f"{path}: برچسب ناشناخته")
            continue
        try:
            rate, audio = wavfile.read(path)
            if audio.ndim != 1:
                errors.append(f"{path}: فایل باید Mono باشد"); continue
            if rate != EXPECTED_RATE:
                errors.append(f"{path}: نرخ نمونه‌برداری {rate} است، باید 16000 باشد"); continue
            if audio.dtype != np.int16:
                errors.append(f"{path}: فرمت نمونه باید PCM 16-bit باشد"); continue
            duration = len(audio) / rate
            if not (MIN_SEC <= duration <= MAX_SEC):
                errors.append(f"{path}: مدت {duration:.2f}s خارج از بازه مجاز است"); continue
            if np.max(np.abs(audio.astype(np.int32))) < 150:
                errors.append(f"{path}: احتمال سکوت یا صدای بسیار ضعیف"); continue
            if speaker not in train | test:
                errors.append(f"{path}: گوینده در فایل‌های split تعریف نشده"); continue
            rows.append({
                "file_path": path.as_posix(),
                "speaker_id": speaker,
                "label": label,
                "duration_seconds": round(duration, 3),
                "sample_rate": rate,
                "split": "train" if speaker in train else "test"
            })
        except Exception as exc:
            errors.append(f"{path}: خطا در خواندن فایل ({exc})")

    if errors:
        print("موارد نیازمند اصلاح:")
        for e in errors: print("-", e)

    if not rows:
        raise SystemExit("هیچ فایل معتبر صوتی پیدا نشد.")
    df = pd.DataFrame(rows).sort_values(["split", "speaker_id", "label", "file_path"])
    # Require every command in both splits; otherwise class-wise evaluation is incomplete.
    for split in ["train", "test"]:
        present = set(df.loc[df.split == split, "label"])
        missing = set(LABELS) - present
        if missing:
            raise SystemExit(f"در بخش {split} این کلاس‌ها نمونه ندارند: {sorted(missing)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\nManifest ساخته شد: {OUT}")
    print(f"تعداد فایل معتبر: {len(df)}")
    print("\nتعداد نمونه در هر کلاس و split:")
    print(pd.crosstab(df["label"], df["split"]).to_string())
    print("\nگویندگان:")
    print(df.groupby("split")["speaker_id"].unique().to_string())

if __name__ == "__main__":
    main()
