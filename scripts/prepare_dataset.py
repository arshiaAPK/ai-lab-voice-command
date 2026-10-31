from pathlib import Path
import pandas as pd
from scipy.io import wavfile
import numpy as np

ROOT = Path("data/raw/tts")
OUT = Path("data/processed/manifest.csv")

LABELS = ["روشن", "خاموش", "شروع", "توقف", "کمک"]

TRAIN_SPEAKERS = {"voice01"}
TEST_SPEAKERS = {"voice02"}

MIN_SEC = 0.3
MAX_SEC = 3.0
EXPECTED_RATE = 16000


def main():
    rows = []
    errors = []

    for path in ROOT.glob("*/*/*.wav"):
        speaker = path.parent.parent.name
        label = path.parent.name

        if label not in LABELS:
            errors.append(f"{path}: برچسب ناشناخته")
            continue

        if speaker not in TRAIN_SPEAKERS | TEST_SPEAKERS:
            errors.append(f"{path}: شناسه گوینده ناشناخته")
            continue

        try:
            rate, audio = wavfile.read(path)

            if audio.ndim != 1:
                errors.append(f"{path}: فایل باید Mono باشد")
                continue

            if rate != EXPECTED_RATE:
                errors.append(f"{path}: نرخ نمونه‌برداری نامعتبر")
                continue

            if audio.dtype != np.int16:
                errors.append(f"{path}: فرمت باید PCM 16-bit باشد")
                continue

            duration = len(audio) / rate

            if not MIN_SEC <= duration <= MAX_SEC:
                errors.append(f"{path}: مدت فایل نامعتبر است")
                continue

            if np.max(np.abs(audio.astype(np.int32))) < 150:
                errors.append(f"{path}: احتمال سکوت")
                continue

            split = (
                "train"
                if speaker in TRAIN_SPEAKERS
                else "test"
            )

            rows.append({
                "file_path": path.as_posix(),
                "speaker_id": speaker,
                "label": label,
                "duration_seconds": round(duration, 3),
                "sample_rate": rate,
                "split": split
            })

        except Exception as exc:
            errors.append(f"{path}: {exc}")

    if errors:
        print("موارد نامعتبر:")
        for error in errors:
            print("-", error)

    if not rows:
        raise SystemExit("هیچ فایل معتبر صوتی پیدا نشد.")

    df = pd.DataFrame(rows)

    for split in ["train", "test"]:
        present = set(df.loc[df["split"] == split, "label"])
        missing = set(LABELS) - present

        if missing:
            raise SystemExit(
                f"کلاس‌های فاقد نمونه در {split}: {sorted(missing)}"
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)

    df = df.sort_values(
        ["split", "speaker_id", "label", "file_path"]
    )

    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    print("\nManifest ساخته شد:", OUT)
    print("تعداد فایل معتبر:", len(df))

    print("\nتعداد نمونه در هر کلاس و بخش:")
    print(pd.crosstab(df["label"], df["split"]))

    print("\nگویندگان:")
    print(df.groupby("split")["speaker_id"].unique())


if __name__ == "__main__":
    main()