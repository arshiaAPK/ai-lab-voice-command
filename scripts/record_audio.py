import argparse
from pathlib import Path
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

LABELS = ["روشن", "خاموش", "شروع", "توقف", "کمک"]

def main():
    p = argparse.ArgumentParser(description="Record one command as 16 kHz mono PCM WAV.")
    p.add_argument("--speaker", required=True, help="Anonymous ID, e.g. spk01")
    p.add_argument("--label", required=True, choices=LABELS)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--root", default="data/raw")
    args = p.parse_args()

    if args.count < 1 or not (0.3 <= args.seconds <= 5):
        raise SystemExit("count must be >=1 and seconds must be between 0.3 and 5.")

    out = Path(args.root) / args.speaker / args.label
    out.mkdir(parents=True, exist_ok=True)
    rate = 16000

    for _ in range(args.count):
        # Choose the next unused index; do not rely on the number of existing files.
        index = 1
        while (out / f"{args.speaker}_{args.label}_{index:03d}.wav").exists():
            index += 1
        input(f"نمونه {index}: آماده شوید؛ Enter را بزنید و دستور «{args.label}» را واضح بگویید...")
        print("در حال ضبط...")
        audio = sd.rec(int(args.seconds * rate), samplerate=rate, channels=1, dtype="int16")
        sd.wait()
        if np.max(np.abs(audio.astype(np.int32))) < 150:
            print("هشدار: صدای ضبط‌شده بسیار ضعیف/احتمالاً سکوت است؛ این نمونه ذخیره نشد.")
            continue
        path = out / f"{args.speaker}_{args.label}_{index:03d}.wav"
        write(str(path), rate, audio)
        print(f"ذخیره شد: {path}")

if __name__ == "__main__":
    main()
