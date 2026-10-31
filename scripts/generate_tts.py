
import asyncio
import subprocess
from pathlib import Path

import edge_tts
import imageio_ffmpeg

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "raw" / "tts"

# دو صدای فارسی
VOICES = {
    "voice01": "fa-IR-DilaraNeural",
    "voice02": "fa-IR-FaridNeural",
}

# عبارت‌های متنوع برای هر فرمان
COMMANDS = {
    "روشن": [
        "روشن",
        "روشن کن",
        "دستگاه را روشن کن",
        "لطفاً روشن کن",
        "چراغ را روشن کن",
    ],
    "خاموش": [
        "خاموش",
        "خاموش کن",
        "دستگاه را خاموش کن",
        "لطفاً خاموش کن",
        "چراغ را خاموش کن",
    ],
    "شروع": [
        "شروع",
        "شروع کن",
        "کار را شروع کن",
        "فرایند را شروع کن",
        "لطفاً شروع کن",
    ],
    "توقف": [
        "توقف",
        "توقف کن",
        "کار را متوقف کن",
        "لطفاً توقف کن",
        "دستگاه را متوقف کن",
    ],
    "کمک": [
        "کمک",
        "کمک کن",
        "به کمک نیاز دارم",
        "لطفاً کمک کن",
        "من کمک می‌خواهم",
    ],
}

async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    for speaker, voice in VOICES.items():
        for label, phrases in COMMANDS.items():

            folder = OUTPUT_DIR / speaker / label
            folder.mkdir(parents=True, exist_ok=True)

            for i, phrase in enumerate(phrases, start=1):
                for speed in ["+0%", "-10%"]:
                    name = f"{label}_{i:02d}_{speed.replace('%', '')}.wav"
                    wav_path = folder / name
                    mp3_path = folder / (name + ".mp3")

                    if wav_path.exists():
                        continue

                    tts = edge_tts.Communicate(
                        phrase,
                        voice,
                        rate=speed
                    )
                    await tts.save(str(mp3_path))

                    subprocess.run(
                        [
                            ffmpeg, "-y",
                            "-i", str(mp3_path),
                            "-ac", "1",
                            "-ar", "16000",
                            "-sample_fmt", "s16",
                            str(wav_path)
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )

                    mp3_path.unlink()
                    print("Created:", wav_path.relative_to(BASE_DIR))

    print("\nتولید فایل‌های صوتی تمام شد.")

if __name__ == "__main__":
    asyncio.run(main())
