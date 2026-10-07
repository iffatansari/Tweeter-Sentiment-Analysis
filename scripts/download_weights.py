"""
Utility script to download required Transformer & Whisper model weights
directly into the local 'models/' directory using huggingface_hub.
"""
from pathlib import Path
import shutil
import urllib.request
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]

ROBERTA_DIR = ROOT / "models" / "twitter-roberta-base-sentiment"
DEBERTA_DIR = ROOT / "models" / "sarcasm-deberta-v3"
WHISPER_DIR = ROOT / "models" / "whisper"

def download_models():
    print("=" * 60)
    print("DOWNLOADING MODEL WEIGHTS")
    print("=" * 60)

    # 1. CardiffNLP Twitter-RoBERTa
    print("\n1/3 Downloading Twitter-RoBERTa weights (cardiffnlp/twitter-roberta-base-sentiment)...")
    ROBERTA_DIR.mkdir(parents=True, exist_ok=True)
    target_roberta = ROBERTA_DIR / "pytorch_model.bin"
    if not target_roberta.exists():
        downloaded = hf_hub_download(
            repo_id="cardiffnlp/twitter-roberta-base-sentiment",
            filename="pytorch_model.bin",
        )
        shutil.copy(downloaded, target_roberta)
        print(f" -> Saved to {target_roberta}")
    else:
        print(" -> Twitter-RoBERTa weights already present.")

    # 2. DeBERTa Sarcasm
    print("\n2/3 Downloading DeBERTa Sarcasm weights (ppokhrel2109/besstie-sarcasm-deberta-v3)...")
    DEBERTA_DIR.mkdir(parents=True, exist_ok=True)
    target_deberta = DEBERTA_DIR / "model.safetensors"
    if not target_deberta.exists():
        downloaded = hf_hub_download(
            repo_id="ppokhrel2109/besstie-sarcasm-deberta-v3",
            filename="model.safetensors",
        )
        shutil.copy(downloaded, target_deberta)
        print(f" -> Saved to {target_deberta}")
    else:
        print(" -> DeBERTa Sarcasm weights already present.")

    # 3. Whisper Tiny
    print("\n3/3 Downloading Whisper Tiny weights...")
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    target_whisper = WHISPER_DIR / "tiny.pt"
    if not target_whisper.exists():
        whisper_url = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d1260e3bae037e257f381305c62626116e69f5b46643ac006415/tiny.pt"
        urllib.request.urlretrieve(whisper_url, str(target_whisper))
        print(f" -> Saved to {target_whisper}")
    else:
        print(" -> Whisper Tiny weights already present.")

    print("\n" + "=" * 60)
    print("ALL WEIGHTS DOWNLOADED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    download_models()
