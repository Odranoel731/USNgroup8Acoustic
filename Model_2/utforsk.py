import sys
from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Relativ sti: Data_sett-mappen ligger ved siden av dette scriptet
DATA_DIR = Path(__file__).parent / "Data_sett" / "IDMT-ISA-ELECTRIC-ENGINE"
SPLIT = "train_cut"
FIL = "pure_0.wav"
KLASSER = {
    "good": "engine1_good",
    "broken": "engine2_broken",
    "heavy load": "engine3_heavyload",
}

if not DATA_DIR.exists():
    sys.exit(f"Finner ikke datasettet i {DATA_DIR}\n"
             "Last ned IDMT-ISA-ELECTRIC-ENGINE fra Zenodo og pakk det ut i Data_sett-mappen.")

fig, axes = plt.subplots(3, 3, figsize=(16, 10))

for i, (navn, mappe) in enumerate(KLASSER.items()):
    sti = DATA_DIR / SPLIT / mappe / FIL
    y, sr = librosa.load(sti, sr=None)  # sr=None beholder 44,1 kHz
    t = np.arange(len(y)) / sr

    # Enkle tidsdomene-features
    rms = np.sqrt(np.mean(y**2))
    peak = np.max(np.abs(y))
    crest = peak / rms

    # FFT av hele klippet
    spekter = np.abs(np.fft.rfft(y))
    frekvenser = np.fft.rfftfreq(len(y), 1 / sr)
    spekter_db = 20 * np.log10(spekter + 1e-12)
    over_20hz = frekvenser > 20  # hopp over DC og helt lave frekvenser
    topp_frekvens = frekvenser[over_20hz][np.argmax(spekter[over_20hz])]

    print(f"{navn:11s} RMS={rms:.4f}  peak={peak:.4f}  crest={crest:.2f}  "
          f"kraftigste frekvens={topp_frekvens:.1f} Hz")

    # 1) Bølgeform
    axes[i, 0].plot(t, y, linewidth=0.4)
    axes[i, 0].set_title(f"{navn}: bølgeform")
    axes[i, 0].set_xlabel("Tid (s)")

    # 2) Frekvensspekter
    axes[i, 1].plot(frekvenser, spekter_db, linewidth=0.4)
    axes[i, 1].set_xlim(0, 5000)
    axes[i, 1].set_title(f"{navn}: FFT-spekter")
    axes[i, 1].set_xlabel("Frekvens (Hz)")
    axes[i, 1].set_ylabel("dB")

    # 3) Log-mel-spektrogram (det CNN-en skal få som input)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048,
                                         hop_length=512, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=sr, hop_length=512,
                             x_axis="time", y_axis="mel", ax=axes[i, 2])
    axes[i, 2].set_title(f"{navn}: log-mel-spektrogram")

plt.tight_layout()
plt.show()