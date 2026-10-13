"""
preprocess.py
Lager log-mel-spektrogrammer av alle 3-sekunders klipp i train_cut og test_cut,
og lagrer dem som .npy-filer i features/. Kjøres én gang (tar et par minutter).
"""
import sys
from pathlib import Path
import numpy as np
import librosa

# Relative stier: fungerer på alle PC-er med samme mappestruktur
PROSJEKT = Path(__file__).parent
DATA_DIR = PROSJEKT / "Data_sett" / "IDMT-ISA-ELECTRIC-ENGINE"
FEATURES = PROSJEKT / "features"

# Spektrogram-innstillinger
SR = 22050     # samplingsrate etter resampling (Hz)
N_FFT = 1024   # vinduslengde (ca. 46 ms)
HOP = 256      # steg mellom vinduer -> 259 tidsrammer per klipp
N_MELS = 64    # antall mel-frekvensbånd

KLASSER = {"engine1_good": 0, "engine2_broken": 1, "engine3_heavyload": 2}

# Stopp tidlig hvis datasettet ikke ligger der det skal
if not DATA_DIR.exists():
    sys.exit(f"Finner ikke datasettet i {DATA_DIR}\n"
             "Last ned IDMT-ISA-ELECTRIC-ENGINE fra Zenodo og pakk det ut i Data_sett-mappen.")

FEATURES.mkdir(exist_ok=True)


def klipp_info(fil):
    """'atmo_high_12.wav' -> ('atmo_high', 12)"""
    navn, nummer = fil.stem.rsplit("_", 1)
    return navn, int(nummer)


def lag_spektrogram(fil):
    """Leser én lydfil og returnerer log-mel-spektrogram (64 x 259)."""
    y, _ = librosa.load(fil, sr=SR)  # leser og resampler til 22 050 Hz
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT,
                                         hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=1.0)  # dB, felles referanse for alle klipp


def behandle(split):
    """Går gjennom alle klippene i én split (train_cut eller test_cut)."""
    X, y, cond = [], [], []
    for mappe, klasse in KLASSER.items():
        # Sorter etter støyforhold og klippnummer (pure_0, pure_1, pure_2, ...)
        filer = sorted((DATA_DIR / split / mappe).glob("*.wav"), key=klipp_info)
        if not filer:
            sys.exit(f"Fant ingen WAV-filer i {DATA_DIR / split / mappe}")
        print(f"{split}/{mappe}: {len(filer)} klipp")
        for fil in filer:
            X.append(lag_spektrogram(fil))
            y.append(klasse)
            cond.append(klipp_info(fil)[0])
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int64),
            np.array(cond))


X_train, y_train, _ = behandle("train_cut")   # støyforhold er alltid "pure" her
X_test, y_test, cond_test = behandle("test_cut")

np.save(FEATURES / "X_train.npy", X_train)
np.save(FEATURES / "y_train.npy", y_train)
np.save(FEATURES / "X_test.npy", X_test)
np.save(FEATURES / "y_test.npy", y_test)
np.save(FEATURES / "cond_test.npy", cond_test)

print(f"\nX_train: {X_train.shape}   y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}  y_test:  {y_test.shape}")
print(f"Lagret i {FEATURES}")