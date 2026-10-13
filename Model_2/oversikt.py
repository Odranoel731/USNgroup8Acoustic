"""
Oversiktsscript for IDMT-ISA-ELECTRIC-ENGINE.
Leser bare filhodet til hver WAV-fil (ikke selve lyden), så det går raskt.
Kan slettes når vi har fått oversikten.
"""
import os
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf  # installert sammen med librosa

# Relativ sti: Data_sett-mappen ligger ved siden av dette scriptet
DATA_DIR = Path(__file__).parent / "Data_sett"

totalt_per_mappe = {}

for root, dirs, files in os.walk(DATA_DIR):
    wav_files = [f for f in files if f.lower().endswith(".wav")]
    if not wav_files:
        continue

    durations = []
    samplerates, channels, formats = set(), set(), set()
    conditions = Counter()

    for f in wav_files:
        info = sf.info(os.path.join(root, f))
        durations.append(info.duration)
        samplerates.add(info.samplerate)
        channels.add(info.channels)
        formats.add(info.subtype)
        # I _cut-mappene: "atmo_high_12.wav" -> "atmo_high"
        # I de ukuttede mappene: "atmo_high.wav" -> "atmo_high"
        navn = f[:-4]
        conditions[navn.rsplit("_", 1)[0] if "_cut" in root else navn] += 1

    durations = np.array(durations)
    rel = os.path.relpath(root, DATA_DIR)
    totalt_per_mappe[rel] = len(wav_files)

    print(f"\n=== {rel} ===")
    print(f"Antall filer:   {len(wav_files)}")
    print(f"Samplingsrate:  {samplerates}  |  Kanaler: {channels}  |  Format: {formats}")
    print(f"Lengde (s):     min {durations.min():.2f}, maks {durations.max():.2f}, "
          f"snitt {durations.mean():.2f}, totalt {durations.sum() / 60:.1f} min")
    print(f"Støyforhold:    {dict(conditions)}")

if not totalt_per_mappe:
    print(f"Fant ingen WAV-filer i {DATA_DIR}")
    print("Har du lagt datasettet (IDMT-ISA-ELECTRIC-ENGINE) inn i Data_sett-mappen?")
else:
    print("\n=== OPPSUMMERING ===")
    for mappe, antall in totalt_per_mappe.items():
        print(f"{mappe:60s} {antall:5d} filer")
    print(f"{'TOTALT':60s} {sum(totalt_per_mappe.values()):5d} filer")