## USN Group 8
###  Classifying Electric Engine Conditions through Acoustic Analysis

# Model 1 - [insert method name here]

*Notes for writing report:

- Renamed "train" and "test" directories to match a (roughly) 80:20 split. Probably misnamed by accident in original dataset.
- SciPy to read WAV files.
- Each dataset class had equal bitrates (44.1kHz), no. of audio samples (132 300), and bit depths, per file.
- First look at data shows a probable need for data prep: scaling/normalization. But we try without first.
- Trying One-vs-One Support Vector Machine (OvO SVM) first, why not. On the raw data, no features. Why not.
    - Raw data/no features was terrible, as expected.
    - Tried histogram with variable bin counts (50). Decent results, ~70% success, on *just* good vs. bad binary classification.
    - Implementing full OvO next...
