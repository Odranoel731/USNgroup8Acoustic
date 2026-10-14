## USN Group 8
####  Classifying Electric Engine Conditions through Acoustic Analysis

# Model 1 - [insert method name here]

### Script found in Jupyter Notebook file

*Notes for writing report:

- Made mistake interpreting train/test dataset folders, luckily team corrected me.
- SciPy to read WAV files.
- Each dataset class had equal bitrates (44.1kHz), no. of audio samples (132 300), and bit depths, per file.
- First look at data shows a probable need for data prep: scaling/normalization. But we try without first.
- Trying One-vs-One Support Vector Machine (OvO SVM) first, why not. On the raw data, no features. Why not.
    - Raw data/no features was terrible, as expected.
    - Tried histogram with variable bin counts (50). Decent results, ~70% success, on *just* good vs. bad binary classification.
    - Implementing full OvO next...
    - Swapped train/test back. Much stronger results, ~100% accuracy.
        Suspicious of overfitting; tested with 5-fold cross validation -> high mean, low variance -> strong results are real.
    - OvO almost done (not usable atm)
