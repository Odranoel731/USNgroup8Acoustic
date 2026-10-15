# Model 1 - [insert method name here]


### Main script in "model_notes.ipynb"


Jupyter notebook intended as sandbox to create and test models:
- Final version should be more streamlined in a Python script.
- Structure is thus:
1.  Reading in training data
2.  Creating input features and target arrays for the OvO of the SVM
3.  Testing individual accuracies (w/ cross validation) before putting together in OvO.
4.  Three-fold voting system (the One vs. One)
5.  Validating against test_cut


* Notes for writing report:

- Made mistake interpreting train/test dataset folders, luckily team corrected me.
- SciPy to read WAV files.
- Each dataset class had equal bitrates (44.1kHz), no. of audio samples (132 300), and bit depths, per file.
- First look at data shows a probable need for data prep: scaling/normalization, etc. But we try without first.

- Trying One-vs-One Support Vector Machine (OvO SVM) first, why not. On the raw data, no features. Why not.
    - Raw data/no features was terrible, as expected.
    - Tried histogram with variable bin counts (50). Decent results, ~70% success, on *just* good vs. bad binary classification.
    - Implementing full OvO next...
    - Swapped train/test back. Much stronger results, ~100% accuracy.
        Suspicious of overfitting; tested with 5-fold cross validation -> high mean, low variance -> strong results are real.
    - OvO finished: logic makes it slightly unreadable, not ideal for extension/maintaining.
        - Near-perfect results, even with a low split (50/50, instead of 80/20).
    - Validation against test_cut finished. Weak results. This suggests high bias in the model; overfitted to the training data by
        assuming a simple histogram is enough information to classify.
        
