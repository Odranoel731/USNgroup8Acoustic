# Model 1 - [insert method name here]


## Main script in "model_notes.ipynb"
<br>

Jupyter notebook intended as sandbox to create and test models:
- Final version should be more streamlined in a Python script.
- Structure is thus:

    1.  Reading in training data
    2.  Creating input features and target arrays.
    3.  Testing individual accuracies (w/ cross validation) before putting together in OvO.
    4.  Three-fold voting system (the One vs. One)
    5.  Validating against test_cut

<br>
<br>

# Notes for writing report:

- Made mistake interpreting train/test dataset folders, luckily team corrected me.
- SciPy to read WAV files.
- Each dataset class (good, broken, heavy) had equal bitrates (44.1kHz), no. of audio samples (132 300), and bit depths, per file.
- First look at data shows a probable need for data prep: scaling/normalization, etc. But we try without first.

- Trying One-vs-One Support Vector Machine (OvO SVM) first, why not. On the raw data, no features. Why not.
    - Raw data/no features was terrible, as expected.
    - Tried histogram with variable bin counts (50). Decent results, ~70% success, on *just* good vs. broken binary classification.
    - Implementing full OvO next...
    - Swapped train/test back. Much stronger results, ~100% accuracy.
        Suspicious of overfitting; tested with 5-fold cross validation -> high mean, low variance -> strong results are real.
    - OvO finished: logic makes it slightly unreadable, not ideal for extension/maintaining.  
        - Near-perfect results, even with a low split (50/50, instead of 80/20).  

        - Validation against test_cut finished. Weak results. This suggests high bias in the model; overfitted to the training data by assuming a simple histogram is enough information to classify. 
        After more research, this makes sense: the waveform that is our input features (after put in histogram) shows the amplitudes, not frequencies. So it is easy to imagine the model being confused because it is trained to classify based on the "loudness".

- Discrete Fourier Transform complete. Decent results with taking the histogram of the magnitude of the DFT (with increase bin size from 50 to 200). Validation accuracy ~70%. PS. All validation runs against test_cut are final, before the core features are modified, to prevent overfitting with tuning parameters.

- Moving on to a ranking of strongest frequencies. Thought process: Histogram of DFT still uses the frequency *amplitudes* as the feature. Increasing binN 50->200 gave accuracy increase from 50%->70%, thus we take the limit and look at the most prominent frequencies directly? Last-sec note: bin size from 200->1000 indeed gave *validation* accuracy of 88%, lending credibility to this idea.
        
