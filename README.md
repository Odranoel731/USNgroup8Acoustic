# USN Group 8
##  Classifying Electric Engine Conditions through Acoustic Analysis

Codebase for our project for the AI & ML course at the University of South-Eastern Norway (USN).
The problem statement is thus:
Create 2 models to classify the dataset "IDMT-ISA-ELECTRIC-ENGINE" into 3 classes; either *good*, *broken*, or *heavy load*.

Each model has its own branch.
**To run, download the ZIP of each branch, and follow the build/run instructions.**

-------------------------------------

# SETTING UP MODEL 1

Open a Jupyter Notebook at the root of the project.
Place the (extracted) dataset one level **above** the root project folder, as shown below:
Run the notebook, all cells, with the required imports at the top.

### Mappestruktur

```
Model_1/
├── IDMT-ISA-ELECTRIC-ENGINE/
│     ├── train/  train_cut/
│     └── test/   test_cut/
    ├── model_notes.ipynb
    ├── README.md
```

# SETTING UP MODEL 2

### Mappestruktur

```
Model_2/
├── Data_sett/                      ← datasettet legges her (ikke i git)
│   └── IDMT-ISA-ELECTRIC-ENGINE/
│       ├── train/  train_cut/
│       └── test/   test_cut/
├── features/                       ← lages av preprocess.py (ikke i git)
├── oversikt.py                     ← sjekker at datasettet ligger riktig
├── utforsk.py                      ← plotter eksempler fra hver klasse
├── preprocess.py                   ← lager spektrogrammene
├── train_model.py                    ← trener og tester CNN-modellen  
├── requirements.txt
├── engine_cnn.pth                  ← beste modell, lages av final_cnn.py (ikke i git)
└── README.md
```

## Oppsett

**1. Hent koden** og åpne `Model_2`-mappen i VS Code.

**2. Installer pakkene** (Python 3.10 eller nyere):
```
cd Model_2
pip install -r requirements.txt
```

**3. Last ned datasettet** fra https://zenodo.org/records/7551261 og pakk det ut slik at stien blir:
```
Model_2/Data_sett/IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine1_good/pure_0.wav
```

**4. Sjekk at datasettet ligger riktig:**
```
python oversikt.py
```
Det skal være 357 filer i `train_cut` og 2 021 i `test_cut`.

**5. Lag spektrogrammene** (én gang, tar et par minutter):
```
python preprocess.py
```

**6. Tren modellen**
Kjør ```train_model.py```

Den skriver ut tapet for hver epoke mens den kjører. Etter treningen skriver det ut nøyaktighet per støyforhold, 
en klassifiseringsrapport og en confusion matrix. Den beste modellen lagres i engine_cnn.pth.
