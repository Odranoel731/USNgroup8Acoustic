# Model 2 – oppsett

Slik setter du opp mappen og miljøet, så du kan bruke preprocessingen som er gjort og lage CNN-modellen selv.

## Mappestruktur

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
├── requirements.txt
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

## Det du får i `features/`

| Fil | Form | Innhold |
|---|---|---|
| `X_train.npy` | (357, 64, 259) | Spektrogrammer fra `train_cut` |
| `y_train.npy` | (357,) | Klasse: 0 = good, 1 = broken, 2 = heavy load |
| `X_test.npy` | (2021, 64, 259) | Spektrogrammer fra `test_cut` |
| `y_test.npy` | (2021,) | Klasse |
| `cond_test.npy` | (2021,) | Støyforhold per testklipp, f.eks. `"talking_1"` |

Klippene er sortert per klasse og etter klippnummer. Dette er input til CNN-modellen.
