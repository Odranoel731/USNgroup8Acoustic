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

## Det du får i `features/`

| Fil | Form | Innhold |
|---|---|---|
| `X_train.npy` | (357, 64, 259) | Spektrogrammer fra `train_cut` |
| `y_train.npy` | (357,) | Klasse: 0 = good, 1 = broken, 2 = heavy load |
| `X_test.npy` | (2021, 64, 259) | Spektrogrammer fra `test_cut` |
| `y_test.npy` | (2021,) | Klasse |
| `cond_test.npy` | (2021,) | Støyforhold per testklipp, f.eks. `"talking_1"` |

Klippene er sortert per klasse og etter klippnummer. Dette er input til CNN-modellen.

## Slik fungerer train_model.py
**Reproduserbarhet**
Seed settes for `random`, NumPy og PyTorch, og GPU-operasjonene gjøres deterministiske. 
To kjøringer gir dermed samme resultat, så endringer i modellen kan sammenlignes rettferdig.

**Innlasting**
Spektrogrammene lastes som `float32` og etikettene som `int64`, som` CrossEntropyLoss` krever. 
`cond_test` lastes også, slik at testresultatene kan deles opp etter støyforhold.

**Splitt i trening og validering**
```def temporal_split(labels, val_fraction=0.2):
    train_idx, val_idx = [], []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        train_idx.extend(idx[:-n_val])
        val_idx.extend(idx[-n_val:])
    return np.array(train_idx), np.array(val_idx)
```

For hver klasse går de siste 20 % av klippene til validering, i stedet for et tilfeldig utvalg. Klippene er 3-sekunders 
segmenter fra samme lange opptak, og nabosegmenter høres nesten like ut. Med en tilfeldig splitt ville et valideringsklipp
nesten alltid hatt naboen sin i treningssettet, og valideringstapet ville blitt urealistisk lavt. Splitten forutsetter 
at klippene er sortert etter klippnummer, slik `preprocess.py` gjør.    

**Normalisering**
```mean, std = X_train.mean(), X_train.std()
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std
```

Verdiene skaleres til gjennomsnitt 0 og standardavvik 1, noe som gjør treningen mer stabil.Gjennomsnitt og standardavvik 
beregnes bare fra treningsdelen, så ingen informasjon fra validering eller test lekker inn i modellen.

Deretter legges det til en kanal-dimensjon, `(N, 64, 259)` → `(N, 1, 64, 259)`, 
slik at hvert spektrogram behandles som et bilde med én kanal.

**Batcher**
Modellen ser 16 spektrogrammer om gangen og justerer vektene litt etter hver batch. 
Rekkefølgen stokkes hver epoke, med en fast generator så stokkingen blir lik hver gang.

**Modellen**
|Lag|Output|
|--|--|
|Input-spektrogram|	1 × 64 × 259|
|Konv 3×3, 16 filtre + BatchNorm + ReLU + MaxPool| 16 × 32 × 129|
|Konv 3×3, 32 filtre + BatchNorm + ReLU + MaxPool| 32 × 16 × 64|
|Konv 3×3, 64 filtre + BatchNorm + ReLU| 64 × 16 × 64|
|Adaptiv average pooling|	64 × 4 × 4|
|Lineær + ReLU + Dropout (0,4)|	64|
|Lineær| 3|

Konvolusjonsblokkene finner mønstre i spektrogrammet. BatchNorm stabiliserer treningen, og pooling gjør bildet mindre.
Den adaptive poolingen beholder et 4×4-rutenett, så modellen vet omtrent hvor i frekvens og tid mønstrene ligger.
Dropout slår tilfeldig av 40 % av de skjulte enhetene under trening, noe som reduserer overtilpasning.

**Tap og optimalisering**
```criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
```

`CrossEntropyLoss` måler hvor feil prediksjonene er.` AdamW `justerer vektene for å redusere feilen, med læringsrate 0,001.
`weight_decay` trekker vektene litt mot null i hvert steg, noe som holder modellen enklere og reduserer overtilpasning.

**TRrening og early stopping**
I hver epoke går modellen gjennom alle treningsbatchene og oppdaterer vektene. 
Etter hver epoke måles tapet på valideringssettet uten at modellen lærer av det.

Blir valideringstapet bedre, lagres modellen sammen med `mean` og `std`:
```torch.save(
    {"state_dict": model.state_dict(), "mean": float(mean), "std": float(std)},
    PROJECT / "engine_cnn.pth",
)
```
Normaliseringsverdiene lagres fordi nye opptak må normaliseres på nøyaktig samme måte før modellen kan klassifisere dem.
Blir valideringstapet ikke bedre på 7 epoker på rad, stopper treningen.

**Testing**
Den beste lagrede modellen lastes inn og klassifiserer alle testklippene. Resultatene skrives ut på tre måter:

- Nøyaktighet per støyforhold viser hvor robust modellen er, for eksempel om den klarer seg med fabrikkstøy `(atmo_high)`.
- Klassifiseringsrapporten viser precision, recall og F1-score per klasse.
- Confusion matrix viser hvilke klasser som blandes med hverandre.

**Begrensninger**
Hver tilstand er tatt opp på sin egen motor (`engine1_good`, `engine2_broken`, `engine3_heavyload`). 
Motor og tilstand er dermed det samme, og modellen kan lære å kjenne igjen den enkelte motoren i stedet for selve feilen. 
Høy testnøyaktighet beviser derfor ikke at modellen vil oppdage en feil på en motor den aldri har hørt.
Treningsdataene inneholder bare `pure`-opptak, mens testsettet har støy. 
Modellen har aldri hørt støy under trening, og det er trolig derfor nøyaktigheten faller med sterk fabrikkstøy. 
Å blande inn støy under treningen er den mest sannsynlige forbedringen.
Treningssettet er lite, 357 klipp, hvorav rundt 70 går til validering. Valideringstapet kan derfor variere en del fra epoke til epoke. 





