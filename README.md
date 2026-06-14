# Goud Portfolio Tracking & Voorspelling

**Eindproef Data Scientist**

## Doel

Het analyseren en voorspellen van goudprijzen om te bepalen of een strategie van
het kopen van 1 goudstaaf (10g) elke 2 maanden zal leiden tot EUR 100.000 aan
24k goud binnen 3 jaar (voor een aanbetaling op een huis).

## Strategie

- **Aankoop**: 1x 10g goudstaaf elke 2 maanden
- **Doel**: EUR 100.000 in 24k goud
- **Horizon**: 3 jaar (start oktober 2025)

## Projectstructuur

```
Goldportfolio/
├── PORTFOLIO.csv                  # Portfolio data (uitgesloten via .gitignore)
├── requirements.txt               # Python dependencies
├── README.md                      # Dit bestand
│
├── data/
│   ├── raw/                       # Rauwe brondata
│   ├── processed/                 # Schone, geanalyseerde data
│   ├── sample/                    # Voorbeeld data voor testen
│   └── models/                    # Opgeslagen ML modellen
│
├── notebooks/
│   ├── 01_data_collection.ipynb   # Data ophalen (Yahoo Finance)
│   ├── 02_data_profiling_cleaning.ipynb  # Profiling & cleaning
│   ├── 03_eda.ipynb               # Exploratory Data Analysis
│   ├── 04_price_prediction.ipynb  # Modellen (LR + ARIMA)
│   ├── 05_scenario_analysis.ipynb # 3 scenario projecties
│   └── 06_dashboard_walkthrough.ipynb  # Dashboard demo
│
├── src/
│   ├── data_loader.py             # Data ophalen & cachen
│   ├── profiling.py               # Data profiling functies
│   ├── predictor.py               # Model training & evaluatie
│   └── utils.py                   # Gedeelde hulpfuncties
│
└── dashboard/
    ├── app.py                     # Dash applicatie
    └── assets/
        └── style.css              # Dashboard styling
```

## Snelle Start

### 1. Virtual Environment aanmaken

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Notebooks uitvoeren

```bash
jupyter notebook notebooks/
```

Voer de notebooks uit in volgorde (01 t/m 06).

> **Let op:** Sommige notebooks verwachten een `PORTFOLIO.csv` met persoonlijke aankoopgegevens.
> Voor testdoeleinden kun je `data/sample/PORTFOLIO_SAMPLE.csv` gebruiken.
> Kopieer dat bestand naar de root en hernoem het naar `PORTFOLIO.csv`.

### 3. Dashboard starten

```bash
cd dashboard
python app.py
```

Open http://127.0.0.1:8050 in je browser.

## Modellen

### Model A: Linear Regression (Baseline)
- Interpreteerbaar: coëfficiënten laten de impact van elke feature zien
- Goed als referentiepunt voor complexere modellen

### Model B: ARIMA (AutoRegressive Integrated Moving Average)
- Ontworpen voor tijdreeks data
- Vangt autocorrelatie en patronen in het verleden
- Automatische parameterkeuze via AIC (Akaike Information Criterion)

### Baseline: Naive (Random Walk)
- Simpelste voorspelling: morgen = vandaag
- Elk serieus model moet dit verslaan

## Scenario Analyse

### Set A: Historisch (2020-2026)

| Scenario | Jaarlijks Rendement | Bron |
|----------|-------------------|------|
| Bullish  | ~+18% | 6-jaar historisch CAGR (2020-2026) |
| Neutraal | ~+7%  | Mediaan jaarlijks rendement 2020-2026 |
| Bearish  | ~+5%  | 25e percentiel jaarlijks rendement 2020-2026 |

### Set B: Klimaat & Geopolitiek (huidig macro-klimaat)

| Scenario | Jaarlijks Rendement | Bron |
|----------|-------------------|------|
| Bullish  | +35% | 2024-2025 tempo houdt aan (geopolitieke escalatie) |
| Neutraal | +18% | 6-jaar CAGR = huidige realiteit |
| Bearish  | +10% | Goldman Sachs base case |

## Data Bronnen

| Bron | Variabele | Waarom |
|------|-----------|--------|
| Yahoo Finance (GC=F) | Goud spot prijs (USD/oz) | Directe goudmarktprijs |
| Yahoo Finance (EURUSD=X) | EUR/USD wisselkoers | Conversie naar EUR |
| Yahoo Finance (GLD) | Goud ETF | Marktsentiment indicator |
| World Gold Council | Historische rendementen & marktcontext | Scenario-parameters en onafhankelijke verificatie |

## Beperkingen

- Financiele tijdreeksen zijn moeilijk te voorspellen (lage signaal/ruis ratio)
- ARIMA kan slechter presteren dan een naive op financiele data
- Modellen zijn **directionele indicatoren**
- Scenario-analyse completeert de voorspelling

## Disclaimer

Dit is een educatief project voor een eindproef.
Het vormt geen financieel advies. Beleggingen in goud brengen risico's met zich mee.
