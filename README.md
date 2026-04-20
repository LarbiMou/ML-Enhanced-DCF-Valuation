# ML-Enhanced-DCF-Valuation
An ML-enhanced Discounted Cash Flow (DCF) valuation engine utilizing XGBoost panel regression and Monte Carlo simulations for macro-adjusted intrinsic value estimation.

## Overview

The DCF Hybrid Valuation Model is a Python-based valuation framework that integrates traditional discounted cash flow (DCF) analysis with machine learning–driven growth forecasting. It combines firm-level financial data, macroeconomic indicators, and cross-sectional peer information to produce more adaptive and data-informed intrinsic value estimates.

The objective of this project is to augment, not replace, fundamental valuation by introducing a structured, repeatable approach to growth estimation.

---

## Core Features

### Hybrid Valuation Engine

* Supports **2-stage** and **3-stage** DCF models
* Projects free cash flow (FCF) using dynamically estimated growth rates
* Computes terminal value with constrained long-term assumptions
* Outputs enterprise value and intrinsic value per share

### Dynamic Cost of Capital (WACC)

* Risk-free rate sourced from FRED (10Y Treasury)
* Rolling beta estimated from historical returns vs S&P 500
* Equity risk premium derived from market-based components
* Cost of debt calculated from financial statements and tax-adjusted

### Machine Learning Growth Model

* Panel-based **XGBoost regression** for forward growth estimation
* Cross-sectional dataset built from sector peers
* Feature set includes:

  * FCF and revenue growth
  * GDP, CPI, and interest rates
  * Beta and interaction terms
  * Macro regime indicators

### Peer-Based Generalization

* Sector peers identified via ETF holdings and curated universes
* Panel structure improves robustness relative to single-firm models

### Monte Carlo Simulation

* Stochastic variation applied to growth rates and WACC
* Outputs valuation distribution (bear, base, bull cases)
* Estimates probability of undervaluation relative to market price

---

## Model Pipeline

1. **Data Ingestion**

   * Financial data: yfinance, SimFin
   * Macroeconomic data: FRED API

2. **Preprocessing**

   * FCF extraction and normalization
   * Growth rate computation with outlier control

3. **Feature Engineering**

   * Panel dataset across peers
   * Macro integration and interaction features

4. **Model Training**

   * XGBoost models for short-term, long-term, and transition growth

5. **Valuation**

   * Cash flow projection and discounting via WACC
   * Terminal value estimation
   * Intrinsic value calculation

6. **Uncertainty Analysis**

   * Monte Carlo simulation of valuation outcomes

---

## Installation

### Requirements

* Python 3.8+
* Dependencies:

```
pandas
numpy
yfinance
requests
xgboost
scikit-learn
simfin
```

### Setup

```
git clone https://github.com/yourusername/dcf-hybrid-model.git
cd dcf-hybrid-model
pip install -r requirements.txt
```

Create a `config.py` file with:

```
FRED_API_KEY = "your_fred_key"
SIMFIN_API_KEY = "your_simfin_key"
```

---

## Usage

Run the main script:

```
python DCF_Hybrid_Valuation.py
```

The program will prompt for:

* Stock ticker
* DCF structure (2-stage or 3-stage)
* Optional Monte Carlo simulation

Outputs include:

* Projected cash flows
* Terminal value
* Enterprise value
* Intrinsic value per share
* Comparison to current market price

---

## Design Notes

* Combines **fundamental valuation discipline** with **data-driven growth estimation**
* Uses peer-based panel data to reduce single-company bias
* Applies stability controls (winsorization, clipping, constrained terminal growth)

---

## Limitations

* Free cash flow is volatile and difficult to forecast
* Model performance depends on panel size and data quality
* Macro relationships are simplified
* Terminal value remains a dominant driver of valuation

This model is intended as a **decision-support tool**, not a substitute for fundamental analysis.

---

## Future Work

* Introduce ROIC-based growth constraints
* Transition primary prediction target to revenue growth
* Improve equity risk premium estimation
* Expand and refine peer dataset
* Add confidence scoring to valuation outputs

---

## Author

Larbi Moukhlis

---

## License

This project is licensed under the MIT License.

Copyright (c) 2026 Larbi Moukhlis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
