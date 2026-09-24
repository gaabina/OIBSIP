# OIBSIP Data Science — Task 2: Unemployment Analysis with Python

This project completes the Oasis Infobyte Data Science Task 2 requirements using the publicly available **Unemployment in India** dataset. It examines regional and temporal unemployment patterns and compares labour-market indicators before and after March 2020.

## Files

- `Unemployment_in_India.csv` — source dataset.
- `Unemployment_Analysis_Task2.ipynb` — clean, commented submission notebook.
- `unemployment_analysis.py` — equivalent standalone Python script.
- `outputs/` — generated charts and summary tables.

## Run locally

```bash
pip install -r requirements.txt
jupyter notebook Unemployment_Analysis_Task2.ipynb
```

Or run the script:

```bash
python unemployment_analysis.py
```

## Requirements covered

The notebook includes loading, shape inspection, null checks, type conversion, cleaning, region-wise averages, month-wise trends, a five-region time series, top-10 bar chart, a three-indicator correlation heatmap, a Pre-COVID/Post-COVID mean comparison, written observations, limitations, and reproducibility notes.

## Data source and limitation

Source: [Kaggle — Unemployment in India](https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india), whose description attributes the underlying source to CMIE. The dataset covers May 2019–June 2020 and is therefore a short descriptive snapshot, not a long-term estimate of unemployment. The source does not include a standalone employment-rate column; `Employment_Rate` is derived as `Labour_Participation_Rate × (1 − Unemployment_Rate/100)` for the required three-indicator comparison.
