# OIBSIP Data Science — Task 3: Car Price Prediction

This project predicts used-car selling prices using the public **Vehicle dataset from Cardekho**. It follows the OIBSIP Task 3 checklist.

## Dataset
`car data.csv` contains car name, year, selling price, present price, kilometres driven, fuel type, seller type, transmission, and owner. Source: [Cardekho vehicle dataset on GitHub](https://github.com/ShuklaPrashant21/Used-Car-Price-Prediction/blob/master/car%20data.csv).

## Included
Data cleaning, duplicate/null handling, categorical normalization, car-age and brand feature engineering, price distribution, fuel-type box plot, price-vs-age scatter plot, one-hot encoding, correlation heatmap, train/test split, Linear Regression, Random Forest Regression, MAE/RMSE/R2 evaluation, and best-model feature importance.

## Run
```bash
pip install -r requirements.txt
python car_price_prediction.py
```
Open `Car_Price_Prediction_Task3.ipynb` in VS Code/Jupyter and run all cells.
