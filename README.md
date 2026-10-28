# WattWise — Electricity Bill Predictor

## Run locally
1. Install Python 3.10 or newer.
2. Open Command Prompt in this folder.
3. Run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Included sections
- Overview dashboard
- Bill and consumption prediction
- Season-wise analysis (Winter, Summer, Monsoon, Post-monsoon)
- Data exploration and plots
- Regression model comparison (Linear, Ridge/L1, Lasso/L2, Decision Tree, Random Forest)
- Classification (Logistic Regression, Decision Tree, Random Forest) and metrics
- Correlation, ANOVA, independent t-test, chi-square test
- K-Means and Gaussian Mixture Model clustering
- Saving tips

## Dataset
The app loads `electricity_bill_dataset.csv` (keep it in the same folder as `app.py`). You can also upload a different CSV with the same columns from the sidebar.

Columns: Fan, Refrigerator, AirConditioner, Television, Monitor, MotorPump, Month, City, Company, MonthlyHours, TariffRate, ElectricityBill.

## Important data notes
- In this dataset `ElectricityBill = MonthlyHours x TariffRate` exactly, and the tariff is fixed per City + Company pair. Models therefore predict the bill from appliances, month, city and company only; using hours and tariff as inputs would give a near-perfect but meaningless score.
- `MotorPump` is 0 in every row, so it is excluded from models and correlations.
- Month has only a small effect on usage in this data, so seasonal differences are weak.
