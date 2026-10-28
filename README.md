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

## Important data note
The default dataset is synthetically generated for demonstration. Predictions and metrics are not validated against real electricity bills. For a real project, replace `make_demo_data()` in `app.py` with a real dataset containing historical meter readings and bill amounts, then retrain and test the models. Adjust the tariff calculation to the relevant electricity provider's current tariff.
