import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="WattWise | Electricity Predictor", page_icon="⚡", layout="wide")

# ----------------------------- Theme / helpers -----------------------------
st.markdown("""
<style>
.stApp { background-color: #f7f8fc; }
[data-testid="stSidebar"] { background-color: #eef2ff; }
h1, h2, h3 { color: #172554; }
div[data-testid="stMetric"] { background: white; border: 1px solid #e0e7ff; padding: 14px; border-radius: 12px; }
div.stButton > button { border-radius: 9px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

SEASON_MONTHS = {
    "Winter": [12, 1, 2],
    "Summer": [3, 4, 5],
    "Monsoon": [6, 7, 8, 9],
    "Post-monsoon": [10, 11],
}
MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

def season_from_month(month):
    for season, months in SEASON_MONTHS.items():
        if int(month) in months:
            return season
    return "Summer"

DATA_PATH = Path(__file__).parent / "electricity_bill_dataset.csv"
APPLIANCES = ["Fan", "Refrigerator", "AirConditioner", "Television", "Monitor"]
CAT_FEATURES = ["City", "Company"]
FEATURES = APPLIANCES + ["Month"] + CAT_FEATURES
REQUIRED_COLS = ["Fan", "Refrigerator", "AirConditioner", "Television", "Monitor", "MotorPump",
                 "Month", "City", "Company", "MonthlyHours", "TariffRate", "ElectricityBill"]

def prepare(df):
    df = df.drop_duplicates().dropna().copy()
    df["Month"] = df["Month"].astype(int)
    df["Season"] = df["Month"].apply(season_from_month)
    df["HighUsage"] = (df["MonthlyHours"] >= df["MonthlyHours"].quantile(0.75)).astype(int)
    return df.reset_index(drop=True)

@st.cache_data
def load_default():
    return prepare(pd.read_csv(DATA_PATH))

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), APPLIANCES + ["Month"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])

def make_pipe(model, scale=True):
    # Trees do not need scaling, but sharing one preprocessor keeps the code simple.
    return Pipeline([("prep", preprocessor()), ("model", model)])

@st.cache_resource(show_spinner="Training regression models...")
def train_regressors(data):
    X, y = data[FEATURES], data["ElectricityBill"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Linear Regression": make_pipe(LinearRegression()),
        "Ridge Regression (L2)": make_pipe(Ridge(alpha=1.0)),
        "Lasso Regression (L1)": make_pipe(Lasso(alpha=0.5, max_iter=10000)),
        "Decision Tree Regressor": make_pipe(DecisionTreeRegressor(max_depth=8, random_state=42)),
        "Random Forest Regressor": make_pipe(RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)),
    }
    results, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({"Model": name, "MAE": mean_absolute_error(y_test, pred),
                        "RMSE": rmse(y_test, pred), "R²": r2_score(y_test, pred)})
        fitted[name] = model
    return fitted, pd.DataFrame(results).sort_values("RMSE")

@st.cache_resource(show_spinner="Training classifiers...")
def train_classifiers(data):
    X, y = data[FEATURES], data["HighUsage"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "Logistic Regression": make_pipe(LogisticRegression(max_iter=1500)),
        "Decision Tree Classifier": make_pipe(DecisionTreeClassifier(max_depth=6, random_state=42)),
        "Random Forest Classifier": make_pipe(RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
    }
    results, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
                        "Precision": precision_score(y_test, pred, zero_division=0),
                        "Recall": recall_score(y_test, pred, zero_division=0),
                        "F1-score": f1_score(y_test, pred, zero_division=0)})
        fitted[name] = model
    return X_test, y_test, fitted, pd.DataFrame(results).sort_values("F1-score", ascending=False)

@st.cache_resource(show_spinner="Training usage-hours model...")
def train_hours_model(data):
    m = make_pipe(RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
    m.fit(data[FEATURES], data["MonthlyHours"])
    return m

# ----------------------------- Load data -----------------------------
st.sidebar.title("⚡ WattWise")
st.sidebar.caption("Electricity Bill Prediction & Seasonal Insights")

uploaded = st.sidebar.file_uploader("Use a different CSV (optional)", type="csv")
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        st.sidebar.error(f"Uploaded CSV is missing columns: {', '.join(missing)}. Using the default dataset.")
        data = load_default()
    else:
        data = prepare(raw)
else:
    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH.name} must be in the same folder as app.py.")
        st.stop()
    data = load_default()

reg_models, reg_metrics = train_regressors(data)
clf_X_test, clf_y_test, clf_models, clf_metrics = train_classifiers(data)
hours_model = train_hours_model(data)

tariff_lookup = data.groupby(["City", "Company"])["TariffRate"].first()
CITIES = sorted(data["City"].unique())
COMPANIES = sorted(data["Company"].unique())
NUM_COLS = [c for c in data.select_dtypes(include=np.number).columns
            if data[c].nunique() > 1 and c != "HighUsage"]

page = st.sidebar.radio("Navigate", [
    "🏠 Overview", "🔮 Bill Prediction", "🌦️ Seasonal Analysis",
    "📊 Data Exploration", "🤖 Model Lab", "📈 Statistics & Tests",
    "🧩 Usage Segments", "💡 Saving Tips"
])
st.sidebar.info(f"Dataset: {len(data):,} records, {data['City'].nunique()} cities, {data['Company'].nunique()} companies.")

# ----------------------------- Overview -----------------------------
if page == "🏠 Overview":
    st.title("⚡ WattWise")
    st.subheader("Electricity Bill Prediction and Seasonal Consumption Analytics")
    st.write("Estimate a monthly electricity bill, compare seasonal usage, and explore ML practicals in one website.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(data):,}")
    c2.metric("Average bill", f"₹{data.ElectricityBill.mean():,.0f}")
    c3.metric("Average usage", f"{data.MonthlyHours.mean():.0f} hrs/month")
    c4.metric("Cities", data.City.nunique())
    st.info("In this dataset, ElectricityBill = MonthlyHours × TariffRate exactly, and the tariff is fixed for each City + Company pair. "
            "Because of that, the models predict the bill from appliances, month, city and company only. Including hours and tariff as inputs "
            "would make the score near-perfect (R² ≈ 1.0) without any real learning.")
    left, right = st.columns(2)
    with left:
        st.markdown("### What this website does")
        st.markdown("- Predicts estimated monthly bill and usage hours\n- Compares winter, summer, monsoon, and post-monsoon\n- Compares regression and classification models\n- Shows correlation, statistical tests, and usage segments")
    with right:
        st.markdown("### Practical concepts included")
        st.markdown("- Pandas, NumPy, missing-value checks, EDA\n- Linear, Ridge, Lasso, Decision Tree, Random Forest\n- Logistic Regression and classification metrics\n- K-Means, Gaussian Mixture, correlation, t-test, ANOVA, chi-square")

# ----------------------------- Prediction -----------------------------
elif page == "🔮 Bill Prediction":
    st.title("🔮 Electricity Bill Predictor")
    st.write("Enter appliance counts, location and month to estimate the monthly bill.")
    def rng_of(col):
        return int(data[col].min()), int(data[col].max()), int(data[col].median())
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            lo, hi, med = rng_of("Fan");            fan = st.slider("Fans", lo, hi, med)
            lo, hi, med = rng_of("Refrigerator");   fridge = st.slider("Refrigerators", lo, hi, med)
            lo, hi, med = rng_of("AirConditioner"); ac = st.slider("Air conditioners", lo, hi, med)
        with c2:
            lo, hi, med = rng_of("Television");     tv = st.slider("Televisions", lo, hi, med)
            lo, hi, med = rng_of("Monitor");        mon = st.slider("Monitors", lo, hi, med)
            month_name = st.selectbox("Month", MONTH_NAMES, index=5)
        with c3:
            city = st.selectbox("City", CITIES, index=CITIES.index("Pune") if "Pune" in CITIES else 0)
            company = st.selectbox("Company", COMPANIES)
            model_name = st.selectbox("Prediction model", list(reg_models.keys()),
                                      index=list(reg_models.keys()).index("Random Forest Regressor"))
        submitted = st.form_submit_button("Predict electricity bill", type="primary", width="stretch")
    if submitted:
        month_num = MONTH_NAMES.index(month_name) + 1
        season = season_from_month(month_num)
        row = pd.DataFrame([{"Fan": fan, "Refrigerator": fridge, "AirConditioner": ac, "Television": tv,
                             "Monitor": mon, "Month": month_num, "City": city, "Company": company}])
        est_bill = max(0.0, float(reg_models[model_name].predict(row[FEATURES])[0]))
        est_hours = max(0.0, float(hours_model.predict(row[FEATURES])[0]))
        tariff = tariff_lookup.get((city, company), np.nan)
        st.success(f"Selected month: {month_name} | Season: {season} | {city} | {company}")
        a, b, c, d = st.columns(4)
        a.metric("Estimated monthly bill", f"₹{est_bill:,.0f}")
        b.metric("Estimated usage", f"{est_hours:,.0f} hrs")
        c.metric("Tariff rate", f"₹{tariff:.2f}/hr" if pd.notna(tariff) else "n/a")
        d.metric("Model used", model_name)
        if pd.notna(tariff):
            st.caption(f"Cross-check: estimated hours × tariff = ₹{est_hours * tariff:,.0f}.")
        st.markdown("### What to do next")
        tips = []
        if ac >= 2:
            tips.append("Air conditioners add a lot to usage; clean filters and use a moderate temperature setting.")
        if fan >= data.Fan.quantile(0.75):
            tips.append("You have many fans for this dataset; switch them off in empty rooms.")
        if tv + mon >= (data.Television + data.Monitor).quantile(0.75):
            tips.append("Screens add up; use sleep timers and power-saving modes.")
        if est_bill > data.ElectricityBill.quantile(0.75):
            tips.append("Your estimated bill is in the top 25% of this dataset; check high-power appliances and compare with real meter readings.")
        if not tips:
            tips.append("Record your meter reading monthly and compare usage across seasons.")
        for tip in tips:
            st.write("• " + tip)
        st.caption("Estimates come from patterns in the uploaded dataset, not from an official utility tariff calculation.")

# ----------------------------- Seasonal Analysis -----------------------------
elif page == "🌦️ Seasonal Analysis":
    st.title("🌦️ Season-wise Electricity Analysis")
    order = ["Winter", "Summer", "Monsoon", "Post-monsoon"]
    seasonal = data.groupby("Season", as_index=False).agg(
        average_hours=("MonthlyHours", "mean"),
        average_bill=("ElectricityBill", "mean"),
        median_bill=("ElectricityBill", "median"),
        average_ac=("AirConditioner", "mean"),
        records=("ElectricityBill", "size"),
    )
    seasonal["Season"] = pd.Categorical(seasonal["Season"], categories=order, ordered=True)
    seasonal = seasonal.sort_values("Season")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        ax.bar(seasonal["Season"].astype(str), seasonal["average_hours"])
        ax.set_title("Average Monthly Usage by Season")
        ax.set_ylabel("Average hours")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        ax.bar(seasonal["Season"].astype(str), seasonal["average_bill"])
        ax.set_title("Average Bill by Season")
        ax.set_ylabel("Average bill (₹)")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)
    st.dataframe(seasonal, width="stretch", hide_index=True)
    st.markdown("### Month-wise average bill")
    monthly = data.groupby("Month")["ElectricityBill"].mean().reindex(range(1, 13))
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(MONTH_NAMES, monthly.values, marker="o")
    ax.set_ylabel("Average bill (₹)")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)
    selected_season = st.selectbox("Inspect a season", order)
    subset = data[data["Season"] == selected_season]
    a, b, c = st.columns(3)
    a.metric("Average bill", f"₹{subset.ElectricityBill.mean():,.0f}")
    b.metric("Average usage", f"{subset.MonthlyHours.mean():.0f} hrs")
    c.metric("Records", f"{len(subset):,}")
    suggestions = {
        "Winter": "Check room heaters and water heaters; use timers where possible.",
        "Summer": "AC and cooling may raise use; clean filters and set a moderate temperature.",
        "Monsoon": "Use daylight when possible and switch off unused lights and appliances.",
        "Post-monsoon": "Compare bills against summer and check whether cooling use can be reduced.",
    }
    st.info(suggestions[selected_season])
    st.caption("Seasons are mapped from the Month column. The dataset shows only small month-to-month differences "
               "(February is lower, partly because it has fewer days), so seasonal effects here are weak compared with appliance and city effects.")

# ----------------------------- Data Exploration -----------------------------
elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration (EDA)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(data):,}")
    c2.metric("Columns", len(data.columns))
    c3.metric("Missing values", int(data.isnull().sum().sum()))
    st.markdown("### Preview")
    st.dataframe(data.head(15), width="stretch")
    st.markdown("### Data quality checks")
    quality = pd.DataFrame({"Data type": data.dtypes.astype(str), "Missing values": data.isnull().sum(),
                            "Unique values": data.nunique()})
    st.dataframe(quality, width="stretch")
    st.caption(f"Duplicate rows removed on load. MotorPump is 0 for every record, so it is excluded from models and correlations."
               if data["MotorPump"].nunique() == 1 else "Duplicate rows removed on load.")
    st.markdown("### Descriptive statistics")
    st.dataframe(data.describe(include="all").T, width="stretch")
    col = st.selectbox("Choose a numeric column for distribution", NUM_COLS, index=NUM_COLS.index("ElectricityBill"))
    fig, ax = plt.subplots()
    sns.histplot(data[col], kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")
    st.pyplot(fig)
    st.markdown("### Bill by city")
    city_avg = data.groupby("City")["ElectricityBill"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(city_avg.index, city_avg.values)
    ax.set_xlabel("Average bill (₹)")
    st.pyplot(fig)
    xcol = st.selectbox("Scatter plot X-axis", ["MonthlyHours", "Fan", "Refrigerator", "AirConditioner", "Television", "Monitor", "TariffRate"])
    ycol = st.selectbox("Scatter plot Y-axis", ["ElectricityBill", "MonthlyHours"])
    sample = data.sample(min(3000, len(data)), random_state=1)
    fig, ax = plt.subplots()
    sns.scatterplot(data=sample, x=xcol, y=ycol, hue="Season", alpha=0.6, ax=ax)
    ax.set_title(f"{ycol} vs {xcol} (random sample of {len(sample):,})")
    st.pyplot(fig)
    st.download_button("Download dataset CSV", data.to_csv(index=False).encode("utf-8"), "electricity_bill_dataset_clean.csv", "text/csv")

# ----------------------------- Model Lab -----------------------------
elif page == "🤖 Model Lab":
    st.title("🤖 Model Comparison Lab")
    st.markdown("### Regression: predict electricity bill")
    st.caption("Inputs: Fan, Refrigerator, AirConditioner, Television, Monitor, Month, City, Company (80/20 train-test split).")
    st.dataframe(reg_metrics.round(3), width="stretch", hide_index=True)
    best_reg = reg_metrics.iloc[0]
    st.info(f"Lowest test RMSE: {best_reg['Model']} (RMSE = ₹{best_reg['RMSE']:.0f}, R² = {best_reg['R²']:.3f}).")
    st.markdown("### Classification: identify high-usage records")
    st.dataframe(clf_metrics.round(3), width="stretch", hide_index=True)
    chosen_clf = st.selectbox("Choose classifier for confusion matrix", list(clf_models.keys()))
    clf_pred = clf_models[chosen_clf].predict(clf_X_test)
    cm = confusion_matrix(clf_y_test, clf_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "High"], yticklabels=["Normal", "High"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {chosen_clf}")
    st.pyplot(fig)
    st.code(classification_report(clf_y_test, clf_pred, target_names=["Normal usage", "High usage"], zero_division=0))
    st.caption("High usage = top 25% of MonthlyHours in the dataset.")

# ----------------------------- Statistics & Tests -----------------------------
elif page == "📈 Statistics & Tests":
    st.title("📈 Statistics, Correlation & Hypothesis Testing")
    numeric = data[NUM_COLS]
    st.markdown("### Correlation heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)
    st.markdown("### Pearson correlation")
    c1, c2 = st.columns(2)
    with c1:
        x_name = st.selectbox("Variable X", NUM_COLS, index=NUM_COLS.index("AirConditioner"))
    with c2:
        y_name = st.selectbox("Variable Y", NUM_COLS, index=NUM_COLS.index("ElectricityBill"))
    r, p = stats.pearsonr(data[x_name], data[y_name])
    st.write(f"Pearson correlation r = **{r:.3f}**, p-value = **{p:.4g}**.")
    st.caption("Correlation describes linear association, not causation.")
    st.markdown("### One-way ANOVA: does average bill differ by season?")
    groups = [data.loc[data.Season == s, "ElectricityBill"].values for s in SEASON_MONTHS.keys()]
    f_stat, anova_p = stats.f_oneway(*groups)
    st.write(f"F-statistic = **{f_stat:.3f}**, p-value = **{anova_p:.4g}**.")
    st.write("At alpha = 0.05: " + ("Reject the null hypothesis; at least one seasonal mean differs." if anova_p < 0.05 else "Fail to reject the null hypothesis; evidence of a seasonal mean difference is insufficient."))
    st.markdown("### Independent t-test: summer vs winter bills")
    summer = data.loc[data.Season == "Summer", "ElectricityBill"]
    winter = data.loc[data.Season == "Winter", "ElectricityBill"]
    t_stat, t_p = stats.ttest_ind(summer, winter, equal_var=False)
    st.write(f"t-statistic = **{t_stat:.3f}**, p-value = **{t_p:.4g}**. Mean summer = ₹{summer.mean():,.0f}, mean winter = ₹{winter.mean():,.0f}.")
    st.markdown("### Chi-square test: season vs high usage")
    contingency = pd.crosstab(data["Season"], data["HighUsage"])
    chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
    st.write(f"Chi-square = **{chi2:.3f}**, degrees of freedom = **{dof}**, p-value = **{chi_p:.4g}**.")
    st.dataframe(contingency, width="stretch")
    st.caption("With tens of thousands of records, even tiny differences can be statistically significant. "
               "Compare the actual size of the differences (means, r values) before calling an effect meaningful.")

# ----------------------------- Clustering -----------------------------
elif page == "🧩 Usage Segments":
    st.title("🧩 Household Usage Segments")
    st.write("K-Means and Gaussian Mixture Model (GMM) group records with similar usage patterns. This is clustering, not a prediction of a guaranteed customer type.")
    cluster_features = APPLIANCES + ["MonthlyHours", "ElectricityBill"]
    X_scaled = StandardScaler().fit_transform(data[cluster_features])
    k = st.slider("Number of clusters", 2, 5, 3)
    algorithm = st.radio("Clustering algorithm", ["K-Means", "Gaussian Mixture Model (GMM)"], horizontal=True)
    if algorithm == "K-Means":
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
    else:
        labels = GaussianMixture(n_components=k, random_state=42).fit_predict(X_scaled)
    clustered = data.copy()
    clustered["cluster"] = labels.astype(str)
    summary = clustered.groupby("cluster")[cluster_features].mean().round(2)
    summary["records"] = clustered.groupby("cluster").size()
    st.dataframe(summary, width="stretch")
    sample = clustered.sample(min(4000, len(clustered)), random_state=1)
    fig, ax = plt.subplots()
    sns.scatterplot(data=sample.sort_values("cluster"), x="MonthlyHours", y="ElectricityBill", hue="cluster",
                    palette="tab10", alpha=0.6, ax=ax)
    ax.set_title(f"{algorithm} Segments (random sample of {len(sample):,})")
    st.pyplot(fig)
    st.caption("StandardScaler is used because clustering depends on distances or feature scales.")

# ----------------------------- Saving tips -----------------------------
elif page == "💡 Saving Tips":
    st.title("💡 Electricity Saving Suggestions")
    st.write("General suggestions based on usage patterns; these are not guaranteed savings.")
    st.markdown("""
    - **Cooling:** Clean AC filters, seal gaps, and choose a moderate temperature.
    - **Lighting:** Use LED bulbs and switch off lights in empty rooms.
    - **Appliances:** Avoid standby power and run full loads where appropriate.
    - **Heating:** Use timers for water heaters and other high-power devices.
    - **Monitoring:** Record meter readings monthly and compare the same season year over year.
    """)
    st.markdown("### Estimate savings from reducing usage hours")
    avg_hours = float(data.MonthlyHours.mean())
    c1, c2 = st.columns(2)
    with c1:
        city = st.selectbox("City", CITIES, key="tip_city")
    with c2:
        company = st.selectbox("Company", COMPANIES, key="tip_company")
    tariff = float(tariff_lookup.get((city, company), data.TariffRate.mean()))
    current = st.number_input("Current monthly usage (hours)", min_value=0.0, value=float(round(avg_hours)), step=10.0)
    reduction = st.slider("Possible usage reduction (%)", 0, 40, 10)
    new_hours = current * (1 - reduction / 100)
    saving = (current - new_hours) * tariff
    a, b = st.columns(2)
    a.metric("Usage after reduction", f"{new_hours:.0f} hrs", f"-{current - new_hours:.0f} hrs")
    b.metric("Estimated monthly saving", f"₹{saving:,.0f}", f"at ₹{tariff:.2f}/hr")
    st.caption("In this dataset bill = hours × tariff rate. Real bills may also include fixed charges, taxes, and slab rules.")
