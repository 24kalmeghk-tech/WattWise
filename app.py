import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

def make_demo_data(n=1800, seed=42):
    """Synthetic demonstration data only; replace with real meter records for real-world accuracy."""
    rng = np.random.default_rng(seed)
    month = rng.integers(1, 13, n)
    season = np.array([season_from_month(m) for m in month])
    household_size = rng.integers(1, 8, n)
    appliance_hours = rng.uniform(2, 12, n)
    ac_hours = np.where(np.isin(month, [3, 4, 5, 10, 11]), rng.uniform(0, 9, n),
                        np.where(np.isin(month, [12, 1, 2]), rng.uniform(0, 3, n), rng.uniform(0, 5, n)))
    heater_hours = np.where(np.isin(month, [12, 1, 2]), rng.uniform(0, 5, n), rng.uniform(0, 1, n))
    temperature = np.select(
        [np.isin(month, [12, 1, 2]), np.isin(month, [3, 4, 5]), np.isin(month, [6, 7, 8, 9])],
        [rng.normal(20, 4, n), rng.normal(34, 4, n), rng.normal(28, 3, n)],
        default=rng.normal(29, 4, n)
    )
    previous_units = rng.uniform(60, 500, n)
    weekend = rng.integers(0, 2, n)
    # Simple illustrative relationship, not a real electricity tariff model.
    units = (35 + household_size * 19 + appliance_hours * 5.2 + ac_hours * 8.5 +
             heater_hours * 5.5 + np.maximum(temperature - 28, 0) * 2.0 +
             previous_units * 0.17 + weekend * 4 + rng.normal(0, 22, n))
    units = np.clip(units, 20, None)
    # Illustrative slab tariff. Adjust these rates to the local electricity provider.
    bill = 80 + np.minimum(units, 100) * 4.0 + np.minimum(np.maximum(units - 100, 0), 200) * 6.0 + np.maximum(units - 300, 0) * 8.0
    bill = bill + rng.normal(0, 65, n)
    bill = np.clip(bill, 100, None)
    data = pd.DataFrame({
        "month": month, "season": season, "household_size": household_size,
        "appliance_hours": appliance_hours.round(1), "ac_hours": ac_hours.round(1),
        "heater_hours": heater_hours.round(1), "temperature": temperature.round(1),
        "previous_units": previous_units.round(1), "weekend": weekend,
        "units_consumed": units.round(1), "bill_amount": bill.round(0)
    })
    data["high_usage"] = (data["units_consumed"] >= data["units_consumed"].quantile(0.75)).astype(int)
    return data

@st.cache_data
def get_data():
    return make_demo_data()

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def train_regressors(data):
    features = ["household_size", "appliance_hours", "ac_hours", "heater_hours",
                "temperature", "previous_units", "weekend", "month"]
    X = data[features]
    y = data["bill_amount"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Linear Regression": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "Ridge Regression (L2)": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Lasso Regression (L1)": Pipeline([("scale", StandardScaler()), ("model", Lasso(alpha=0.05, max_iter=10000))]),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1),
    }
    results, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({"Model": name, "MAE": mean_absolute_error(y_test, pred),
                        "RMSE": rmse(y_test, pred), "R²": r2_score(y_test, pred)})
        fitted[name] = model
    metrics = pd.DataFrame(results).sort_values("RMSE")
    return features, X_train, X_test, y_train, y_test, fitted, metrics

def train_classifiers(data):
    features = ["household_size", "appliance_hours", "ac_hours", "heater_hours",
                "temperature", "previous_units", "weekend", "month"]
    X = data[features]
    y = data["high_usage"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "Logistic Regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1500))]),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
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
    return features, X_train, X_test, y_train, y_test, fitted, pd.DataFrame(results).sort_values("F1-score", ascending=False)

data = get_data()
features, X_train, X_test, y_train, y_test, reg_models, reg_metrics = train_regressors(data)
clf_features, Xc_train, Xc_test, yc_train, yc_test, clf_models, clf_metrics = train_classifiers(data)

# ----------------------------- Sidebar navigation -----------------------------
st.sidebar.title("⚡ WattWise")
st.sidebar.caption("Electricity Bill Prediction & Seasonal Insights")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview", "🔮 Bill Prediction", "🌦️ Seasonal Analysis",
    "📊 Data Exploration", "🤖 Model Lab", "📈 Statistics & Tests",
    "🧩 Usage Segments", "💡 Saving Tips"
])
st.sidebar.info("Demo uses synthetic data. Replace it with real meter records before relying on predictions.")

# ----------------------------- Overview -----------------------------
if page == "🏠 Overview":
    st.title("⚡ WattWise")
    st.subheader("Electricity Bill Prediction and Seasonal Consumption Analytics")
    st.write("Estimate a household's electricity bill, compare seasonal usage, and explore ML practicals in one website.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Demo records", f"{len(data):,}")
    c2.metric("Average bill", f"₹{data.bill_amount.mean():,.0f}")
    c3.metric("Average units", f"{data.units_consumed.mean():.0f} kWh")
    c4.metric("Seasons tracked", "4")
    st.warning("Important: the included data is synthetically generated for a college demonstration. The displayed model scores do not prove real-world accuracy.")
    left, right = st.columns(2)
    with left:
        st.markdown("### What this website does")
        st.markdown("- Predicts estimated monthly bill and units\n- Compares winter, summer, monsoon, and post-monsoon\n- Compares regression and classification models\n- Shows correlation, statistical tests, and customer segments")
    with right:
        st.markdown("### Practical concepts included")
        st.markdown("- Pandas, NumPy, missing-value checks, EDA\n- Linear, Ridge, Lasso, Decision Tree, Random Forest\n- Logistic Regression and classification metrics\n- K-Means, Gaussian Mixture, correlation, t-test, ANOVA, chi-square")

# ----------------------------- Prediction -----------------------------
elif page == "🔮 Bill Prediction":
    st.title("🔮 Electricity Bill Predictor")
    st.write("Enter household details to estimate the monthly bill. Season is selected directly and also mapped to a month for the model.")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            household_size = st.slider("Number of people", 1, 12, 4)
            appliance_hours = st.slider("Average appliance use (hours/day)", 1.0, 18.0, 7.0, 0.5)
            previous_units = st.number_input("Previous month's units (kWh)", min_value=0.0, max_value=3000.0, value=180.0, step=10.0)
        with c2:
            season = st.selectbox("Season", list(SEASON_MONTHS.keys()), index=1)
            season_month_names = [MONTH_NAMES[m - 1] for m in SEASON_MONTHS[season]]
            month_name = st.selectbox("Month", season_month_names)
            ac_hours = st.slider("AC use (hours/day)", 0.0, 18.0, 3.0, 0.5)
        with c3:
            heater_hours = st.slider("Heater use (hours/day)", 0.0, 12.0, 0.5, 0.5)
            temperature = st.slider("Average outdoor temperature (°C)", 0.0, 48.0, 30.0, 1.0)
            weekend = st.selectbox("Typical day pattern", ["Mostly weekdays", "Frequent weekend use"])
        model_name = st.selectbox("Prediction model", list(reg_models.keys()))
        submitted = st.form_submit_button("Predict electricity bill", type="primary", use_container_width=True)
    if submitted:
        month_num = MONTH_NAMES.index(month_name) + 1
        # Keep the user's chosen season visible; the model's numeric month is the actual model feature.
        row = pd.DataFrame([{
            "household_size": household_size, "appliance_hours": appliance_hours,
            "ac_hours": ac_hours, "heater_hours": heater_hours, "temperature": temperature,
            "previous_units": previous_units, "weekend": 1 if weekend == "Frequent weekend use" else 0,
            "month": month_num
        }])
        estimated_bill = max(0, float(reg_models[model_name].predict(row[features])[0]))
        # Estimate units with a secondary model trained on synthetic units.
        unit_model = RandomForestRegressor(n_estimators=100, random_state=42)
        unit_features = ["household_size", "appliance_hours", "ac_hours", "heater_hours", "temperature", "previous_units", "weekend", "month"]
        unit_model.fit(data[unit_features], data["units_consumed"])
        estimated_units = max(0, float(unit_model.predict(row[unit_features])[0]))
        st.success(f"Selected season: {season} | Selected month: {month_name}")
        a, b, c = st.columns(3)
        a.metric("Estimated monthly bill", f"₹{estimated_bill:,.0f}")
        b.metric("Estimated consumption", f"{estimated_units:,.0f} kWh")
        c.metric("Model used", model_name)
        st.caption("These are demo estimates from synthetic data, not an official utility tariff calculation. For meaningful estimates, train with your own past bills and local tariff slabs.")
        st.markdown("### What to do next")
        tips = []
        if ac_hours >= 4:
            tips.append("Try reducing AC usage by 30–60 minutes daily and use a moderate temperature setting.")
        if appliance_hours >= 8:
            tips.append("Switch off unused appliances and avoid leaving devices on standby.")
        if heater_hours >= 2:
            tips.append("Use a timer for high-power heating appliances.")
        if estimated_units > data["units_consumed"].quantile(0.75):
            tips.append("Your estimated use is high compared with this demo dataset; check high-power appliances and compare actual meter readings.")
        if not tips:
            tips.append("Record your meter reading monthly and compare usage across seasons.")
        for tip in tips:
            st.write("• " + tip)

# ----------------------------- Seasonal Analysis -----------------------------
elif page == "🌦️ Seasonal Analysis":
    st.title("🌦️ Season-wise Electricity Analysis")
    st.write("Compare electricity use and estimated bills across four illustrative seasons.")
    seasonal = data.groupby("season", as_index=False).agg(
        average_units=("units_consumed", "mean"),
        average_bill=("bill_amount", "mean"),
        median_bill=("bill_amount", "median"),
        records=("bill_amount", "size"),
        average_temperature=("temperature", "mean")
    )
    order = ["Winter", "Summer", "Monsoon", "Post-monsoon"]
    seasonal["season"] = pd.Categorical(seasonal["season"], categories=order, ordered=True)
    seasonal = seasonal.sort_values("season")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        ax.bar(seasonal["season"].astype(str), seasonal["average_units"])
        ax.set_title("Average Electricity Consumption by Season")
        ax.set_ylabel("Average units (kWh)")
        ax.set_xlabel("Season")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        ax.bar(seasonal["season"].astype(str), seasonal["average_bill"])
        ax.set_title("Average Estimated Bill by Season")
        ax.set_ylabel("Average bill (₹)")
        ax.set_xlabel("Season")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)
    st.dataframe(seasonal, use_container_width=True, hide_index=True)
    selected_season = st.selectbox("Inspect a season", order)
    subset = data[data["season"] == selected_season]
    a, b, c = st.columns(3)
    a.metric("Average bill", f"₹{subset.bill_amount.mean():,.0f}")
    b.metric("Average units", f"{subset.units_consumed.mean():.1f} kWh")
    c.metric("Average temperature", f"{subset.temperature.mean():.1f} °C")
    st.markdown("### Season-aware suggestions")
    suggestions = {
        "Winter": "Check room heaters and water heaters; use timers where possible.",
        "Summer": "AC and cooling may raise use; clean filters and set a moderate temperature.",
        "Monsoon": "Use daylight when possible and switch off unused lights and appliances.",
        "Post-monsoon": "Compare bills against summer and check whether cooling use can be reduced."
    }
    st.info(suggestions[selected_season])
    st.caption("Season definitions are illustrative for this demo. Adapt the month ranges to your local climate.")

# ----------------------------- Data Exploration -----------------------------
elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration (EDA)")
    st.write("This section demonstrates data inspection, preprocessing checks, and plots.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(data))
    c2.metric("Columns", len(data.columns))
    c3.metric("Missing values", int(data.isnull().sum().sum()))
    st.markdown("### Preview")
    st.dataframe(data.head(15), use_container_width=True)
    st.markdown("### Data quality checks")
    quality = pd.DataFrame({
        "Data type": data.dtypes.astype(str),
        "Missing values": data.isnull().sum(),
        "Duplicate rows": [int(data.duplicated().sum())] + [0] * (len(data.columns)-1)
    })
    st.dataframe(quality, use_container_width=True)
    st.markdown("### Descriptive statistics")
    st.dataframe(data.describe(include="all").T, use_container_width=True)
    col = st.selectbox("Choose a numeric column for distribution", data.select_dtypes(include=np.number).columns.tolist())
    fig, ax = plt.subplots()
    sns.histplot(data[col], kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")
    st.pyplot(fig)
    xcol = st.selectbox("Scatter plot X-axis", ["temperature", "household_size", "ac_hours", "previous_units"])
    ycol = st.selectbox("Scatter plot Y-axis", ["bill_amount", "units_consumed"])
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x=xcol, y=ycol, hue="season", ax=ax)
    ax.set_title(f"{ycol} vs {xcol}")
    st.pyplot(fig)
    st.download_button("Download demo dataset CSV", data.to_csv(index=False).encode("utf-8"), "electricity_demo_data.csv", "text/csv")

# ----------------------------- Model Lab -----------------------------
elif page == "🤖 Model Lab":
    st.title("🤖 Model Comparison Lab")
    st.markdown("### Regression: predict bill amount")
    st.dataframe(reg_metrics.round(3), use_container_width=True, hide_index=True)
    best_reg = reg_metrics.iloc[0]
    st.info(f"Lowest test RMSE in this run: {best_reg['Model']} (RMSE = {best_reg['RMSE']:.2f}). This is a synthetic-data demonstration.")
    st.markdown("### Classification: identify high-usage households")
    st.dataframe(clf_metrics.round(3), use_container_width=True, hide_index=True)
    chosen_clf = st.selectbox("Choose classifier for confusion matrix", list(clf_models.keys()))
    clf_pred = clf_models[chosen_clf].predict(Xc_test)
    cm = confusion_matrix(yc_test, clf_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "High"], yticklabels=["Normal", "High"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {chosen_clf}")
    st.pyplot(fig)
    st.code(classification_report(yc_test, clf_pred, target_names=["Normal usage", "High usage"], zero_division=0))
    st.caption("High usage is defined as the top 25% of consumption in this demo dataset.")

# ----------------------------- Statistics & Tests -----------------------------
elif page == "📈 Statistics & Tests":
    st.title("📈 Statistics, Correlation & Hypothesis Testing")
    numeric = data.select_dtypes(include=np.number)
    st.markdown("### Correlation heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)
    st.markdown("### Pearson correlation")
    c1, c2 = st.columns(2)
    with c1:
        x_name = st.selectbox("Variable X", numeric.columns.tolist(), index=numeric.columns.get_loc("ac_hours"))
    with c2:
        y_name = st.selectbox("Variable Y", numeric.columns.tolist(), index=numeric.columns.get_loc("bill_amount"))
    r, p = stats.pearsonr(data[x_name], data[y_name])
    st.write(f"Pearson correlation r = **{r:.3f}**, p-value = **{p:.4g}**.")
    st.caption("Correlation describes linear association, not causation.")
    st.markdown("### One-way ANOVA: does average bill differ by season?")
    groups = [data.loc[data.season == s, "bill_amount"].values for s in SEASON_MONTHS.keys()]
    f_stat, anova_p = stats.f_oneway(*groups)
    st.write(f"F-statistic = **{f_stat:.3f}**, p-value = **{anova_p:.4g}**.")
    st.write("At alpha = 0.05: " + ("Reject the null hypothesis; at least one seasonal mean may differ." if anova_p < 0.05 else "Fail to reject the null hypothesis; evidence of a seasonal mean difference is insufficient."))
    st.markdown("### Independent t-test: summer vs winter bills")
    summer = data.loc[data.season == "Summer", "bill_amount"]
    winter = data.loc[data.season == "Winter", "bill_amount"]
    t_stat, t_p = stats.ttest_ind(summer, winter, equal_var=False)
    st.write(f"t-statistic = **{t_stat:.3f}**, p-value = **{t_p:.4g}**.")
    st.markdown("### Chi-square test: season vs high usage")
    contingency = pd.crosstab(data["season"], data["high_usage"])
    chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
    st.write(f"Chi-square = **{chi2:.3f}**, degrees of freedom = **{dof}**, p-value = **{chi_p:.4g}**.")
    st.dataframe(contingency, use_container_width=True)
    st.caption("These tests illustrate the practicals on synthetic data. Check assumptions and use real observations for meaningful conclusions.")

# ----------------------------- Clustering -----------------------------
elif page == "🧩 Usage Segments":
    st.title("🧩 Household Usage Segments")
    st.write("K-Means and Gaussian Mixture Model (GMM) group records with similar usage patterns. This is clustering, not a prediction of a guaranteed customer type.")
    cluster_features = ["household_size", "appliance_hours", "ac_hours", "heater_hours", "previous_units", "bill_amount"]
    X = data[cluster_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    k = st.slider("Number of clusters", 2, 5, 3)
    algorithm = st.radio("Clustering algorithm", ["K-Means", "Gaussian Mixture Model (GMM)"], horizontal=True)
    if algorithm == "K-Means":
        clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = clusterer.fit_predict(X_scaled)
    else:
        clusterer = GaussianMixture(n_components=k, random_state=42)
        labels = clusterer.fit_predict(X_scaled)
    clustered = data.copy()
    clustered["cluster"] = labels.astype(str)
    st.dataframe(clustered.groupby("cluster")[cluster_features].mean().round(2), use_container_width=True)
    fig, ax = plt.subplots()
    sns.scatterplot(data=clustered, x="previous_units", y="bill_amount", hue="cluster", palette="tab10", ax=ax)
    ax.set_title(f"{algorithm} Segments")
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
    st.markdown("### Example: compare your usage")
    current = st.number_input("Current monthly units (kWh)", min_value=0.0, value=250.0, step=10.0)
    reduction = st.slider("Possible usage reduction (%)", 0, 40, 10)
    new_units = current * (1 - reduction/100)
    st.metric("Illustrative units after reduction", f"{new_units:.1f} kWh", f"-{current-new_units:.1f} kWh")
    st.caption("Actual bill savings depend on the utility tariff, fixed charges, taxes, and any subsidy or slab rules.")
