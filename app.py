import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

@st.cache_data
def load_data(file):
    df = load_data(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date")

st.set_page_config(page_title="Revenue Forecast Dashboard", layout="wide")

st.title("📊 AI Revenue Forecast Dashboard")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:

    df = load_data(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    st.success("✅ Data Loaded Successfully")

    # -----------------------------
    # TRAIN MODEL
    # -----------------------------
    features = [
        "Demand_Index","Project_Hours","Market_Trend_Index",
        "Client","Service","Marketing_Campaign","Season"
    ]

    X = df[features].copy().astype(str)

    encoders = {}
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    y = df["Revenue_INR"]

    split = int(len(df)*0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # -----------------------------
    # DASHBOARD SECTION
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Actual vs Predicted")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=y_test.values,
            mode='lines',
            name='Actual'
        ))

        fig.add_trace(go.Scatter(
            y=preds,
            mode='lines',
            name='Predicted'
        ))

        fig.update_layout(template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Key Metrics")

        mae = np.mean(np.abs(y_test - preds))
        rmse = np.sqrt(np.mean((y_test - preds)**2))

        st.metric("MAE", f"{mae:,.0f}")
        st.metric("RMSE", f"{rmse:,.0f}")

    # -----------------------------
    # BUSINESS SIMULATOR
    # -----------------------------
    st.subheader("🎯 Revenue Prediction Simulator")

    col1, col2, col3 = st.columns(3)

    with col1:
        demand = st.slider("Demand Index", 0, 100, 50)
        hours = st.slider("Project Hours", 0, 500, 200)

    with col2:
        market = st.slider("Market Index", 0, 100, 50)
        client = st.selectbox("Client", df["Client"].astype(str).unique())

    with col3:
        service = st.selectbox("Service", df["Service"].astype(str).unique())
        campaign = st.selectbox("Campaign", df["Marketing_Campaign"].astype(str).unique())
        season = st.selectbox("Season", df["Season"].astype(str).unique())

    if st.button("Predict Revenue"):

        input_dict = {
            "Demand_Index": demand,
            "Project_Hours": hours,
            "Market_Trend_Index": market,
            "Client": client,
            "Service": service,
            "Marketing_Campaign": campaign,
            "Season": season
        }

        input_df = pd.DataFrame([input_dict])

        for col in input_df.columns:
            input_df[col] = encoders[col].transform(input_df[col].astype(str))

        prediction = model.predict(input_df)[0]

        st.success(f"💰 Predicted Revenue: ₹ {prediction:,.0f}")

    # -----------------------------
    # TIME SERIES VIEW
    # -----------------------------
    st.subheader("📅 Revenue Trend")

    daily = (
        df.groupby("Date")["Revenue_INR"]
          .sum()
          .reset_index()
    )

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=daily["Date"],
        y=daily["Revenue_INR"],
        mode='lines',
        name="Revenue"
    ))

    st.plotly_chart(fig2, use_container_width=True)
