import streamlit as st
import pandas as pd
import joblib

kmeans = joblib.load("model_kmeans.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="🛍 Customer Segmentation", page_icon="📊", layout="centered")

st.markdown(
    """
    <h1 style='text-align:center; color:#4CAF50;'>🛍 Customer Segmentation App</h1>
    <p style='text-align:center; font-size:18px;'>Predict which customer segment best fits your client's profile</p>
    <hr>
    """,
    unsafe_allow_html=True
)

st.subheader("📋 Enter Customer Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age", min_value=18, max_value=100, value=35)
    income = st.number_input("💰 Income", min_value=0, max_value=200000, value=50000)
    total_spending = st.number_input("🛒 Total Spending", min_value=0, max_value=5000, value=1000)
    recency = st.number_input("⏳ Recency (days since last buy)", min_value=0, max_value=365, value=7)

with col2:
    num_web_purchases = st.number_input("🛍 Web Purchases", min_value=0, max_value=100, value=10)
    num_store_purchases = st.number_input("🏪 Store Purchases", min_value=0, max_value=100, value=10)
    num_web_visits = st.number_input("🌐 Web Visits / month", min_value=0, max_value=50, value=5)

input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})

input_scaled = scaler.transform(input_data)

cluster_info = {
    0: {"emoji": "💼", "desc": "High budget, frequent web visitors", "color": "#2196F3"},
    1: {"emoji": "💎", "desc": "High spenders with premium habits", "color": "#9C27B0"},
    2: {"emoji": "🌐", "desc": "Frequent web visitors, moderate spenders", "color": "#FF9800"},
    3: {"emoji": "🏠", "desc": "Loyal store shoppers", "color": "#4CAF50"},
    4: {"emoji": "🛍", "desc": "Occasional shoppers with balanced habits", "color": "#FFC107"},
    5: {"emoji": "📉", "desc": "Low spenders, infrequent visits", "color": "#F44336"}
}

if st.button("🔍 Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]

    st.markdown(f"### 🎯 Predicted Segment: **Cluster {cluster}** {cluster_info[cluster]['emoji']}")
    st.markdown("---")

    for cid, info in cluster_info.items():
        bg_color = info["color"] if cid == cluster else "#f0f0f0"
        text_color = "white" if cid == cluster else "black"

        st.markdown(
            f"""
            <div style="
                background-color:{bg_color};
                color:{text_color};
                padding:12px;
                border-radius:10px;
                margin-bottom:8px;
            ">
            <b>Cluster {cid} {info['emoji']}</b> – {info['desc']}
            </div>
            """,
            unsafe_allow_html=True
        )
