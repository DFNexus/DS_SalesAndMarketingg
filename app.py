import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

# ─────────────────────────────────────────────
# 27 FITUR LENGKAP — urutan sesuai scaler.feature_names_in_
# ─────────────────────────────────────────────
FEATURE_COLUMNS = [
    "age", "country", "city", "acquisition_channel", "device_type",
    "subscription_type", "is_premium_user", "total_visits", "avg_session_time",
    "pages_per_session", "email_open_rate", "email_click_rate", "total_spent",
    "avg_order_value", "discount_used", "coupon_code", "support_tickets",
    "refund_requested", "delivery_delay_days", "payment_method",
    "satisfaction_score", "nps_score", "marketing_spend_per_user",
    "lifetime_value", "last_3_month_purchase_freq", "customer_age_days",
    "days_since_last_purchase",
]

# Nilai median/default untuk 17 fitur yang tidak ditampilkan ke user
# Diambil dari scaler.mean_ sebagai representasi nilai tengah dataset
FEATURE_DEFAULTS = {
    "age":                      35,
    "country":                  2,
    "city":                     3,
    "acquisition_channel":      2,
    "device_type":              1,
    "subscription_type":        1,
    "is_premium_user":          0,
    "total_visits":             15,
    "email_open_rate":          0.5,
    "email_click_rate":         0.25,
    "discount_used":            0,
    "coupon_code":              2,
    "refund_requested":         0,
    "delivery_delay_days":      3.0,
    "payment_method":           2,
    "nps_score":                5,
    "last_3_month_purchase_freq": 7,
}

# 10 TOP FEATURES — yang ditampilkan ke user
TOP_FEATURE_INFO = {
    "total_spent":              "Total pengeluaran pelanggan selama berlangganan (USD)",
    "satisfaction_score":       "Skor kepuasan pelanggan terhadap layanan (1.0 - 5.0)",
    "support_tickets":          "Jumlah tiket support/keluhan yang dibuka pelanggan",
    "avg_session_time":         "Rata-rata durasi sesi penggunaan platform (menit)",
    "days_since_last_purchase": "Jumlah hari sejak transaksi terakhir pelanggan",
    "marketing_spend_per_user": "Pengeluaran marketing yang dialokasikan per pelanggan (USD)",
    "lifetime_value":           "Estimasi total nilai pelanggan selama masa langganan / LTV (USD)",
    "pages_per_session":        "Rata-rata jumlah halaman yang dikunjungi per sesi",
    "avg_order_value":          "Rata-rata nilai per transaksi pelanggan (USD)",
    "customer_age_days":        "Lama pelanggan terdaftar sejak pertama signup (hari)",
}

# ─────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_churn(model, scaler, user_input: dict):
    # Mulai dari default values untuk semua 27 fitur
    row = FEATURE_DEFAULTS.copy()
    # Timpa dengan input user (10 fitur terbaik)
    row.update(user_input)
    # Susun sesuai urutan FEATURE_COLUMNS
    input_df = pd.DataFrame([[row[f] for f in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]
    churn_prob = proba[1] * 100
    no_churn_prob = proba[0] * 100
    return prediction, churn_prob, no_churn_prob

def get_confidence_label(prob: float) -> str:
    if prob >= 90:
        return "Sangat Tinggi (>= 90%)"
    elif prob >= 75:
        return "Tinggi (75% - 89%)"
    elif prob >= 60:
        return "Sedang (60% - 74%)"
    else:
        return "Rendah (< 60%)"

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.title("Customer Churn Prediction")
st.caption("Prediksi kemungkinan pelanggan churn menggunakan Random Forest Best Model")
st.divider()

try:
    model = load_model()
    scaler = load_scaler()
except FileNotFoundError as e:
    st.error(
        f"File model tidak ditemukan: {e}\n\n"
        "Pastikan `random_forest_model.pkl` dan `scaler.pkl` berada "
        "di direktori yang sama dengan `app.py`."
    )
    st.stop()

st.subheader("Input Data Pelanggan")
st.markdown("Masukkan 10 fitur terpenting pelanggan, kemudian klik **Predict Churn**.")

# ─────────────────────────────────────────────
# INPUT FORM — 2 kolom, 10 fitur terbaik saja
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    total_spent = st.number_input(
        "Total Spent (USD)", min_value=0.0, max_value=100000.0,
        value=522.0, step=10.0, format="%.2f",
        help=TOP_FEATURE_INFO["total_spent"])

    satisfaction_score = st.number_input(
        "Satisfaction Score (1.0 - 5.0)", min_value=1.0, max_value=5.0,
        value=3.6, step=0.1, format="%.1f",
        help=TOP_FEATURE_INFO["satisfaction_score"])

    support_tickets = st.number_input(
        "Support Tickets", min_value=0, max_value=100, value=2, step=1,
        help=TOP_FEATURE_INFO["support_tickets"])

    avg_session_time = st.number_input(
        "Avg Session Time (menit)", min_value=0.0, max_value=300.0,
        value=8.0, step=0.5, format="%.1f",
        help=TOP_FEATURE_INFO["avg_session_time"])

    days_since_last_purchase = st.number_input(
        "Days Since Last Purchase", min_value=0, max_value=50000, value=30, step=1,
        help=TOP_FEATURE_INFO["days_since_last_purchase"])

with col2:
    marketing_spend_per_user = st.number_input(
        "Marketing Spend per User (USD)", min_value=0.0, max_value=5000.0,
        value=17.5, step=1.0, format="%.2f",
        help=TOP_FEATURE_INFO["marketing_spend_per_user"])

    lifetime_value = st.number_input(
        "Lifetime Value / LTV (USD)", min_value=0.0, max_value=100000.0,
        value=1233.0, step=50.0, format="%.2f",
        help=TOP_FEATURE_INFO["lifetime_value"])

    pages_per_session = st.number_input(
        "Pages per Session", min_value=0.0, max_value=100.0,
        value=4.0, step=0.5, format="%.1f",
        help=TOP_FEATURE_INFO["pages_per_session"])

    avg_order_value = st.number_input(
        "Avg Order Value (USD)", min_value=0.0, max_value=10000.0,
        value=60.0, step=5.0, format="%.2f",
        help=TOP_FEATURE_INFO["avg_order_value"])

    customer_age_days = st.number_input(
        "Customer Age (hari sejak signup)", min_value=0, max_value=50000,
        value=365, step=1,
        help=TOP_FEATURE_INFO["customer_age_days"])

st.divider()

# ─────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────
predict_btn = st.button("Predict Churn", type="primary", use_container_width=True)

if predict_btn:
    user_input = {
        "total_spent":              total_spent,
        "satisfaction_score":       satisfaction_score,
        "support_tickets":          support_tickets,
        "avg_session_time":         avg_session_time,
        "days_since_last_purchase": days_since_last_purchase,
        "marketing_spend_per_user": marketing_spend_per_user,
        "lifetime_value":           lifetime_value,
        "pages_per_session":        pages_per_session,
        "avg_order_value":          avg_order_value,
        "customer_age_days":        customer_age_days,
    }

    prediction, churn_prob, no_churn_prob = predict_churn(model, scaler, user_input)

    st.subheader("Hasil Prediksi")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if prediction == 1:
            st.error("Customer Berpotensi Churn")
            st.metric(label="Churn Probability", value=f"{churn_prob:.2f}%")
            st.metric(label="Confidence Level", value=get_confidence_label(churn_prob))
        else:
            st.success("Customer Tidak Berpotensi Churn")
            st.metric(label="Churn Probability", value=f"{churn_prob:.2f}%")
            st.metric(label="Confidence Level", value=get_confidence_label(no_churn_prob))

    with res_col2:
        prob_df = pd.DataFrame(
            {"Status": ["Tidak Churn", "Churn"], "Probabilitas (%)": [no_churn_prob, churn_prob]}
        ).set_index("Status")
        st.bar_chart(prob_df)

    st.divider()

    # Penjelasan fitur yang digunakan
    st.subheader("Penjelasan Fitur Input")
    for feat, desc in TOP_FEATURE_INFO.items():
        st.markdown(f"**{feat}** — {desc}")

    st.caption("Prediksi dibuat menggunakan Random Forest Best Model yang telah di-training dan di-tune sebelumnya.")