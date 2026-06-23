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
    initial_sidebar_state="expanded",
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
# FEATURE ORDER — harus sama persis dengan X saat training
# df_fs.drop('churn', axis=1) setelah drop customer_id, gender,
# signup_date, last_purchase_date dan tambah customer_age_days,
# days_since_last_purchase
# ─────────────────────────────────────────────
FEATURE_COLUMNS = [
    "age",
    "is_premium_user",
    "total_visits",
    "avg_session_time",
    "pages_per_session",
    "email_open_rate",
    "email_click_rate",
    "total_spent",
    "avg_order_value",
    "discount_used",
    "support_tickets",
    "refund_requested",
    "delivery_delay_days",
    "satisfaction_score",
    "nps_score",
    "marketing_spend_per_user",
    "lifetime_value",
    "last_3_month_purchase_freq",
    "customer_age_days",
    "days_since_last_purchase",
]

FEATURE_INFO = {
    "age":                      "Usia pelanggan (tahun)",
    "is_premium_user":          "Apakah pelanggan adalah pengguna premium (0 = Tidak, 1 = Ya)",
    "total_visits":             "Total kunjungan pelanggan ke platform",
    "avg_session_time":         "Rata-rata durasi sesi pelanggan (menit)",
    "pages_per_session":        "Rata-rata halaman yang dikunjungi per sesi",
    "email_open_rate":          "Tingkat pembukaan email marketing (0.0 - 1.0)",
    "email_click_rate":         "Tingkat klik email marketing (0.0 - 1.0)",
    "total_spent":              "Total pengeluaran pelanggan (USD)",
    "avg_order_value":          "Rata-rata nilai pesanan pelanggan (USD)",
    "discount_used":            "Jumlah diskon yang digunakan pelanggan",
    "support_tickets":          "Jumlah tiket support yang dibuka pelanggan",
    "refund_requested":         "Jumlah permintaan refund yang dilakukan pelanggan",
    "delivery_delay_days":      "Rata-rata keterlambatan pengiriman (hari)",
    "satisfaction_score":       "Skor kepuasan pelanggan (1 - 5)",
    "nps_score":                "Net Promoter Score pelanggan (-100 hingga 100)",
    "marketing_spend_per_user": "Pengeluaran marketing per pelanggan (USD)",
    "lifetime_value":           "Nilai seumur hidup pelanggan / LTV (USD)",
    "last_3_month_purchase_freq": "Frekuensi pembelian dalam 3 bulan terakhir",
    "customer_age_days":        "Lama pelanggan terdaftar sejak signup (hari)",
    "days_since_last_purchase": "Jumlah hari sejak pembelian terakhir",
}

# ─────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_churn(model, scaler, input_df: pd.DataFrame):
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
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Model Information")
    st.markdown("""
| Field | Detail |
|---|---|
| **Algorithm** | Random Forest |
| **Dataset** | Sales and Marketing Customer Dataset |
| **Target** | Churn |
| **Features** | 20 fitur numerik |
| **Deployment** | Streamlit Cloud |
""")

    st.divider()
    st.header("Feature Information")
    for feat, desc in FEATURE_INFO.items():
        st.markdown(f"**{feat}**  \n{desc}")

    st.divider()

    try:
        model_sidebar = load_model()
        if hasattr(model_sidebar, "feature_importances_"):
            st.header("Feature Importance")
            importances = model_sidebar.feature_importances_
            fi_df = pd.DataFrame({
                "Feature": FEATURE_COLUMNS,
                "Importance": importances,
            }).sort_values("Importance", ascending=True)
            st.bar_chart(fi_df.set_index("Feature"))
    except Exception:
        pass

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
st.markdown("Isi seluruh informasi pelanggan di bawah ini, kemudian klik **Predict Churn**.")

# ─────────────────────────────────────────────
# INPUT FORM — 3 kolom
# ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age (tahun)", min_value=18, max_value=100, value=35, step=1,
                          help=FEATURE_INFO["age"])

    is_premium_user = st.selectbox("Is Premium User", options=["Tidak (0)", "Ya (1)"],
                                   help=FEATURE_INFO["is_premium_user"])
    is_premium_val = 1 if "Ya" in is_premium_user else 0

    total_visits = st.number_input("Total Visits", min_value=0, max_value=10000, value=50, step=1,
                                   help=FEATURE_INFO["total_visits"])

    avg_session_time = st.number_input("Avg Session Time (menit)", min_value=0.0, max_value=300.0,
                                       value=15.0, step=0.5, format="%.1f",
                                       help=FEATURE_INFO["avg_session_time"])

    pages_per_session = st.number_input("Pages per Session", min_value=0.0, max_value=100.0,
                                        value=5.0, step=0.5, format="%.1f",
                                        help=FEATURE_INFO["pages_per_session"])

    email_open_rate = st.number_input("Email Open Rate (0.0 - 1.0)", min_value=0.0, max_value=1.0,
                                      value=0.3, step=0.01, format="%.2f",
                                      help=FEATURE_INFO["email_open_rate"])

    email_click_rate = st.number_input("Email Click Rate (0.0 - 1.0)", min_value=0.0, max_value=1.0,
                                       value=0.1, step=0.01, format="%.2f",
                                       help=FEATURE_INFO["email_click_rate"])

with col2:
    total_spent = st.number_input("Total Spent (USD)", min_value=0.0, max_value=100000.0,
                                  value=500.0, step=10.0, format="%.2f",
                                  help=FEATURE_INFO["total_spent"])

    avg_order_value = st.number_input("Avg Order Value (USD)", min_value=0.0, max_value=10000.0,
                                      value=75.0, step=5.0, format="%.2f",
                                      help=FEATURE_INFO["avg_order_value"])

    discount_used = st.number_input("Discount Used", min_value=0, max_value=500, value=5, step=1,
                                    help=FEATURE_INFO["discount_used"])

    support_tickets = st.number_input("Support Tickets", min_value=0, max_value=100, value=2, step=1,
                                      help=FEATURE_INFO["support_tickets"])

    refund_requested = st.number_input("Refund Requested", min_value=0, max_value=50, value=0, step=1,
                                       help=FEATURE_INFO["refund_requested"])

    delivery_delay_days = st.number_input("Delivery Delay Days", min_value=0.0, max_value=60.0,
                                          value=2.0, step=0.5, format="%.1f",
                                          help=FEATURE_INFO["delivery_delay_days"])

    satisfaction_score = st.number_input("Satisfaction Score (1 - 5)", min_value=1.0, max_value=5.0,
                                         value=3.5, step=0.1, format="%.1f",
                                         help=FEATURE_INFO["satisfaction_score"])

with col3:
    nps_score = st.number_input("NPS Score (-100 - 100)", min_value=-100, max_value=100,
                                value=20, step=1, help=FEATURE_INFO["nps_score"])

    marketing_spend_per_user = st.number_input("Marketing Spend per User (USD)", min_value=0.0,
                                               max_value=5000.0, value=50.0, step=1.0,
                                               format="%.2f",
                                               help=FEATURE_INFO["marketing_spend_per_user"])

    lifetime_value = st.number_input("Lifetime Value / LTV (USD)", min_value=0.0,
                                     max_value=100000.0, value=1000.0, step=50.0,
                                     format="%.2f", help=FEATURE_INFO["lifetime_value"])

    last_3_month_purchase_freq = st.number_input("Last 3 Month Purchase Freq", min_value=0,
                                                 max_value=200, value=5, step=1,
                                                 help=FEATURE_INFO["last_3_month_purchase_freq"])

    customer_age_days = st.number_input("Customer Age (hari sejak signup)", min_value=0,
                                        max_value=5000, value=365, step=1,
                                        help=FEATURE_INFO["customer_age_days"])

    days_since_last_purchase = st.number_input("Days Since Last Purchase", min_value=0,
                                               max_value=1000, value=30, step=1,
                                               help=FEATURE_INFO["days_since_last_purchase"])

st.divider()

# ─────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────
predict_btn = st.button("Predict Churn", type="primary", use_container_width=True)

if predict_btn:
    input_data = pd.DataFrame(
        [[
            age,
            is_premium_val,
            total_visits,
            avg_session_time,
            pages_per_session,
            email_open_rate,
            email_click_rate,
            total_spent,
            avg_order_value,
            discount_used,
            support_tickets,
            refund_requested,
            delivery_delay_days,
            satisfaction_score,
            nps_score,
            marketing_spend_per_user,
            lifetime_value,
            last_3_month_purchase_freq,
            customer_age_days,
            days_since_last_purchase,
        ]],
        columns=FEATURE_COLUMNS,
    )

    prediction, churn_prob, no_churn_prob = predict_churn(model, scaler, input_data)

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
    st.caption("Prediksi dibuat menggunakan Random Forest Best Model yang telah di-training dan di-tune sebelumnya.")