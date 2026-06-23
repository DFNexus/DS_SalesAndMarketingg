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
# FEATURE ORDER — diambil langsung dari scaler.feature_names_in_
# 27 fitur, urutan HARUS sama persis
# ─────────────────────────────────────────────
FEATURE_COLUMNS = [
    "age",
    "country",
    "city",
    "acquisition_channel",
    "device_type",
    "subscription_type",
    "is_premium_user",
    "total_visits",
    "avg_session_time",
    "pages_per_session",
    "email_open_rate",
    "email_click_rate",
    "total_spent",
    "avg_order_value",
    "discount_used",
    "coupon_code",
    "support_tickets",
    "refund_requested",
    "delivery_delay_days",
    "payment_method",
    "satisfaction_score",
    "nps_score",
    "marketing_spend_per_user",
    "lifetime_value",
    "last_3_month_purchase_freq",
    "customer_age_days",
    "days_since_last_purchase",
]

FEATURE_INFO = {
    "age":                       "Usia pelanggan (tahun)",
    "country":                   "Negara pelanggan (encoded)",
    "city":                      "Kota pelanggan (encoded)",
    "acquisition_channel":       "Saluran akuisisi pelanggan (encoded)",
    "device_type":               "Tipe perangkat yang digunakan (encoded)",
    "subscription_type":         "Tipe langganan pelanggan (encoded)",
    "is_premium_user":           "Status premium pelanggan (0=Tidak, 1=Ya)",
    "total_visits":              "Total kunjungan ke platform",
    "avg_session_time":          "Rata-rata durasi sesi (menit)",
    "pages_per_session":         "Rata-rata halaman per sesi",
    "email_open_rate":           "Tingkat pembukaan email (0.0 - 1.0)",
    "email_click_rate":          "Tingkat klik email (0.0 - 1.0)",
    "total_spent":               "Total pengeluaran pelanggan (USD)",
    "avg_order_value":           "Rata-rata nilai pesanan (USD)",
    "discount_used":             "Penggunaan diskon (0=Tidak, 1=Ya)",
    "coupon_code":               "Kode kupon yang digunakan (encoded)",
    "support_tickets":           "Jumlah tiket support yang dibuka",
    "refund_requested":          "Permintaan refund (0=Tidak, 1=Ya)",
    "delivery_delay_days":       "Rata-rata keterlambatan pengiriman (hari)",
    "payment_method":            "Metode pembayaran (encoded)",
    "satisfaction_score":        "Skor kepuasan pelanggan (1 - 5)",
    "nps_score":                 "Net Promoter Score (1 - 10)",
    "marketing_spend_per_user":  "Pengeluaran marketing per pelanggan (USD)",
    "lifetime_value":            "Lifetime Value / LTV pelanggan (USD)",
    "last_3_month_purchase_freq":"Frekuensi pembelian 3 bulan terakhir",
    "customer_age_days":         "Lama terdaftar sejak signup (hari)",
    "days_since_last_purchase":  "Hari sejak pembelian terakhir",
}

# Nilai encoded dari LabelEncoder — sesuai urutan alfabetis sklearn
# (sklearn LabelEncoder mengurutkan secara alfabetis)
COUNTRY_OPTIONS    = {"Australia": 0, "Brazil": 1, "Canada": 2, "Germany": 3, "India": 4, "UK": 5}
CITY_OPTIONS       = {"Berlin": 0, "Chennai": 1, "London": 2, "Mumbai": 3, "New York": 4, "Sydney": 5, "Toronto": 6}
ACQ_CHANNEL        = {"Direct": 0, "Email": 1, "Organic": 2, "Paid Ad": 3, "Referral": 4, "Social Media": 5}
DEVICE_TYPE        = {"Desktop": 0, "Mobile": 1, "Tablet": 2}
SUBSCRIPTION_TYPE  = {"Free": 0, "Premium": 1}
COUPON_CODE        = {"BLACKFRIDAY": 0, "NEWYEAR": 1, "None": 2, "SAVE10": 3, "SUMMER": 4, "WELCOME": 5}
PAYMENT_METHOD     = {"Bank Transfer": 0, "Credit Card": 1, "Debit Card": 2, "PayPal": 3, "UPI": 4}

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
| **Features** | 27 fitur |
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

    country_label = st.selectbox("Country", options=list(COUNTRY_OPTIONS.keys()),
                                 help=FEATURE_INFO["country"])
    country_val = COUNTRY_OPTIONS[country_label]

    city_label = st.selectbox("City", options=list(CITY_OPTIONS.keys()),
                              help=FEATURE_INFO["city"])
    city_val = CITY_OPTIONS[city_label]

    acq_label = st.selectbox("Acquisition Channel", options=list(ACQ_CHANNEL.keys()),
                             help=FEATURE_INFO["acquisition_channel"])
    acq_val = ACQ_CHANNEL[acq_label]

    device_label = st.selectbox("Device Type", options=list(DEVICE_TYPE.keys()),
                                help=FEATURE_INFO["device_type"])
    device_val = DEVICE_TYPE[device_label]

    sub_label = st.selectbox("Subscription Type", options=list(SUBSCRIPTION_TYPE.keys()),
                             help=FEATURE_INFO["subscription_type"])
    sub_val = SUBSCRIPTION_TYPE[sub_label]

    is_premium_label = st.selectbox("Is Premium User", options=["Tidak (0)", "Ya (1)"],
                                    help=FEATURE_INFO["is_premium_user"])
    is_premium_val = 1 if "Ya" in is_premium_label else 0

    total_visits = st.number_input("Total Visits", min_value=0, max_value=10000, value=15, step=1,
                                   help=FEATURE_INFO["total_visits"])

    avg_session_time = st.number_input("Avg Session Time (menit)", min_value=0.0, max_value=300.0,
                                       value=8.0, step=0.5, format="%.1f",
                                       help=FEATURE_INFO["avg_session_time"])

with col2:
    pages_per_session = st.number_input("Pages per Session", min_value=0.0, max_value=100.0,
                                        value=4.0, step=0.5, format="%.1f",
                                        help=FEATURE_INFO["pages_per_session"])

    email_open_rate = st.number_input("Email Open Rate (0.0 - 1.0)", min_value=0.0, max_value=1.0,
                                      value=0.5, step=0.01, format="%.2f",
                                      help=FEATURE_INFO["email_open_rate"])

    email_click_rate = st.number_input("Email Click Rate (0.0 - 1.0)", min_value=0.0, max_value=1.0,
                                       value=0.25, step=0.01, format="%.2f",
                                       help=FEATURE_INFO["email_click_rate"])

    total_spent = st.number_input("Total Spent (USD)", min_value=0.0, max_value=100000.0,
                                  value=522.0, step=10.0, format="%.2f",
                                  help=FEATURE_INFO["total_spent"])

    avg_order_value = st.number_input("Avg Order Value (USD)", min_value=0.0, max_value=10000.0,
                                      value=60.0, step=5.0, format="%.2f",
                                      help=FEATURE_INFO["avg_order_value"])

    discount_used_label = st.selectbox("Discount Used", options=["Tidak (0)", "Ya (1)"],
                                       help=FEATURE_INFO["discount_used"])
    discount_val = 1 if "Ya" in discount_used_label else 0

    coupon_label = st.selectbox("Coupon Code", options=list(COUPON_CODE.keys()),
                                help=FEATURE_INFO["coupon_code"])
    coupon_val = COUPON_CODE[coupon_label]

    support_tickets = st.number_input("Support Tickets", min_value=0, max_value=100, value=2, step=1,
                                      help=FEATURE_INFO["support_tickets"])

    refund_label = st.selectbox("Refund Requested", options=["Tidak (0)", "Ya (1)"],
                                help=FEATURE_INFO["refund_requested"])
    refund_val = 1 if "Ya" in refund_label else 0

with col3:
    delivery_delay_days = st.number_input("Delivery Delay Days", min_value=0.0, max_value=60.0,
                                          value=3.0, step=0.5, format="%.1f",
                                          help=FEATURE_INFO["delivery_delay_days"])

    payment_label = st.selectbox("Payment Method", options=list(PAYMENT_METHOD.keys()),
                                 help=FEATURE_INFO["payment_method"])
    payment_val = PAYMENT_METHOD[payment_label]

    satisfaction_score = st.number_input("Satisfaction Score (1 - 5)", min_value=1.0, max_value=5.0,
                                         value=3.6, step=0.1, format="%.1f",
                                         help=FEATURE_INFO["satisfaction_score"])

    nps_score = st.number_input("NPS Score (1 - 10)", min_value=1, max_value=10,
                                value=5, step=1, help=FEATURE_INFO["nps_score"])

    marketing_spend_per_user = st.number_input("Marketing Spend per User (USD)", min_value=0.0,
                                               max_value=5000.0, value=17.5, step=1.0,
                                               format="%.2f",
                                               help=FEATURE_INFO["marketing_spend_per_user"])

    lifetime_value = st.number_input("Lifetime Value / LTV (USD)", min_value=0.0,
                                     max_value=100000.0, value=1233.0, step=50.0,
                                     format="%.2f", help=FEATURE_INFO["lifetime_value"])

    last_3_month_purchase_freq = st.number_input("Last 3 Month Purchase Freq", min_value=0,
                                                 max_value=200, value=7, step=1,
                                                 help=FEATURE_INFO["last_3_month_purchase_freq"])

    customer_age_days = st.number_input("Customer Age (hari sejak signup)", min_value=0,
                                        max_value=50000, value=365, step=1,
                                        help=FEATURE_INFO["customer_age_days"])

    days_since_last_purchase = st.number_input("Days Since Last Purchase", min_value=0,
                                               max_value=50000, value=30, step=1,
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
            country_val,
            city_val,
            acq_val,
            device_val,
            sub_val,
            is_premium_val,
            total_visits,
            avg_session_time,
            pages_per_session,
            email_open_rate,
            email_click_rate,
            total_spent,
            avg_order_value,
            discount_val,
            coupon_val,
            support_tickets,
            refund_val,
            delivery_delay_days,
            payment_val,
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