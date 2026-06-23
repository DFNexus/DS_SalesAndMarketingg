import io
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# 
# Page configuration
# 
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title(" Customer Churn Prediction Dashboard")
st.caption("Analisis & Prediksi Churn Pelanggan | Machine Learning Dashboard")
st.divider()

# 
# Sidebar
# 
with st.sidebar:
    st.header(" Konfigurasi")
    uploaded_file = st.file_uploader(" Upload Dataset CSV", type=["csv"])
    st.divider()
    st.markdown("""
**Panduan Penggunaan:**
1.  Upload file CSV dataset
2.  Eksplorasi data di tab **EDA**
3.  Jalankan **Direct Modeling**
4.  Coba **Preprocessing + Modeling**
5.  Lihat **Feature Selection** & simpan model
""")

if uploaded_file is None:
    st.info(" Silakan upload file CSV dataset pelanggan di sidebar untuk memulai analisis.")
    st.stop()

# 
# Load data
# 
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df_raw = load_data(uploaded_file)

if "churn" not in df_raw.columns:
    st.error(" Kolom 'churn' tidak ditemukan. Pastikan dataset memiliki kolom target 'churn'.")
    st.stop()

# 
# Helper utilities
# 
def plot_conf_matrix(cm, title, ax):
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


def report_to_df(y_true, y_pred):
    return pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True)
    ).transpose()


def safe_subplots(n_items, n_cols, fw=5, fh=4):
    """Return (fig, flat_axes) with squeeze=False for safe flattening."""
    n_rows = max(1, (n_items + n_cols - 1) // n_cols)
    n_cols = min(n_cols, n_items)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fw * n_cols, fh * n_rows), squeeze=False)
    return fig, axes.flatten()


def preprocess_base(df):
    """
    Common preprocessing pipeline:
      - Fill numeric NaN with median
      - Remove outliers on 'age' via IQR
      - LabelEncode categoricals
      - Drop customer_id, gender, date columns (with feature engineering first)
    Returns the processed DataFrame.
    """
    df = df.copy()

    # Identify column types
    date_cols = [c for c in df.columns if "date" in c.lower()]
    id_cols   = [c for c in ["customer_id"] if c in df.columns]
    drop_now  = id_cols + date_cols

    # Feature engineering from date columns
    for dc in date_cols:
        try:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
            if "signup" in dc.lower():
                df["customer_age_days"] = (pd.Timestamp.now() - df[dc]).dt.days.fillna(0).astype(int)
            if "purchase" in dc.lower() or "last" in dc.lower():
                df["days_since_last_purchase"] = (pd.Timestamp.now() - df[dc]).dt.days.fillna(0).astype(int)
        except Exception:
            pass

    df.drop(columns=[c for c in drop_now if c in df.columns], inplace=True, errors="ignore")

    # Fill numeric missing with median
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.difference(["churn"])
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Outlier removal on 'age'
    if "age" in df.columns:
        Q1, Q3 = df["age"].quantile(0.25), df["age"].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df["age"] >= Q1 - 1.5 * IQR) & (df["age"] <= Q3 + 1.5 * IQR)]

    # Drop irrelevant columns
    for col in ["gender"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Label encode remaining object columns
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def build_models():
    lr  = LogisticRegression(max_iter=1000, random_state=42)
    rf  = RandomForestClassifier(random_state=42)
    knn = KNeighborsClassifier()
    svm = SVC(probability=True)
    voting = VotingClassifier(
        estimators=[
            ("lr",  LogisticRegression(max_iter=1000)),
            ("knn", KNeighborsClassifier()),
            ("svm", SVC(probability=True)),
        ],
        voting="soft",
    )
    return {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "Voting Classifier": voting,
    }


def train_and_show(models_dict, X_train, y_train, X_test, y_test):
    summary = []
    for name, model in models_dict.items():
        st.markdown(f"####  {name}")
        with st.spinner(f"Melatih {name}..."):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm  = confusion_matrix(y_test, y_pred)
            rep = report_to_df(y_test, y_pred)
            summary.append({"Model": name, "Accuracy": round(acc, 4)})

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Accuracy", f"{acc:.4f}")
            st.dataframe(rep.round(4), use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(5, 4))
            plot_conf_matrix(cm, f"Confusion Matrix - {name}", ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        st.divider()

    st.subheader(" Ringkasan Perbandingan Model")
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)


# 
# TABS
# 
tab1, tab2, tab3, tab4 = st.tabs([
    " EDA",
    " Direct Modeling",
    " Preprocessing + Modeling",
    " Feature Selection & Best Model",
])

# 
# TAB 1 - EDA
# 
with tab1:
    st.header("Exploratory Data Analysis")

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Baris",    f"{df_raw.shape[0]:,}")
    c2.metric("Jumlah Kolom",    df_raw.shape[1])
    c3.metric("Duplikasi",       f"{df_raw.duplicated().sum():,}")
    c4.metric("Total Missing",   f"{df_raw.isnull().sum().sum():,}")
    st.divider()

    #  Data Preview 
    st.subheader(" Preview Data")
    n_preview = st.slider("Tampilkan berapa baris?", 5, 50, 10, key="preview_rows")
    st.dataframe(df_raw.head(n_preview), use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        st.subheader(" Info Dataset")
        st.dataframe(pd.DataFrame({
            "Kolom":    df_raw.columns,
            "Dtype":    df_raw.dtypes.astype(str).values,
            "Non-Null": df_raw.notnull().sum().values,
            "Null":     df_raw.isnull().sum().values,
        }), use_container_width=True, hide_index=True)
    with cb:
        st.subheader(" Statistik Deskriptif")
        st.dataframe(df_raw.describe().T.round(3), use_container_width=True)

    st.divider()

    #  Missing Values 
    st.subheader(" Analisis Missing Values")
    missing_series = df_raw.isnull().sum()
    missing_f = missing_series[missing_series > 0]

    if missing_f.empty:
        st.success(" Tidak terdapat missing value pada dataset!")
    else:
        cm1, cm2 = st.columns([1, 2])
        with cm1:
            mv_df = missing_f.reset_index()
            mv_df.columns = ["Kolom", "Jumlah"]
            mv_df["%"] = (mv_df["Jumlah"] / len(df_raw) * 100).round(2)
            st.dataframe(mv_df, use_container_width=True, hide_index=True)
        with cm2:
            fig, ax = plt.subplots(figsize=(8, 4))
            missing_f.plot(kind="bar", ax=ax, color="salmon", edgecolor="white")
            ax.set_title("Missing Value per Kolom", fontsize=13)
            ax.set_ylabel("Jumlah Missing")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.divider()

    #  Target Distribution 
    st.subheader(" Distribusi Target (Churn)")
    ct1, ct2 = st.columns([1, 2])
    with ct1:
        vc = df_raw["churn"].value_counts().reset_index()
        vc.columns = ["Churn", "Jumlah"]
        vc["%"] = (vc["Jumlah"] / len(df_raw) * 100).round(2)
        st.dataframe(vc, use_container_width=True, hide_index=True)
        st.warning(" Terdapat ketidakseimbangan kelas (class imbalance)")
    with ct2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.countplot(x="churn", data=df_raw, palette="Set2", ax=axes[0])
        axes[0].set_title("Count Plot Churn")
        vc2 = df_raw["churn"].value_counts()
        axes[1].pie(vc2, labels=["No Churn", "Churn"], autopct="%1.1f%%",
                    colors=["#66b3ff", "#ff9999"], startangle=90)
        axes[1].set_title("Proporsi Churn")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    #  Numeric Distributions 
    st.subheader(" Distribusi Fitur Numerik")
    num_plot = [c for c in df_raw.select_dtypes(include=np.number).columns
                if c not in ("churn", "customer_id")]
    if num_plot:
        fig, axes = safe_subplots(len(num_plot), 3, fw=5, fh=3)
        for i, col in enumerate(num_plot):
            df_raw[col].hist(ax=axes[i], bins=25, color="steelblue", edgecolor="white")
            axes[i].set_title(col, fontsize=9)
        for j in range(len(num_plot), len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Distribusi Fitur Numerik", fontsize=14, y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    #  Categorical Distributions 
    st.subheader(" Distribusi Fitur Kategorik")
    cat_plot = [c for c in df_raw.select_dtypes(include="object").columns
                if "date" not in c.lower()]
    if not cat_plot:
        st.info("i Tidak ada fitur kategorik (non-date) yang ditemukan.")
    else:
        fig, axes = safe_subplots(len(cat_plot), 3, fw=5, fh=4)
        for i, col in enumerate(cat_plot):
            sns.countplot(x=df_raw[col], ax=axes[i], palette="Set2")
            axes[i].set_title(col, fontsize=9)
            axes[i].tick_params(axis="x", rotation=45)
        for j in range(len(cat_plot), len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    #  Outlier Detection 
    st.subheader(" Deteksi Outlier (Boxplot)")
    if num_plot:
        fig, axes = safe_subplots(len(num_plot), 3, fw=5, fh=3)
        for i, col in enumerate(num_plot):
            sns.boxplot(x=df_raw[col], ax=axes[i], color="lightcoral")
            axes[i].set_title(col, fontsize=9)
        for j in range(len(num_plot), len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Boxplot Deteksi Outlier", fontsize=14, y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.info("i Terdapat outlier yang akan ditangani pada tahap preprocessing (IQR method).")

    st.divider()

    #  Correlation Heatmap 
    st.subheader(" Heatmap Korelasi Fitur Numerik")
    corr_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    if len(corr_cols) > 1:
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(
            df_raw[corr_cols].corr(),
            annot=True, fmt=".2f", cmap="coolwarm",
            annot_kws={"size": 7}, linewidths=0.5, ax=ax,
        )
        ax.set_title("Heatmap Korelasi Fitur Numerik", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# 
# TAB 2 - DIRECT MODELING
# 
with tab2:
    st.header("Direct Modeling (Tanpa Preprocessing)")
    st.info("""
**Alur modeling langsung:**
- Label Encoding untuk fitur kategorik
- Missing values diisi dengan median
- Train-test split 80:20 (stratified)
- **Model:** Logistic Regression - Random Forest - Voting Classifier (LR + KNN + SVM)
""")

    if st.button(" Jalankan Direct Modeling", type="primary", key="btn_direct"):
        with st.spinner("Mempersiapkan data..."):
            df_dm = df_raw.copy()

            # Drop date & id columns (can't encode without proper handling)
            drop_dm = [c for c in df_dm.columns
                       if c in ["customer_id"] or "date" in c.lower()]
            df_dm.drop(columns=[c for c in drop_dm if c in df_dm.columns], inplace=True)

            # Fill numeric NaN
            num_dm = df_dm.select_dtypes(include=np.number).columns.difference(["churn"])
            df_dm[num_dm] = df_dm[num_dm].fillna(df_dm[num_dm].median())

            # Label encode categoricals
            le_dm = LabelEncoder()
            for col in df_dm.select_dtypes(include="object").columns:
                df_dm[col] = le_dm.fit_transform(df_dm[col].astype(str))

        X_dm = df_dm.drop("churn", axis=1)
        y_dm = df_dm["churn"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_dm, y_dm, test_size=0.2, random_state=42, stratify=y_dm
        )

        st.subheader(" Distribusi Data Split")
        sc1, sc2 = st.columns(2)
        with sc1:
            td = y_tr.value_counts().reset_index()
            td.columns = ["Kelas", "Jumlah"]
            st.markdown("**Train Set**")
            st.dataframe(td, use_container_width=True, hide_index=True)
        with sc2:
            td2 = y_te.value_counts().reset_index()
            td2.columns = ["Kelas", "Jumlah"]
            st.markdown("**Test Set**")
            st.dataframe(td2, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader(" Hasil Evaluasi Model")
        train_and_show(build_models(), X_tr, y_tr, X_te, y_te)


# 
# TAB 3 - PREPROCESSING + MODELING
# 
with tab3:
    st.header("Modeling dengan Preprocessing")
    st.info("""
**Pipeline preprocessing yang diterapkan:**
1.  **Missing value handling** -> imputasi median untuk fitur numerik
2.  **Outlier removal** -> IQR method pada kolom `age`
3.  **Label Encoding** -> fitur kategorik
4.  **Standard Scaling** -> normalisasi fitur numerik
5.  **SMOTE** -> menangani class imbalance pada data training
""")

    if st.button(" Jalankan Preprocessing + Modeling", type="primary", key="btn_prep"):
        steps = st.container()

        with steps:
            df_prep = preprocess_base(df_raw)
            st.success(f" Preprocessing selesai. Shape data: {df_prep.shape}")

        X_pp = df_prep.drop("churn", axis=1)
        y_pp = df_prep["churn"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_pp, y_pp, test_size=0.2, random_state=42
        )

        scaler_pp = StandardScaler()
        X_tr_sc = scaler_pp.fit_transform(X_tr)
        X_te_sc = scaler_pp.transform(X_te)
        st.success(" StandardScaler diterapkan.")

        smote = SMOTE(random_state=42)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr_sc, y_tr)
        st.success(
            f" SMOTE selesai. Distribusi setelah resampling: "
            f"{pd.Series(y_tr_res).value_counts().to_dict()}"
        )

        st.divider()
        st.subheader(" Hasil Evaluasi Model (dengan Preprocessing)")
        train_and_show(build_models(), X_tr_res, y_tr_res, X_te_sc, y_te)


# 
# TAB 4 - FEATURE SELECTION + BEST MODEL
# 
with tab4:
    st.header("Feature Selection & Best Model (GridSearchCV)")
    st.info("""
**Alur pada tahap ini:**
1.  **Feature Importance** via Random Forest
2.  **Top-10 fitur** terpilih
3.  **GridSearchCV** untuk hyperparameter tuning Random Forest
4.  **Evaluasi model terbaik** pada test set
5.  **Download model** (.pkl)
""")

    if st.button(" Jalankan Feature Selection & GridSearchCV", type="primary", key="btn_fs"):

        #  Preprocessing 
        with st.spinner("Melakukan preprocessing..."):
            df_fs = preprocess_base(df_raw)
        st.success(f" Data siap. Shape: {df_fs.shape}")

        X_all = df_fs.drop("churn", axis=1)
        y_all = df_fs["churn"]

        #  Feature Importance 
        st.subheader(" Feature Importance")
        with st.spinner("Menghitung feature importance..."):
            rf_fi = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_fi.fit(X_all, y_all)
            importance = pd.Series(
                rf_fi.feature_importances_, index=X_all.columns
            ).sort_values(ascending=False)

        fi_c1, fi_c2 = st.columns([1, 2])
        with fi_c1:
            fi_df = importance.reset_index()
            fi_df.columns = ["Fitur", "Importance"]
            st.dataframe(fi_df.round(5), use_container_width=True, hide_index=True)
        with fi_c2:
            fig, ax = plt.subplots(figsize=(9, 5))
            importance.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
            ax.set_title("Feature Importance - Random Forest", fontsize=13)
            ax.set_ylabel("Importance Score")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        top_features = importance.head(10).index.tolist()
        st.success(f" Top-10 fitur terpilih: **{', '.join(top_features)}**")
        st.divider()

        #  Prepare selected features 
        X_sel = df_fs[top_features]
        y_sel = df_fs["churn"]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sel, y_sel, test_size=0.2, random_state=42, stratify=y_sel
        )
        scaler_fs = StandardScaler()
        X_tr_sc = scaler_fs.fit_transform(X_tr)
        X_te_sc = scaler_fs.transform(X_te)

        smote_fs = SMOTE(random_state=42)
        X_tr_res, y_tr_res = smote_fs.fit_resample(X_tr_sc, y_tr)

        #  GridSearchCV 
        st.subheader(" Hyperparameter Tuning - GridSearchCV")
        param_grid = {
            "n_estimators":     [100, 200],
            "max_depth":        [None, 10, 20],
            "min_samples_split":[2, 5],
            "min_samples_leaf": [1, 2],
        }
        col_pg1, col_pg2 = st.columns(2)
        with col_pg1:
            st.markdown("**Parameter Grid:**")
            st.json(param_grid)
        with col_pg2:
            st.markdown("**Scoring:** `f1` (cocok untuk data imbalanced)")
            st.markdown("**CV:** 3-fold | **n_jobs:** -1 (semua core)")

        with st.spinner("Menjalankan GridSearchCV... ini mungkin memerlukan beberapa menit "):
            grid = GridSearchCV(
                estimator=RandomForestClassifier(random_state=42),
                param_grid=param_grid,
                cv=3,
                scoring="f1",
                n_jobs=-1,
                verbose=0,
            )
            grid.fit(X_tr_res, y_tr_res)

        st.success(" GridSearchCV selesai!")

        gs1, gs2 = st.columns(2)
        with gs1:
            st.subheader(" Parameter Terbaik")
            st.json(grid.best_params_)
        with gs2:
            st.subheader(" Best F1 Score (CV)")
            st.metric("Best F1", f"{grid.best_score_:.4f}")

        st.divider()

        #  Evaluate best model 
        st.subheader(" Evaluasi Model Terbaik di Test Set")
        best_model = grid.best_estimator_
        best_model.fit(X_tr_res, y_tr_res)
        y_pred = best_model.predict(X_te_sc)

        acc = accuracy_score(y_te, y_pred)
        cm  = confusion_matrix(y_te, y_pred)
        rep = report_to_df(y_te, y_pred)

        bm1, bm2 = st.columns(2)
        with bm1:
            st.metric("Accuracy", f"{acc:.4f}")
            st.dataframe(rep.round(4), use_container_width=True)
        with bm2:
            fig, ax = plt.subplots(figsize=(5, 4))
            plot_conf_matrix(cm, "Confusion Matrix - Best Model", ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.divider()

        #  Download model 
        st.subheader(" Download Model")

        model_buf = io.BytesIO()
        joblib.dump(best_model, model_buf)
        model_buf.seek(0)

        scaler_buf = io.BytesIO()
        joblib.dump(scaler_fs, scaler_buf)
        scaler_buf.seek(0)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                label=" Download Model (random_forest_model.pkl)",
                data=model_buf,
                file_name="random_forest_model.pkl",
                mime="application/octet-stream",
            )
        with dl2:
            st.download_button(
                label=" Download Scaler (scaler.pkl)",
                data=scaler_buf,
                file_name="scaler.pkl",
                mime="application/octet-stream",
            )