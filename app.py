import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predicting the Compressive Strength of Concrete",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] {background: #1a1a2e;}
    [data-testid="stSidebar"] * {color: #e0e0e0 !important;}
    .sidebar-title {font-size: 22px; font-weight: 700; color: #4fc3f7 !important; margin-bottom: 8px;}
    div[data-testid="stMetric"] {
        background: #16213e; border: 1px solid #0f3460;
        border-radius: 10px; padding: 16px;
    }
    div[data-testid="stMetric"] label {
        color: #a0c4ff !important; font-size: 13px !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important; font-size: 26px !important; font-weight: 700 !important;
    }
    .stButton > button {border-radius: 8px; font-weight: 600;}
    h1 {color: #4fc3f7;}
    h2, h3 {color: #81d4fa;}
</style>
""", unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────────────────────────────
for key in ["page", "df", "X_train", "X_test", "y_train", "y_test",
            "scaler", "scaler_name", "features", "models", "test_size",
            "random_state", "eval_results"]:
    if key not in st.session_state:
        st.session_state[key] = "Home" if key == "page" else None

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_features(df):
    target = "Concrete compressive strength(MPa, megapascals) "
    cols = list(df.select_dtypes(include=["float64", "int64", "Int64"]).columns)
    if target in cols:
        cols.remove(target)
    return cols, target

def eval_table_html(df_res):
    """Render evaluation table with dark text on light background rows."""
    rows_html = ""
    for i, (_, row) in enumerate(df_res.iterrows()):
        bg = "#ffffff" if i % 2 == 0 else "#f0f4f8"
        cells = ""
        for col in df_res.columns:
            val = row[col]
            display = f"{val:.4f}" if isinstance(val, float) else str(val)
            cells += (
                f'<td style="padding:9px 14px; border:1px solid #cdd5df; '
                f'color:#1a1a2e; font-size:14px; background:{bg};">{display}</td>'
            )
        rows_html += f"<tr>{cells}</tr>"

    headers = "".join(
        f'<th style="background:#0f3460; color:#ffffff; padding:10px 14px; '
        f'border:1px solid #0f3460; font-size:14px; font-weight:700; text-align:left;">{c}</th>'
        for c in df_res.columns
    )
    return f"""
    <div style="overflow-x:auto; margin-top:8px; margin-bottom:12px;">
      <table style="width:100%; border-collapse:collapse; border-radius:8px; overflow:hidden;">
        <thead><tr>{headers}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""

def save_pickle(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def list_saved_pickles():
    return sorted([f for f in os.listdir(".") if f.endswith(".pickle")])

# ── Sidebar Nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🏗️ Concrete Strength</div>', unsafe_allow_html=True)
    st.markdown("**Machine Learning App**")
    st.markdown("---")
    pages = {
        "🏠 Home": "Home",
        "📊 EDA": "EDA",
        "⚙️ Preprocessing": "Preprocessing",
        "🤖 Model Selection & Evaluation": "Model",
        "🔮 Prediction": "Prediction",
    }
    for label, key in pages.items():
        is_active = st.session_state.page == key
        if st.button(label, key=f"nav_{key}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.page = key
            st.rerun()
    st.markdown("---")
    st.markdown('Dataset Source: ')
    st.link_button('UCI Machine Learning Repository', "https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength")
    st.markdown("---")
    st.markdown("""
    ## Made By BINUS University Students:
    - Albertus Adrian W.
    - Darren Star L.
    - Jonathan Raffael
    - Steven Hosea
    - Nicholas Driyadis T.
    """)

# =============================================================================
# PAGE 0 — HOME
# =============================================================================
if st.session_state.page == "Home":
    st.markdown("""
    <style>
        .home-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 20px 40px;
        }
        .home-headline {
            font-size: 28px;
            font-weight: 700;
            color: #4fc3f7;
            line-height: 1.4;
            max-width: 750px;
            margin-bottom: 16px;
        }
        .home-subtext {
            font-size: 17px;
            color: #b0bec5;
            max-width: 700px;
            line-height: 1.7;
            margin-bottom: 32px;
        }
        .home-img {
            border-radius: 20px;
            max-width: 680px;
            width: 100%;
            box-shadow: 0 8px 32px rgba(79,195,247,0.15);
            margin-bottom: 36px;
        }
    </style>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 3, 1])
    with col_c:
        st.markdown('''
        <div class="home-container">
            <div class="home-headline">
                Have you ever think if your concrete compressive strength is strong enough to support the load?
            </div>
            <div class="home-subtext">
                Understanding how strong a concrete mix will be is crucial for building safe and reliable structures.
                Using machine learning, it could help predict your concrete strength based on your materials input.
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.image(
            "homepageImg.png",
            use_container_width=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Get Started", type="primary", use_container_width=True, key="btn_get_started"):
            st.session_state.page = "EDA"
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# PAGE 1 — EDA
# =============================================================================
if st.session_state.page == "EDA":
    st.title("📊 Exploratory Data Analysis")

    if st.session_state.df is None:
        try:
            df = pd.read_excel("Concrete_Data.xls")
            st.session_state.df = df
        except FileNotFoundError:
            st.error("❌ File `Concrete_Data.xls` tidak ditemukan. "
                     "Pastikan file berada di folder yang sama dengan `app.py`.")
            st.stop()

    df = st.session_state.df
    features, target = get_features(df)
    st.session_state.features = features

    st.success(f"✅ Dataset dimuat — {df.shape[0]} baris, {df.shape[1]} kolom")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗂️ Data Overview", "📈 Histogram", "🔥 Heatmap", "📉 Scatter & Box"]
    )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Head (5 baris pertama)")
            st.dataframe(df.head(), use_container_width=True)
        with c2:
            st.subheader("Statistik Deskriptif")
            st.dataframe(df.describe(), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Tipe Data")
            dtype_df = pd.DataFrame(df.dtypes, columns=["Type"]).reset_index()
            dtype_df.columns = ["Column", "Type"]
            st.dataframe(dtype_df, use_container_width=True)
        with c4:
            st.subheader("Missing Values")
            null_df = pd.DataFrame(df.isnull().sum(), columns=["Null Count"]).reset_index()
            null_df.columns = ["Column", "Null Count"]
            null_df["Status"] = null_df["Null Count"].apply(
                lambda x: "✅ OK" if x == 0 else f"⚠️ {x} missing")
            st.dataframe(null_df, use_container_width=True)

    with tab2:
        st.subheader("Histogram")
        sel_hist = st.multiselect("Pilih fitur:", features, default=features[:3], key="hist_sel")
        if sel_hist:
            rows = [sel_hist[i:i+3] for i in range(0, len(sel_hist), 3)]
            for row in rows:
                cols_ui = st.columns(len(row))
                for col_ui, feat in zip(cols_ui, row):
                    with col_ui:
                        fig, ax = plt.subplots(figsize=(5, 3.5))
                        fig.patch.set_facecolor("#0e1117")
                        ax.set_facecolor("#16213e")
                        sns.histplot(df[feat], kde=True, ax=ax, color="#4fc3f7")
                        ax.set_title(feat, color="white", fontsize=9)
                        ax.tick_params(colors="white")
                        ax.xaxis.label.set_color("white")
                        ax.yaxis.label.set_color("white")
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

    with tab3:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(11, 8))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax,
                    annot_kws={"size": 8}, fmt=".2f",
                    linewidths=0.5, linecolor="#1a1a2e")
        ax.tick_params(colors="white")
        plt.xticks(rotation=45, ha="right", color="white", fontsize=8)
        plt.yticks(color="white", fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab4:
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Scatter Plot (vs Target)")
            sel_scatter = st.selectbox("Pilih fitur:", features, key="scatter_sel")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#16213e")
            sns.regplot(x=df[sel_scatter], y=df[target], ax=ax,
                        scatter_kws={"alpha": 0.5, "s": 20, "color": "#4fc3f7"},
                        line_kws={"color": "#ff7043"})
            ax.set_ylabel("Strength (MPa)", color="white")
            ax.set_xlabel(sel_scatter, color="white")
            ax.tick_params(colors="white")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with col_right:
            st.subheader("Box Plot")
            sel_box = st.selectbox("Pilih fitur:", features, key="box_sel")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#16213e")
            sns.boxplot(y=df[sel_box], ax=ax, color="#4fc3f7")
            ax.set_title(sel_box, color="white")
            ax.tick_params(colors="white")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# =============================================================================
# PAGE 2 — PREPROCESSING
# =============================================================================
elif st.session_state.page == "Preprocessing":
    st.title("⚙️ Preprocessing")

    if st.session_state.df is None:
        st.warning("⚠️ Buka halaman EDA terlebih dahulu agar dataset dimuat otomatis.")
        st.stop()

    df = st.session_state.df
    features, target = get_features(df)

    st.subheader("1️⃣  Train-Test Split")
    c1, c2 = st.columns(2)
    with c1:
        test_size = st.slider("Ukuran Test Set (%)", 10, 40, 20, 5)
        st.caption(f"Train: **{100 - test_size}%** | Test: **{test_size}%**")
    with c2:
        random_state = st.number_input("Random State", 0, 999, 42, 1)

    st.markdown("---")
    st.subheader("2️⃣  Normalisasi / Scaling")
    scaler_option = st.radio(
        "Pilih metode:",
        ["Standard Scaler (Z-score)", "Min-Max Normalization"],
        horizontal=True,
    )

    if st.button("🚀 Jalankan Preprocessing", type="primary", use_container_width=True):
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=int(random_state)
        )

        if "Standard" in scaler_option:
            scaler = StandardScaler()
            scaler_name = "StandardScaler"
        else:
            scaler = MinMaxScaler()
            scaler_name = "MinMaxScaler"

        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        st.session_state.X_train      = X_train_sc
        st.session_state.X_test       = X_test_sc
        st.session_state.y_train      = y_train
        st.session_state.y_test       = y_test
        st.session_state.scaler       = scaler
        st.session_state.scaler_name  = scaler_name
        st.session_state.test_size    = test_size
        st.session_state.random_state = random_state
        st.session_state.models       = None
        st.session_state.eval_results = None

        st.success("✅ Preprocessing selesai!")

        ca, cb, cc = st.columns(3)
        ca.metric("Total Sampel",     len(df))
        cb.metric("Training Samples", len(X_train_sc))
        cc.metric("Testing Samples",  len(X_test_sc))

        st.markdown("---")
        df_scaled      = pd.DataFrame(X_train_sc, columns=features)
        df_test_scaled = pd.DataFrame(X_test_sc,  columns=features)

        st.subheader(f"📋 Hasil {scaler_name} — Training Set (10 baris pertama)")
        st.dataframe(df_scaled.head(10).style.background_gradient(cmap="Blues"),
                     use_container_width=True)

        st.subheader(f"📋 Hasil {scaler_name} — Test Set (10 baris pertama)")
        st.dataframe(df_test_scaled.head(10).style.background_gradient(cmap="Oranges"),
                     use_container_width=True)

        st.subheader("📊 Statistik Sebelum vs Sesudah Scaling (Training)")
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("**Sebelum Scaling**")
            st.dataframe(X_train.describe().style.format("{:.4f}"), use_container_width=True)
        with col_b:
            st.caption(f"**Setelah {scaler_name}**")
            st.dataframe(df_scaled.describe().style.format("{:.4f}"), use_container_width=True)

    elif st.session_state.X_train is not None:
        st.info(
            f"ℹ️ Preprocessing sudah dijalankan — "
            f"Test size: **{st.session_state.test_size}%**, "
            f"Scaler: **{st.session_state.scaler_name}**. "
            "Ubah setting lalu klik tombol untuk jalankan ulang."
        )


# =============================================================================
# PAGE 3 — MODEL SELECTION & EVALUATION
# =============================================================================
elif st.session_state.page == "Model":
    st.title("🤖 Model Selection & Evaluation")

    if st.session_state.X_train is None:
        st.warning("⚠️ Lakukan Preprocessing terlebih dahulu.")
        st.stop()

    X_train = st.session_state.X_train
    X_test  = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test  = st.session_state.y_test

    MODEL_OPTIONS = {
        "Linear Regression":               "lr",
        "Random Forest Regressor":         "rf",
        "Support Vector Regression (SVR)": "svr",
    }

    st.subheader("1️⃣  Pilih Model")
    selected_models = st.multiselect(
        "Pilih satu atau lebih model untuk dilatih:",
        list(MODEL_OPTIONS.keys()),
        default=["Linear Regression"],
    )

    if st.button("🏋️ Train & Evaluate", type="primary", use_container_width=True):
        if not selected_models:
            st.error("Pilih minimal satu model.")
            st.stop()

        results        = {}
        trained_models = {}
        progress = st.progress(0, "Melatih model...")

        for i, name in enumerate(selected_models):
            progress.progress(i / len(selected_models), f"Melatih {name}...")
            k = MODEL_OPTIONS[name]
            if k == "lr":
                model = LinearRegression()
            elif k == "rf":
                model = RandomForestRegressor(random_state=42)
            else:
                model = SVR()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results[name] = {
                "R² Score": r2_score(y_test, y_pred),
                "MAE":      mean_absolute_error(y_test, y_pred),
                "MSE":      mean_squared_error(y_test, y_pred),
                "RMSE":     np.sqrt(mean_squared_error(y_test, y_pred)),
            }
            trained_models[name] = (model, y_pred)

        progress.progress(1.0, "✅ Selesai!")
        st.session_state.models       = trained_models
        st.session_state.eval_results = results

    # ── Evaluation results (persists across reruns) ──
    if st.session_state.eval_results:
        results        = st.session_state.eval_results
        trained_models = st.session_state.models

        st.markdown("---")
        st.subheader("📊 Hasil Evaluasi")

        # Plain table — dark text on white/light-grey alternating rows, no highlight
        df_results = pd.DataFrame(results).T.reset_index()
        df_results.columns = ["Model", "R² Score", "MAE", "MSE", "RMSE"]
        st.markdown(eval_table_html(df_results), unsafe_allow_html=True)

        # Per-model metric cards + scatter plot
        for name, (model, y_pred) in trained_models.items():
            st.markdown(f"### 📌 {name}")
            m = results[name]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R² Score", f"{m['R² Score']:.4f}")
            c2.metric("MAE",      f"{m['MAE']:.4f}")
            c3.metric("MSE",      f"{m['MSE']:.4f}")
            c4.metric("RMSE",     f"{m['RMSE']:.4f}")

            fig, ax = plt.subplots(figsize=(7, 3.5))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#16213e")
            ax.scatter(y_test, y_pred, alpha=0.5, s=20, color="#4fc3f7", label="Predictions")
            mn = min(float(y_test.min()), float(y_pred.min()))
            mx = max(float(y_test.max()), float(y_pred.max()))
            ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
            ax.set_xlabel("Actual",    color="white")
            ax.set_ylabel("Predicted", color="white")
            ax.set_title(f"Actual vs Predicted — {name}", color="white")
            ax.tick_params(colors="white")
            ax.legend(facecolor="#16213e", labelcolor="white")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("---")

        # ── Save Model & Scaler to disk ──
        st.subheader("💾 Simpan Model & Scaler")
        col_save1, col_save2 = st.columns(2)

        with col_save1:
            save_model_choice = st.selectbox(
                "Pilih model yang ingin disimpan:",
                list(trained_models.keys()),
                key="save_model_sel",
            )
            model_filename = st.text_input(
                "Nama file model (tanpa ekstensi):",
                value=save_model_choice.replace(" ", "_"),
                key="model_fname",
            )
            if st.button("💾 Simpan Model", type="primary", key="btn_save_model"):
                fpath = f"{model_filename}.pickle"
                save_pickle(trained_models[save_model_choice][0], fpath)
                st.success(f"✅ Tersimpan: `{fpath}`")

        with col_save2:
            if st.session_state.scaler is not None:
                scaler_filename = st.text_input(
                    "Nama file scaler (tanpa ekstensi):",
                    value=st.session_state.scaler_name,
                    key="scaler_fname",
                )
                if st.button("💾 Simpan Scaler", type="primary", key="btn_save_scaler"):
                    fpath = f"{scaler_filename}.pickle"
                    save_pickle(st.session_state.scaler, fpath)
                    st.success(f"✅ Tersimpan: `{fpath}`")


# =============================================================================
# PAGE 4 — PREDICTION
# =============================================================================
elif st.session_state.page == "Prediction":
    st.title("🔮 Prediction")
    st.markdown("Pilih model & scaler yang sudah tersimpan, masukkan nilai fitur, lalu prediksi.")

    # ── Auto-detect .pickle files in current directory ────────────────────────
    all_pickles = list_saved_pickles()

    SCALER_KEYWORDS = ["scaler", "standardscaler", "minmaxscaler", "normalization"]
    scaler_files = [f for f in all_pickles
                    if any(k in f.lower() for k in SCALER_KEYWORDS)]
    model_files  = [f for f in all_pickles if f not in scaler_files]

    st.subheader("1️⃣  Pilih Model")
    if not model_files:
        st.warning("⚠️ Belum ada file model `.pickle` di direktori. "
                   "Latih dan simpan model di halaman **Model Selection & Evaluation** terlebih dahulu.")
        selected_model_file = None
    else:
        selected_model_file = st.selectbox(
            "Pilih file model:",
            model_files,
            key="sel_model_file",
        )

    st.subheader("2️⃣  Pilih Scaler (Opsional)")
    if not scaler_files:
        st.info("ℹ️ Tidak ada file scaler `.pickle` ditemukan. Prediksi tanpa scaling.")
        selected_scaler_file = None
    else:
        scaler_options = ["— Tanpa Scaler —"] + scaler_files
        sel_scaler = st.selectbox(
            "Pilih scaler yang digunakan saat training:",
            scaler_options,
            key="sel_scaler_file",
        )
        selected_scaler_file = None if sel_scaler == "— Tanpa Scaler —" else sel_scaler

    # ── Load from disk ────────────────────────────────────────────────────────
    loaded_model = None
    pred_scaler  = None

    if selected_model_file:
        try:
            with open(selected_model_file, "rb") as f:
                obj = pickle.load(f)
            if not hasattr(obj, "predict"):
                st.error(f"❌ `{selected_model_file}` bukan model yang valid.")
            else:
                loaded_model = obj
                st.success(f"✅ Model aktif: `{selected_model_file}` ({type(obj).__name__})")
        except Exception as e:
            st.error(f"❌ Gagal memuat model: {e}")

    if selected_scaler_file:
        try:
            with open(selected_scaler_file, "rb") as f:
                obj = pickle.load(f)
            if not hasattr(obj, "transform"):
                st.error(f"❌ `{selected_scaler_file}` bukan scaler yang valid.")
            else:
                pred_scaler = obj
                st.success(f"✅ Scaler aktif: `{selected_scaler_file}` ({type(obj).__name__})")
        except Exception as e:
            st.error(f"❌ Gagal memuat scaler: {e}")

    # ── Feature Input ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("3️⃣  Input Nilai Fitur")

    FEATURE_META = {
        "Cement (component 1)(kg in a m^3 mixture)":             (100.0, 540.0,  281.17, "kg/m³"),
        "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": (0.0,   359.4,  73.90,  "kg/m³"),
        "Fly Ash (component 3)(kg in a m^3 mixture)":            (0.0,   200.1,  54.19,  "kg/m³"),
        "Water  (component 4)(kg in a m^3 mixture)":             (121.8, 247.0,  181.57, "kg/m³"),
        "Superplasticizer (component 5)(kg in a m^3 mixture)":   (0.0,   32.2,   6.20,   "kg/m³"),
        "Coarse Aggregate  (component 6)(kg in a m^3 mixture)":  (801.0, 1145.0, 972.92, "kg/m³"),
        "Fine Aggregate (component 7)(kg in a m^3 mixture)":     (594.0, 992.6,  773.58, "kg/m³"),
        "Age (day)":                                              (1.0,   365.0,  45.66,  "hari"),
    }

    feature_keys = st.session_state.features if st.session_state.features else list(FEATURE_META.keys())

    user_input = {}
    col1, col2 = st.columns(2)
    for i, feat in enumerate(feature_keys):
        col = col1 if i % 2 == 0 else col2
        meta = FEATURE_META.get(feat, (0.0, 1000.0, 100.0, ""))
        min_v, max_v, default_v, unit = meta
        with col:
            val = st.number_input(
                f"{feat.split('(')[0].strip()} ({unit})" if unit else feat,
                min_value=float(min_v),
                max_value=float(max_v),
                value=float(default_v),
                step=0.1,
                help=f"Range: {min_v} – {max_v}",
                key=f"input_{i}",
            )
            user_input[feat] = val

    # ── Predict ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("4️⃣  Prediksi")

    if st.button("🔮 Prediksi Kuat Tekan Beton", type="primary", use_container_width=True):
        if loaded_model is None:
            st.error("❌ Pilih model terlebih dahulu.")
        else:
            input_array = np.array([[user_input[f] for f in feature_keys]])
            if pred_scaler is not None:
                input_array = pred_scaler.transform(input_array)

            prediction = loaded_model.predict(input_array)[0]

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #0f3460, #16213e);
                border: 2px solid #4fc3f7; border-radius: 16px;
                padding: 32px; text-align: center; margin: 16px 0;">
                <div style="font-size:14px; color:#a0a0b0; margin-bottom:8px;">
                    Prediksi Kuat Tekan Beton
                </div>
                <div style="font-size:52px; font-weight:800; color:#4fc3f7;">
                    {prediction:.2f}
                </div>
                <div style="font-size:18px; color:#81d4fa; margin-top:4px;">
                    MPa (megapascals)
                </div>
            </div>
            """, unsafe_allow_html=True)

            if prediction < 20:
                cat, clr = "Rendah (Low Strength)",              "#ef5350"
            elif prediction < 40:
                cat, clr = "Normal (Normal Strength)",            "#ffa726"
            elif prediction < 60:
                cat, clr = "Tinggi (High Strength)",              "#66bb6a"
            else:
                cat, clr = "Sangat Tinggi (Very High Strength)",  "#42a5f5"

            st.markdown(f"""
            <div style="text-align:center; margin-top:8px;">
                <span style="background:{clr}22; border:1px solid {clr};
                    border-radius:20px; padding:6px 20px;
                    color:{clr}; font-weight:600; font-size:15px;">
                    {cat}
                </span>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📋 Ringkasan Input"):
                st.dataframe(
                    pd.DataFrame([user_input]).T.rename(columns={0: "Value"}),
                    use_container_width=True,
                )