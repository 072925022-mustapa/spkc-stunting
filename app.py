import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.set_page_config(page_title="SPKC Stunting", layout="wide")
st.title("📊 Sistem Prediksi Klasifikasi Stunting")

# ── GLOBAL STYLE ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.titlesize':   14,
    'axes.titleweight': 'bold',
    'axes.titlepad':    12,
    'axes.labelsize':   12,
    'axes.labelpad':    8,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  10,
    'legend.title_fontsize': 11,
    'figure.dpi':       120,
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor':   '#FAFAFA',
    'axes.grid':        True,
    'grid.color':       '#E8E8E8',
    'grid.linewidth':   0.6,
})
PALETTE_STATUS  = ["#74C476", "#FDAE6B", "#E34A33"]    # hijau lembut - oranye lembut -merah lembut
PALETTE_JK      = ["#74C476", "#9ECAE1"]               # hijau & biru muda
PALETTE_HIST    = "#A8C8E8"
PALETTE_SCATTER = ["#6BAED6", "#FD8D3C", "#E6550D"]
PALETTE_BAR     = ["#5B9BD5", "#ED7D31"]               # model comparison
CMAP_RF         = "Blues"
CMAP_XGB        = "Oranges"
CMAP_HEAT       = "RdYlGn_r"

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[['top', 'right']].set_visible(False)

def add_bar_labels(ax, fmt="{:.3f}", offset=4):
    for p in ax.patches:
        h = p.get_height()
        if h > 0.001:
            ax.annotate(fmt.format(h),
                        (p.get_x() + p.get_width() / 2, h),
                        ha='center', va='bottom', fontsize=9,
                        xytext=(0, offset), textcoords='offset points')

def _kde(series, n=200):
    from scipy.stats import gaussian_kde
    data = series.dropna().values
    kde  = gaussian_kde(data, bw_method='scott')
    x    = np.linspace(data.min(), data.max(), n)
    return x, kde(x)

# ── LOAD & PREPROCESS ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('dataset_stunting_skripsi.csv')

@st.cache_data
def preprocess(df):
    df = df.drop_duplicates()
    df = df.drop(columns=['Unnamed: 10', 'Unnamed: 11', 'Tanggal_Pengukuran', 'TB_U', 'ZS_TB_U'], errors='ignore')
    df['BB_Lahir'] = df['BB_Lahir'].fillna(df['BB_Lahir'].median())
    df['TB_Lahir'] = df['TB_Lahir'].fillna(df['TB_Lahir'].median())
    # Hapus nilai 0 dan outlier ekstrem pada BB_Lahir dan TB_Lahir
    df = df[df['BB_Lahir'] > 0]
    df = df[df['TB_Lahir'] > 0]
    df = df[df['BB_Lahir'] <= 6]    # max berat lahir wajar ~6 kg
    df = df[df['TB_Lahir'] <= 60]   # max tinggi lahir wajar ~60 cm
    return df

@st.cache_resource
def train_models(df):
    df_model = df.copy()
    df_model['JK'] = df_model['JK'].map({'L': 1, 'P': 0, 'Laki-laki': 1, 'Perempuan': 0})
    le = LabelEncoder()
    df_model['Status_Encoded'] = le.fit_transform(df_model['Status'])
    X = df_model.drop(columns=['Status', 'Status_Encoded'])
    y = df_model['Status_Encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train_s, y_train)
    sw = compute_sample_weight('balanced', y_train)
    xgb = XGBClassifier(random_state=42, eval_metric='mlogloss')
    xgb.fit(X_train_s, y_train)
    return rf, xgb, scaler, le, X_test_s, y_test, X.columns.tolist()

df_raw = load_data()
df     = preprocess(df_raw)
rf, xgb, scaler, le, X_test_s, y_test, feature_cols = train_models(df)
rf_pred  = rf.predict(X_test_s);  rf_prob  = rf.predict_proba(X_test_s)
xgb_pred = xgb.predict(X_test_s); xgb_prob = xgb.predict_proba(X_test_s)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📁 Dataset", "📈 EDA", "🤖 Model", "💡 Rekomendasi"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Informasi Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Baris", df_raw.shape[0])
    c2.metric("Jumlah Kolom", df_raw.shape[1])
    c3.metric("Duplikat", df_raw.duplicated().sum())

    st.subheader("5 Baris Pertama")
    st.dataframe(df_raw.head())

    st.subheader("Statistik Deskriptif")
    st.dataframe(df_raw.describe())

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    df2 = df.copy()
    
    STATUS_ORDER = ['Normal', 'Stunting', 'Severely Stunting']
    STATUS_COLOR = {'Normal': '#74C476', 'Stunting': '#E34A33', 'Severely Stunting': '#FDAE6B'}
    JK_ORDER = ['L', 'P']
    JK_COLOR = {'L': '#6BAED6', 'P': '#E74C3C'}    
    NUM_COLS     = ['Umur', 'Berat', 'Tinggi', 'BB_Lahir', 'TB_Lahir']
    NUM_LABELS   = {'Umur': 'Umur (bulan)', 'Berat': 'Berat (kg)',
                    'Tinggi': 'Tinggi (cm)', 'BB_Lahir': 'BB Lahir (kg)', 'TB_Lahir': 'TB Lahir (cm)'}

    # ── UNIVARIATE ─────────────────────────────────────────────────────────────
    st.subheader("1. Univariate Analysis")

    # Bar chart distribusi Status
    status_counts = df2['Status'].value_counts().reindex(STATUS_ORDER, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    bars0 = axes[0].bar(STATUS_ORDER, status_counts.values,
                        color=[STATUS_COLOR[s] for s in STATUS_ORDER], edgecolor='white', width=0.5)
    for b in bars0:
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 10,
                     f"{int(b.get_height())}", ha='center', va='bottom', fontsize=10)
    style_ax(axes[0], 'Distribusi Status Stunting', 'Status', 'Jumlah')

    # Bar chart distribusi JK
    jk_map    = {1: 'Laki-laki', 0: 'Perempuan', 'L': 'Laki-laki', 'P': 'Perempuan'}
    jk_series = df2['JK'].map(jk_map)
    jk_order  = ['Laki-laki', 'Perempuan']
    jk_vals   = jk_series.value_counts().reindex(jk_order, fill_value=0).values.tolist()
    bars1 = axes[1].bar(jk_order, jk_vals, color=PALETTE_JK, edgecolor='white', width=0.5)
    for b in bars1:
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 10,
                     f"{int(b.get_height())}", ha='center', va='bottom', fontsize=10)
    style_ax(axes[1], 'Distribusi Jenis Kelamin', 'Jenis Kelamin', 'Jumlah')

    fig.patch.set_facecolor('#FAFAFA')
    plt.tight_layout(pad=2)
    st.pyplot(fig); plt.close()
    
    # KDE plot semua fitur numerik per Status
    XLIM = {'BB_Lahir': (0, 50), 'TB_Lahir': (0, 150)}
    fig, axes = plt.subplots(1, len(NUM_COLS), figsize=(18, 4))
    for ax, col in zip(axes, NUM_COLS):
        for status in STATUS_ORDER:
            sub = df2[df2['Status'] == status][col].dropna()
            ax.plot(*_kde(sub), color=STATUS_COLOR[status], label=status, linewidth=2)
            ax.fill_between(*_kde(sub), alpha=0.12, color=STATUS_COLOR[status])
        if col in XLIM:
            ax.set_xlim(XLIM[col])
        style_ax(ax, NUM_LABELS[col], NUM_LABELS[col], 'Densitas')
    axes[-1].legend(title='Status', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    fig.patch.set_facecolor('#FAFAFA')
    plt.tight_layout(pad=2)
    st.pyplot(fig); plt.close()

    # ── Missing Values
    mv    = df_raw.isnull().sum().sort_values(ascending=False)
    total = len(df_raw)
    pct   = (mv / total * 100).round(1)
    bar_colors = ["#F4A582" if v > 0 else "#92C5DE" for v in mv.values]
    fig, ax = plt.subplots(figsize=(max(10, len(mv) * 0.85), 4.5))
    bars = ax.bar(mv.index, mv.values, color=bar_colors, edgecolor='white', linewidth=0.8)
    for bar, p in zip(bars, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.002,
                f"{p}%", ha='center', va='bottom', fontsize=9, color='#444')
    style_ax(ax, "Missing Values per Kolom", "Kolom", "Jumlah Missing")
    plt.xticks(rotation=30, ha='right')
    fig.patch.set_facecolor('#FAFAFA')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ── BIVARIATE ──────────────────────────────────────────────────────────────
    st.subheader("2. Bivariate Analysis")

    # Heatmap + Stacked bar dalam 1 baris 2 kolom
    df2['Kelompok_Umur'] = pd.cut(df2['Umur'],
        bins=[0, 6, 12, 24, 36, 48, 60],
        labels=['0-6', '7-12', '13-24', '25-36', '37-48', '49-60'])
    pivot     = df2.groupby(['Kelompok_Umur', 'Status'], observed=True).size().unstack(fill_value=0)
    pivot     = pivot.reindex(columns=[c for c in STATUS_ORDER if c in pivot.columns])
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    df_num    = df2.drop(columns=['Status', 'Status_Encoded', 'Kelompok_Umur'], errors='ignore').copy()
    df_num['JK'] = df_num['JK'].map({'Laki-laki': 1, 'Perempuan': 0, 'L': 1, 'P': 0}).fillna(df_num['JK'])
    df_num    = df_num.select_dtypes(include='number')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap (kiri)
    # Custom colormap: Hijau -> Oranye -> Merah
    custom_cmap = LinearSegmentedColormap.from_list(
        "green_orange_red",
        ["#4CAF50", "#FF9800", "#E53935"]
    )
    sns.heatmap(df_num.corr(), annot=True, cmap=custom_cmap, fmt=".2f", ax=axes[0],
                linewidths=0.5, linecolor='white', annot_kws={"size": 10},
                vmin=-1, vmax=1, square=True)
    style_ax(axes[0], "Heatmap Korelasi Antar Fitur")

    # Stacked bar (kanan)
    bottom = np.zeros(len(pivot_pct))
    for status in pivot_pct.columns:
        bars = axes[1].bar(pivot_pct.index.astype(str), pivot_pct[status],
                           bottom=bottom, color=STATUS_COLOR[status],
                           label=status, edgecolor='white', width=0.6)
        for bar, val in zip(bars, pivot_pct[status]):
            if val > 5:
                axes[1].text(bar.get_x() + bar.get_width()/2,
                             bar.get_y() + bar.get_height()/2,
                             f"{val:.0f}%", ha='center', va='center',
                             fontsize=9, color='white', fontweight='bold')
        bottom += pivot_pct[status].values
    axes[1].set_ylim(0, 105)
    axes[1].legend(title='Status', bbox_to_anchor=(1.01, 1), loc='upper left')
    style_ax(axes[1], 'Proporsi Status Stunting per Kelompok Umur', 'Kelompok Umur (bulan)', 'Persentase (%)')

    fig.patch.set_facecolor('#FAFAFA')
    plt.tight_layout(pad=2)
    st.pyplot(fig); plt.close()

    # Violin plot: Berat & Tinggi per Status
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (col, ylabel) in zip(axes, [('Berat', 'Berat (kg)'), ('Tinggi', 'Tinggi (cm)')]):
        parts = ax.violinplot(
            [df2[df2['Status'] == s][col].dropna() for s in STATUS_ORDER],
            positions=range(len(STATUS_ORDER)), showmedians=True, showextrema=False)
        for i, (pc, status) in enumerate(zip(parts['bodies'], STATUS_ORDER)):
            pc.set_facecolor(STATUS_COLOR[status]); pc.set_alpha(0.7)
        parts['cmedians'].set_color('#333'); parts['cmedians'].set_linewidth(2)
        ax.set_xticks(range(len(STATUS_ORDER))); ax.set_xticklabels(STATUS_ORDER, rotation=10)
        style_ax(ax, f'{ylabel} per Status Stunting', 'Status', ylabel)
    fig.patch.set_facecolor('#FAFAFA')
    plt.tight_layout(pad=2)
    st.pyplot(fig); plt.close()

    # Scatter Umur vs Tinggi dengan regresi per Status
    fig, ax = plt.subplots(figsize=(11, 5))
    for status in STATUS_ORDER:
        sub = df2[df2['Status'] == status]
        ax.scatter(sub['Umur'], sub['Tinggi'], color=STATUS_COLOR[status],
                   alpha=0.35, s=20, edgecolors='none', label=status)
        # regression line
        m, b = np.polyfit(sub['Umur'], sub['Tinggi'], 1)
        x_line = np.linspace(sub['Umur'].min(), sub['Umur'].max(), 100)
        ax.plot(x_line, m * x_line + b, color=STATUS_COLOR[status], linewidth=2)
    ax.legend(title='Status', framealpha=0.7)
    style_ax(ax, 'Hubungan Umur vs Tinggi Badan per Status', 'Umur (bulan)', 'Tinggi (cm)')
    fig.patch.set_facecolor('#FAFAFA')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Stacked bar: proporsi Status per kelompok Umur

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    def get_metrics(y_true, y_pred, y_prob):
        return {
            'Accuracy':  accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted'),
            'Recall':    recall_score(y_true, y_pred, average='weighted'),
            'F1-Score':  f1_score(y_true, y_pred, average='weighted'),
            'AUC':       roc_auc_score(y_true, y_prob, multi_class='ovr'),
        }

    rf_m  = get_metrics(y_test, rf_pred, rf_prob)
    xgb_m = get_metrics(y_test, xgb_pred, xgb_prob)

    st.subheader("Perbandingan Metrik Evaluasi")
    df_m = pd.DataFrame([rf_m, xgb_m], index=['Random Forest', 'XGBoost (Balanced)'])
    st.dataframe(df_m.style.format("{:.4f}").background_gradient(cmap='Blues', axis=1))

    df_melt = df_m.T.reset_index().melt(id_vars='index', var_name='Model', value_name='Score')
    df_melt.rename(columns={'index': 'Metrics'}, inplace=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=df_melt, x='Metrics', y='Score', hue='Model',
                palette=PALETTE_BAR, ax=ax, edgecolor='white')
    
    # Menampilkan label dengan 4 digit desimal
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f'{height:.4f}',
            (p.get_x() + p.get_width() / 2., height),
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    ax.set_ylim(0, 1.18)
    style_ax(ax, "Perbandingan Performa Model: Random Forest vs XGBoost", "Metrik Evaluasi", "Skor")
    ax.legend(title='Model', framealpha=0.7)
    fig.patch.set_facecolor('#FAFAFA')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ── Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax_i, (pred, cmap, title) in enumerate([
        (rf_pred,  CMAP_RF,  "Random Forest"),
        (xgb_pred, CMAP_XGB, "XGBoost (Balanced)"),
    ]):
        sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap=cmap,
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    ax=axes[ax_i], linewidths=0.5, linecolor='white',
                    annot_kws={"size": 11})
        style_ax(axes[ax_i], title, "Prediksi", "Aktual")
    fig.patch.set_facecolor('#FAFAFA')
    plt.tight_layout(pad=2)
    st.pyplot(fig); plt.close()

    # ── Classification Report
    st.subheader("Classification Report")

    from sklearn.metrics import precision_recall_fscore_support

    def plot_classification_report(ax, y_true, y_pred, title):
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(le.classes_)))
        classes = le.classes_
        x = np.arange(len(classes))
        w = 0.25
        ax.bar(x - w,   p, w, label='Precision', color='#5B9BD5', edgecolor='white')
        ax.bar(x,       r, w, label='Recall',    color='#70AD47', edgecolor='white')
        ax.bar(x + w,   f, w, label='F1-Score',  color='#ED7D31', edgecolor='white')
        for i, (pv, rv, fv) in enumerate(zip(p, r, f)):
            for xpos, val in [(i - w, pv), (i, rv), (i + w, fv)]:
                ax.text(xpos, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(classes, rotation=10)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9, framealpha=0.7)
        style_ax(ax, title, "Kelas", "Skor")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_classification_report(axes[0], y_test, rf_pred,  "Classification Report — Random Forest")
    plot_classification_report(axes[1], y_test, xgb_pred, "Classification Report — XGBoost (Balanced)")
    fig.patch.set_facecolor('#FAFAFA')
    plt.tight_layout(pad=2)
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — REKOMENDASI
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("💡 Insight & Rekomendasi Kebijakan")
    
    rf_f1  = f1_score(y_test, rf_pred,  average='weighted')
    xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
    best_name  = "Random Forest" if rf_f1 >= xgb_f1 else "XGBoost (Balanced)"
    best_model = rf  if rf_f1 >= xgb_f1 else xgb
    best_pred  = rf_pred  if rf_f1 >= xgb_f1 else xgb_pred
    best_prob  = rf_prob  if rf_f1 >= xgb_f1 else xgb_prob

    acc = accuracy_score(y_test, best_pred)
    f1  = f1_score(y_test, best_pred, average='weighted')
    auc = roc_auc_score(y_test, best_prob, multi_class='ovr')

    fi   = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top3 = fi.head(3).index.tolist()

    rekomendasi_map = {
        'Umur':     ("⏱️", "Program pemantauan tumbuh kembang rutin sesuai usia balita."),
        'Berat':    ("⚖️", "Pemantauan berat badan berkala di posyandu."),
        'Tinggi':   ("📏", "Pengukuran tinggi badan rutin dan deteksi dini stunting."),
        'BB_Lahir': ("🏥", "Peningkatan layanan kesehatan ibu hamil untuk cegah BBLR."),
        'TB_Lahir': ("🏥", "Pemantauan pertumbuhan sejak lahir di fasilitas kesehatan."),
        'JK':       ("👶", "Program gizi sensitif gender untuk balita laki-laki dan perempuan."),
    }

    st.markdown(f"""
    <div style="background:#f0f7f4;border-left:4px solid #52b788;border-radius:6px;padding:16px 20px;margin-bottom:16px">
    <b>🏆 Model Terbaik: {best_name}</b><br>
    Accuracy: <b>{acc:.2%}</b> &nbsp;·&nbsp; F1-Score: <b>{f1:.4f}</b> &nbsp;·&nbsp; AUC: <b>{auc:.4f}</b>
    <br><br>
    <b>🔑 3 Faktor Paling Berpengaruh:</b> {', '.join(top3)}
    <br><br>
    <b>Rekomendasi Intervensi:</b>
    <ul>
    {''.join(f"<li>{rekomendasi_map.get(f,('📌',f'Perhatikan faktor {f}.'))[0]} {rekomendasi_map.get(f,('📌',f'Perhatikan faktor {f}.'))[1]}</li>" for f in top3)}
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Feature importance chart
    fi_colors = ["#52B788" if i < 3 else "#B7D7C2" for i in range(len(fi))]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(fi.index, fi.values, color=fi_colors, edgecolor='white')
    for i, (idx, val) in enumerate(fi.items()):
        ax.text(i, val + 0.002, f"{val:.3f}", ha='center', va='bottom', fontsize=9, color='#333')
    style_ax(ax, f"Feature Importance — {best_name}", "", "Importance")
    plt.xticks(rotation=30, ha='right')
    fig.patch.set_facecolor('#FAFAFA')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.divider()

    # ── PREDIKSI ──
    st.subheader(f"🔍 Prediksi Status Stunting - Model: {best_name}")
    c1, c2 = st.columns(2)
    with c1:
        jk       = st.selectbox("Jenis Kelamin", ["L (Laki-laki)", "P (Perempuan)"])
        umur     = st.number_input("Umur (bulan)", 0, 60, 24)
        berat    = st.number_input("Berat Badan (kg)", 1.0, 30.0, 10.0, step=0.1)
    with c2:
        tinggi   = st.number_input("Tinggi Badan (cm)", 30.0, 120.0, 80.0, step=0.1)
        bb_lahir = st.number_input("Berat Lahir (kg)", 0.5, 6.0, 3.0, step=0.1)
        tb_lahir = st.number_input("Tinggi Lahir (cm)", 30.0, 60.0, 48.0, step=0.1)

    if st.button("Prediksi", type="primary"):
        jk_val     = 1 if jk.startswith("L") else 0
        input_data = pd.DataFrame([[jk_val, bb_lahir, tb_lahir, umur, berat, tinggi]], columns=feature_cols)
        input_s    = scaler.transform(input_data)
        pred       = best_model.predict(input_s)[0]
        prob       = best_model.predict_proba(input_s)[0]
        label      = le.inverse_transform([pred])[0]
        confidence = prob[pred]

        color = {"Normal": "green", "Stunting": "orange", "Severely Stunting": "red"}.get(label, "blue")
        st.markdown(f"### Hasil: :{color}[**{label}**]")
        st.metric("Confidence", f"{confidence:.2%}")
