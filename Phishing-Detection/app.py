"""
app.py — Phishing URL Detector
"""

import os
import pickle
import html as html_lib
import json
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from urllib.parse import urlsplit, urlunsplit

from lime.lime_tabular import LimeTabularExplainer

from features import extract_features, full_analysis


# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "best_model.pkl")
DATA_PATH = os.path.join(BASE, "dataset.csv")


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.card {
    background: #ffffff;
    border: 1px solid #e6e8eb;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    color: #111827;
}

.stat {
    background: #f6f8fa;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    color: #111827 !important;
    min-height: 100px;
}

.stat .lbl {
    font-size: 0.8rem;
    color: #4b5563 !important;
}

.stat .val {
    font-size: 1.6rem;
    font-weight: 600;
    color: #111827 !important;
}

.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.p-red {
    background: #fdecec;
    color: #c0392b !important;
}

.p-amber {
    background: #fef6e7;
    color: #b9770e !important;
}

.p-green {
    background: #eafaf1;
    color: #1e8449 !important;
}

.p-grey {
    background: #eef0f2;
    color: #566573 !important;
}

.mono {
    font-family: ui-monospace, Menlo, monospace;
    font-size: 0.85rem;
    word-break: break-all;
    color: #111827 !important;
}

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border: 1px solid #e6e8eb;
    border-radius: 12px;
    padding: 0.7rem 1.2rem;
    margin-bottom: 1rem;
    color: #111827;
}

.topbar .brand {
    font-size: 1.15rem;
    font-weight: 600;
    color: #1f2937 !important;
}

.topbar .brand small {
    font-size: 0.72rem;
    font-weight: 400;
    color: #6b7280 !important;
    margin-left: 8px;
}

.topbar .right {
    display: flex;
    align-items: center;
    gap: 14px;
    font-size: 0.8rem;
    color: #6b7280;
}

.topbar .who {
    text-align: right;
    line-height: 1.25;
}

.topbar .who b {
    color: #1f2937 !important;
}

.topbar .mdl {
    background: #eafaf1;
    color: #1e8449 !important;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    white-space: nowrap;
}

header[data-testid="stHeader"] {
    display: none;
}

.block-container {
    padding-top: 1.2rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner="Preparing URL reference data for analysis…")
def load_reference_data(cols):
    """Build a cached, consistently engineered reference matrix and labels."""
    if not os.path.exists(DATA_PATH):
        return None, None
    try:
        raw = pd.read_csv(DATA_PATH, usecols=['url', 'label'])
        raw = raw.dropna(subset=['label']).copy()
        labels = pd.to_numeric(raw['label'], errors='coerce')
        keep = labels.isin([0, 1]) & raw['url'].notna()
        raw = raw.loc[keep].copy()
        y = labels.loc[keep].astype(int).to_numpy()
        rows = [extract_features(u) for u in raw['url'].fillna('').astype(str)]
        frame = pd.DataFrame(rows).reindex(columns=cols, fill_value=0)
        frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0)
        return frame.to_numpy(dtype=float), y
    except Exception:
        return None, None


def load_train(cols):
    """Backward-compatible helper used by LIME."""
    X, _ = load_reference_data(cols)
    return X


# ============================================================
# PILL
# ============================================================

def pill(level):

    cls = {
        "High": "p-red",
        "Elevated": "p-amber",
        "Medium": "p-amber",
        "Low": "p-green",
        "Neutral": "p-green"
    }.get(level, "p-grey")

    return f"<span class='pill {cls}'>{level}</span>"


# ============================================================
# GAUGE
# ============================================================

def gauge(prob):

    import matplotlib.patches as mp

    fig, ax = plt.subplots(figsize=(3.6, 2.1))

    for a0, a1, col in [
        (0, 60, "#1e8449"),
        (60, 120, "#e67e22"),
        (120, 180, "#c0392b")
    ]:

        ax.add_patch(
            mp.Wedge(
                (0, 0),
                1,
                180 - a1,
                180 - a0,
                width=0.34,
                facecolor=col,
                alpha=0.9
            )
        )

    ang = np.pi * (1 - prob)

    ax.plot(
        [0, 0.8 * np.cos(ang)],
        [0, 0.8 * np.sin(ang)],
        lw=3,
        color="#2c3e50"
    )

    ax.add_patch(
        mp.Circle(
            (0, 0),
            0.05,
            color="#2c3e50"
        )
    )

    ax.text(
        0,
        -0.18,
        f"{prob * 100:.0f}%",
        ha="center",
        fontsize=17,
        weight="bold",
        color="#111827"
    )

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.35, 1.15)
    ax.axis("off")
    ax.set_aspect("equal")

    plt.tight_layout()

    return fig


# ============================================================
# SHAP
# ============================================================

def shap_fig(model, X1, cols):

    expl = shap.TreeExplainer(model)

    sv = expl.shap_values(
        X1,
        check_additivity=False
    )

    if isinstance(sv, list):
        sv = sv[1]

    sv = np.array(sv)

    if sv.ndim == 3:
        sv = sv[:, :, 1]

    vals = np.array(sv[0]).ravel()[:len(cols)]

    idx = np.argsort(np.abs(vals))[-10:]

    fig, ax = plt.subplots(figsize=(6, 4))

    colors = [
        "#c0392b" if v > 0 else "#2980b9"
        for v in vals[idx]
    ]

    ax.barh(
        [cols[i] for i in idx],
        vals[idx],
        color=colors
    )

    ax.axvline(
        0,
        color="grey",
        lw=0.8,
        ls="--"
    )

    ax.set_xlabel(
        "← legitimate    SHAP contribution    phishing →"
    )

    ax.set_title(
        "SHAP — why this decision"
    )

    plt.tight_layout()

    return fig


# ============================================================
# LIME
# ============================================================

def lime_fig(model, X1, cols, Xtrain):

    ex = LimeTabularExplainer(
        Xtrain,
        feature_names=cols,
        class_names=["Legitimate", "Phishing"],
        mode="classification",
        discretize_continuous=True
    )

    e = ex.explain_instance(
        X1.values[0],
        model.predict_proba,
        num_features=8
    )

    fig = e.as_pyplot_figure()

    fig.set_size_inches(6, 4)

    plt.title(
        "LIME — local explanation"
    )

    plt.tight_layout()

    return fig


# ============================================================
# ADDITIONAL RESEARCH VISUALIZATIONS
# ============================================================

def _class1_shap_values(model, X):
    """Return class-1 SHAP values and expected value across SHAP API versions."""
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X, check_additivity=False)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        # SHAP newer versions: rows x features x classes
        values = values[:, :, 1] if values.shape[-1] == 2 else values[:, :, 0]
    base = np.asarray(explainer.expected_value)
    if base.ndim:
        base = base[1] if base.size > 1 else base.ravel()[0]
    return values, float(base)


def shap_waterfall_fig(model, X1, cols):
    values, base = _class1_shap_values(model, X1)
    contributions = np.asarray(values[0]).ravel()
    # Keep the largest eight effects and aggregate the rest so the final bar still
    # represents the complete SHAP decomposition rather than a truncated ranking.
    order = np.argsort(np.abs(contributions))[::-1]
    keep = order[:8]
    rest = order[8:]
    items = [(cols[i], float(contributions[i])) for i in keep]
    if len(rest):
        items.append(('Other features (combined)', float(contributions[rest].sum())))
    cumulative = float(base)
    starts, heights, colors, labels = [], [], [], []
    for label, contribution in items:
        starts.append(cumulative)
        heights.append(contribution)
        colors.append('#b23a48' if contribution >= 0 else '#3478a8')
        labels.append(label)
        cumulative += contribution
    fig, ax = plt.subplots(figsize=(8, 5.2))
    x = np.arange(len(items))
    for i, (start_value, height, color) in enumerate(zip(starts, heights, colors)):
        ax.bar(i, height, bottom=start_value, color=color, width=.72)
        ax.plot([i + .36, i + .64], [start_value + height, start_value + height], color='#666666', linewidth=.7)
    ax.bar(len(items), cumulative, color='#4b5563', width=.72, label='Final model output')
    ax.set_xticks(np.arange(len(items) + 1))
    ax.set_xticklabels(labels + ['Final output'], rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Model output scale')
    ax.set_title(f'SHAP waterfall: baseline {base:.3f} → output {cumulative:.3f}')
    ax.axhline(base, color='#555555', linestyle=':', linewidth=.9, label=f'Baseline {base:.3f}')
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    return fig


def feature_profile_fig(url, Xref, yref, cols):
    frame = pd.DataFrame(Xref, columns=cols)
    candidates = ['url_length', 'num_subdomains', 'digit_ratio', 'url_entropy', 'num_suspicious_words', 'has_ip']
    names = [c for c in candidates if c in frame.columns]
    if len(names) < 3:
        names = list(frame.columns[:6])
    current = extract_features(url)
    # Convert unlike units to empirical percentile ranks against the supplied reference data.
    values = []
    legitimate, phishing = [], []
    for name in names:
        col = frame[name].to_numpy(dtype=float)
        values.append(float(np.mean(col <= float(current.get(name, 0)))))
        legitimate.append(float(np.mean(col <= np.median(col[yref == 0]))) if np.any(yref == 0) else 0.5)
        phishing.append(float(np.mean(col <= np.median(col[yref == 1]))) if np.any(yref == 1) else 0.5)
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.4, 5.0), subplot_kw={'polar': True})
    for vals, label, style in [(values, 'Scanned URL', '-o'), (legitimate, 'Legitimate median reference', '--'), (phishing, 'Phishing median reference', '--')]:
        ax.plot(angles, vals + vals[:1], style, linewidth=2, label=label)
        if label == 'Scanned URL':
            ax.fill(angles, vals + vals[:1], alpha=.10)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1); ax.set_yticks([.25, .5, .75, 1.0]); ax.set_yticklabels(['25th', '50th', '75th', '100th'], fontsize=7)
    ax.set_title('URL feature profile (empirical percentiles)', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=7)
    fig.tight_layout()
    return fig


def reference_distribution_fig(url, Xref, yref, cols):
    frame = pd.DataFrame(Xref, columns=cols)
    values = frame['url_length'].to_numpy(dtype=float)
    current = float(extract_features(url).get('url_length', 0))
    # Clip only the visual range to the 99th percentile; original URL length remains marked.
    upper = max(float(np.quantile(values, .99)), current, 20)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, upper, 36)
    for label, color in [(0, '#3478a8'), (1, '#b23a48')]:
        subset = values[yref == label]
        if len(subset):
            ax.hist(np.clip(subset, 0, upper), bins=bins, density=True, alpha=.42,
                    label='Legitimate' if label == 0 else 'Phishing', color=color)
    ax.axvline(current, color='#222222', linestyle='--', linewidth=2,
               label=f'Scanned URL ({int(current)} chars)')
    ax.set_xlabel('URL length (values above 99th percentile clipped for display)')
    ax.set_ylabel('Density'); ax.set_title('Reference URL-length distributions')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2); fig.tight_layout()
    return fig


def risk_indicator_fig(threat):
    reasons = threat.get('reasons', [])
    fig, ax = plt.subplots(figsize=(7, max(2.8, .42 * max(len(reasons), 1))))
    if reasons:
        reasons = sorted(reasons, key=lambda item: item[0])
        labels = [str(reason) for _, reason in reasons]
        scores = [int(points) for points, _ in reasons]
        ax.barh(labels, scores, color='#b9770e')
        ax.set_xlabel('Heuristic points (not ML probability)')
        ax.set_title('Triggered heuristic indicators')
        ax.grid(axis='x', alpha=.2)
    else:
        ax.text(.5, .5, 'No configured heuristic indicators triggered', ha='center', va='center')
        ax.set_axis_off()
    fig.tight_layout()
    return fig


def model_agreement_fig(artifact, X1):
    models = artifact.get('candidate_models', {})
    if not models:
        return None, 'Only the selected model is saved in this artifact. Re-run train.py to save comparison models.'
    rows = []
    for name, candidate in models.items():
        try:
            p = float(candidate.predict_proba(X1)[0, 1])
            threshold = float(artifact.get('threshold', .5))
            rows.append((name, p, 'Phishing' if p >= threshold else 'Legitimate'))
        except Exception:
            continue
    if not rows:
        return None, 'No saved comparison models could score this URL.'
    fig, ax = plt.subplots(figsize=(7, max(3, .55 * len(rows))))
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    ax.barh(labels, vals, color=['#b23a48' if v >= float(artifact.get('threshold', .5)) else '#3478a8' for v in vals])
    ax.axvline(float(artifact.get('threshold', .5)), color='#333333', linestyle='--', label='Selected threshold')
    ax.set_xlim(0, 1); ax.set_xlabel('Predicted phishing probability')
    ax.set_title('Model agreement / probability spread'); ax.legend(fontsize=8); ax.grid(axis='x', alpha=.2)
    for i, row in enumerate(rows):
        ax.text(min(row[1] + .02, .82), i, row[2], va='center', fontsize=8)
    fig.tight_layout()
    return fig, None


def explanation_stability_fig(model, url, cols):
    """Compare local SHAP-vector similarity for controlled path/query variants.

    These variants are different URLs; this is a sensitivity probe, not a security guarantee.
    """
    raw = str(url).strip()
    candidate = raw if '://' in raw else '//' + raw
    try:
        parsed = urlsplit(candidate)
        if not parsed.netloc:
            raise ValueError('No hostname')
    except Exception:
        return None, 'Could not safely parse URL variations.'
    variants = [('Original URL', raw)]
    if parsed.query:
        variants.append(('Query removed', urlunsplit((parsed.scheme, parsed.netloc, parsed.path, '', ''))))
    else:
        variants.append(('Add benign query', urlunsplit((parsed.scheme, parsed.netloc, parsed.path, 'ref=123', ''))))
    path = parsed.path or ''
    new_path = (path.rstrip('/') + '/login') if path else '/login'
    variants.append(('Path + /login', urlunsplit((parsed.scheme, parsed.netloc, new_path, parsed.query, ''))))
    variants.append(('Path + /account', urlunsplit((parsed.scheme, parsed.netloc, (path.rstrip('/') + '/account') if path else '/account', parsed.query, ''))))
    unique = []
    seen = set()
    for label, value in variants:
        if value not in seen:
            unique.append((label, value)); seen.add(value)
    try:
        frames = pd.DataFrame([extract_features(v) for _, v in unique]).reindex(columns=cols, fill_value=0)
        sv, _ = _class1_shap_values(model, frames)
        reference = np.asarray(sv[0], dtype=float)
        ref_norm = np.linalg.norm(reference)
        similarities, probabilities, labels = [], [], []
        for i, (label, _) in enumerate(unique):
            vec = np.asarray(sv[i], dtype=float)
            denom = ref_norm * np.linalg.norm(vec)
            similarity = float(np.dot(reference, vec) / denom) if denom > 0 else (1.0 if np.allclose(reference, vec) else 0.0)
            similarities.append(float(np.clip(similarity, -1, 1)))
            probabilities.append(float(model.predict_proba(frames.iloc[[i]])[0, 1]))
            labels.append(label)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        bars = ax.barh(labels, similarities, color='#537895')
        ax.axvline(1, color='#555555', linestyle='--', linewidth=.8)
        ax.set_xlim(-1, 1.05); ax.set_xlabel('Cosine similarity of SHAP contribution vectors vs original')
        ax.set_title('Explanation sensitivity to URL path/query changes')
        ax.grid(axis='x', alpha=.2)
        for bar, prob in zip(bars, probabilities):
            ax.text(min(bar.get_width() + .03, .78), bar.get_y() + bar.get_height()/2,
                    f'p(phishing)={prob:.2f}', va='center', fontsize=8)
        fig.tight_layout()
        return fig, 'Path/query variants are distinct inputs and may legitimately change the prediction; similarity is descriptive, not a robustness guarantee.'
    except Exception as exc:
        return None, f'Explanation sensitivity could not be calculated: {exc}'


def calibration_with_current_fig(url_probability):
    path = os.path.join(BASE, 'artifacts', 'calibration_points.csv')
    if not os.path.exists(path):
        return None
    try:
        points = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot([0, 1], [0, 1], 'k--', label='Ideal calibration')
        ax.plot(points['mean_predicted_probability'], points['observed_phishing_frequency'], 'o-', label='Held-out reference bins')
        ax.axvline(float(url_probability), color='#b23a48', linestyle=':', label=f'This URL: {url_probability:.2f}')
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted phishing probability', ylabel='Observed phishing frequency', title='Population calibration + current URL')
        ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
        return fig
    except Exception:
        return None


def threshold_tradeoff_fig(threshold):
    path = os.path.join(BASE, 'artifacts', 'threshold_analysis.csv')
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for key in ['precision', 'recall', 'f1']:
            if key in df:
                ax.plot(df['threshold'], df[key], marker='.', label=key.title())
        ax.axvline(float(threshold), color='#b23a48', linestyle='--', label=f'App threshold: {threshold:.2f}')
        ax.set(xlabel='Decision threshold', ylabel='Validation score', ylim=(0, 1.02), title='Threshold trade-off (validation data)')
        ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
        return fig
    except Exception:
        return None


# ============================================================
# EVALUATION DASHBOARD
# ============================================================

def evaluation_dashboard():
    st.title("Model evaluation dashboard")
    metrics_path = os.path.join(BASE, "artifacts", "metrics.json")
    if not os.path.exists(metrics_path):
        st.warning("Evaluation artifacts are missing. Run `python train.py` to train models and generate the graphs.")
        return
    with open(metrics_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    test = report.get("test_metrics", {})
    st.caption("Metrics are generated by train.py on a held-out split. Check the report's caveats before interpreting results.")
    cols = st.columns(4)
    for col, label, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1-score"], ["accuracy", "precision", "recall", "f1"]):
        value = test.get(key)
        col.metric(label, f"{value:.3f}" if isinstance(value, (int, float)) else "N/A")
    st.write(f"**Selected model:** {report.get('selected_model', 'Unknown')}")
    st.write(f"**Evaluation split:** {report.get('split_method', 'Not recorded')}")
    st.warning("Dataset labels have not been independently verified. A domain-grouped split reduces some leakage, but it is not a substitute for a timestamped external test set.")
    graph_names = [
        ("model_comparison.png", "Validation model comparison"),
        ("roc_curves.png", "ROC curves"),
        ("precision_recall_curves.png", "Precision–recall curves"),
        ("confusion_matrix.png", "Confusion matrix"),
        ("calibration_curve.png", "Probability calibration"),
        ("threshold_analysis.png", "Threshold analysis"),
        ("permutation_importance.png", "Permutation feature importance"),
        ("class_distribution.png", "Dataset label distribution"),
        ("adaptive_learning.png", "Adaptive-learning simulation"),
    ]
    for i in range(0, len(graph_names), 2):
        left, right = st.columns(2)
        for col, item in zip([left, right], graph_names[i:i+2]):
            path, title = item
            full = os.path.join(BASE, "artifacts", path)
            with col:
                if os.path.exists(full):
                    st.subheader(title)
                    st.image(full, use_container_width=True)
                elif path == "adaptive_learning.png":
                    st.info("Adaptive-learning graph was not generated for this dataset split.")
    with st.expander("Evaluation caveats and full metrics"):
        st.json({"audit": report.get("audit"), "test_metrics": test, "warnings": report.get("warnings", [])})


# ============================================================
# MAIN
# ============================================================

def main():

    page = st.sidebar.radio("Navigate", ["URL analysis", "Model evaluation"])
    if page == "Model evaluation":
        evaluation_dashboard()
        return

    art = load_model()

    if art:
        model_label = f"Model: {art['model_name']}"
        mdl_class = "mdl"
    else:
        model_label = "Run train.py first"
        mdl_class = "pill p-red"


    # ========================================================
    # TOP BAR
    # ========================================================

    topbar_html = f"""<div class="topbar">
<div class="brand">🛡️ URL Detector <small>Adaptive &amp; Explainable ML</small></div>
<div class="right">
<div class="who"><b>Model evaluation &amp; analysis</b><br>URL string features only</div>
<span class="{mdl_class}">{model_label}</span>
</div>
</div>"""

    st.markdown(
        topbar_html,
        unsafe_allow_html=True
    )


    # ========================================================
    # TITLE
    # ========================================================

    st.markdown(
        "## Phishing URL threat analysis"
    )
    st.caption("The model probability is not a guarantee of maliciousness or safety. Threat score is a separate heuristic.")

    st.caption(
        "String-based analysis only. URLs are never visited."
    )


    # ========================================================
    # URL INPUT
    # ========================================================

    url = st.text_input(
        "URL to analyse",
        placeholder="http://secure-login-paypal.xyz/verify@account"
    )

    go = st.button(
        "🔍 Analyse",
        type="primary"
    )


    if not go or not url.strip():

        st.info(
            "Enter a URL and click Analyse."
        )

        return


    url = url.strip()


    # ========================================================
    # ANALYSIS
    # ========================================================

    a = full_analysis(url)

    ssl = a["ssl"]
    dns = a["dns"]
    geo = a["geo"]
    tok = a["tokens"]
    thr = a["threat"]


    # ========================================================
    # CURRENT URL
    # ========================================================

    url_html = f"""<div class="card">
<div style="font-size:0.8rem;color:#6b7280">Current URL</div>
<div class="mono">{html_lib.escape(url)}</div>
</div>"""

    st.markdown(
        url_html,
        unsafe_allow_html=True
    )


    # ========================================================
    # MODEL
    # ========================================================

    X1 = None
    is_phish = None
    proba = None

    if art:

        model = art["model"]
        cols = art["feature_cols"]

        X1 = pd.DataFrame(
            [extract_features(url)]
        )[cols]

        proba = model.predict_proba(X1)[0]

        threshold = float(art.get("threshold", 0.5))
        is_phish = float(proba[1]) >= threshold


    # ========================================================
    # TOP CARDS
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)


    # --------------------------------------------------------
    # VERDICT
    # --------------------------------------------------------

    with c1:

        if is_phish is None:

            verdict = "—"

        elif is_phish:

            verdict = (
                "<span class='pill p-red'>Phishing</span>"
            )

        else:

            verdict = (
                "<span class='pill p-green'>Legitimate</span>"
            )

        verdict_html = f"""<div class="stat">
<div class="lbl">ML verdict</div>
<div style="margin-top:6px">{verdict}</div>
</div>"""

        st.markdown(
            verdict_html,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # PHISHING PROBABILITY
    # --------------------------------------------------------

    with c2:

        if proba is not None:

            phishing_probability = float(
                proba[1]
            )

            probability_text = (
                f"{phishing_probability * 100:.0f}%"
            )

        else:

            probability_text = "—"

        html = f"""<div class="stat">
<div class="lbl">Phishing probability</div>
<div class="val">{probability_text}</div>
</div>"""

        st.markdown(
            html,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # THREAT SCORE
    # --------------------------------------------------------

    with c3:

        html = f"""<div class="stat">
<div class="lbl">Threat score</div>
<div class="val">{thr['score']}<span style="font-size:0.9rem;color:#6b7280">/100</span></div>
</div>"""

        st.markdown(
            html,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # THREAT LEVEL
    # --------------------------------------------------------

    with c4:

        html = f"""<div class="stat">
<div class="lbl">Threat level</div>
<div style="margin-top:6px">{pill(thr['level'])}</div>
</div>"""

        st.markdown(
            html,
            unsafe_allow_html=True
        )


    # ========================================================
    # GAUGE
    # ========================================================

    if proba is not None:

        g1, g2 = st.columns([1, 2])

        with g1:

            st.pyplot(
                gauge(proba[1])
            )

            plt.close()


        with g2:

            st.markdown(
                "**Threat score breakdown**"
            )

            if thr["reasons"]:

                st.markdown(
                    "\n".join(
                        f"- **+{p}** {m}"
                        for p, m in thr["reasons"]
                    )
                )

            else:

                st.markdown(
                    "- No risk indicators triggered"
                )


    st.markdown("---")


    # ========================================================
    # DETAILED ANALYSIS
    # ========================================================

    st.markdown(
        "#### Detailed analysis"
    )

    d1, d2, d3, d4 = st.columns(4)


    with d1:

        html = f"""<div class="card">
<b>🔒 SSL</b><br>
{ssl['verdict']}<br>
{pill(ssl['risk'])}
</div>"""

        st.markdown(
            html,
            unsafe_allow_html=True
        )


    with d2:

        html = f"""<div class="card">
<b>🌐 DNS</b><br>
{dns['tld']} · {dns['num_subdomains']} sub<br>
{pill(dns['risk'])}
</div>"""

        st.markdown(
            html,
            unsafe_allow_html=True
        )


    with d3:

        html = f"""<div class="card">
<b>📍 Region</b><br>
{geo['inferred_region']}<br>
{pill(geo['risk'])}
</div>"""

        st.markdown(
            html,
            unsafe_allow_html=True
        )


    with d4:

        toks = (
            ", ".join(
                tok["suspicious_tokens"][:3]
            )
            or "none"
        )

        html = f"""<div class="card">
<b>🔑 Tokens</b><br>
{toks}<br>
{pill(tok['risk'])}
</div>"""

        st.markdown(
            html,
            unsafe_allow_html=True
        )


    st.caption("SSL, DNS and region panels are URL-string heuristics only. This application does not query DNS, validate certificates, visit the website or determine server geolocation.")
    st.markdown("---")


    # ========================================================
    # SHAP + LIME
    # ========================================================

    if art and X1 is not None:

        st.markdown(
            "#### Explainability"
        )

        e1, e2 = st.columns(2)


        with e1:

            try:

                st.pyplot(
                    shap_fig(
                        model,
                        X1,
                        cols
                    )
                )

                plt.close()

            except Exception as ex:

                st.warning(
                    f"SHAP unavailable: {ex}"
                )


        with e2:

            Xtr = load_train(cols)

            if Xtr is None:

                st.info(
                    "LIME needs training data."
                )

            else:

                try:

                    st.pyplot(
                        lime_fig(
                            model,
                            X1,
                            cols,
                            Xtr
                        )
                    )

                    plt.close()

                except Exception as ex:

                    st.warning(
                        f"LIME could not render: {ex}"
                    )


        # ====================================================
        # RESEARCH-ORIENTED URL VISUALIZATIONS
        # ====================================================
        st.markdown("#### Research visualizations for this URL")
        Xref, yref = load_reference_data(cols)
        tab_profile, tab_reliability, tab_sensitivity = st.tabs([
            "URL characteristics", "Reliability & risk", "Model agreement & stability"
        ])

        with tab_profile:
            p1, p2 = st.columns(2)
            with p1:
                if Xref is not None and yref is not None:
                    try:
                        st.pyplot(feature_profile_fig(url, Xref, yref, cols), clear_figure=True)
                        st.caption("Empirical percentile profile against the supplied dataset. This describes features; it does not establish maliciousness.")
                    except Exception as ex:
                        st.warning(f"Feature profile unavailable: {ex}")
                else:
                    st.info("Reference dataset features could not be loaded.")
            with p2:
                if Xref is not None and yref is not None:
                    try:
                        st.pyplot(reference_distribution_fig(url, Xref, yref, cols), clear_figure=True)
                        st.caption("Reference distributions are derived from dataset labels, which have not been independently verified.")
                    except Exception as ex:
                        st.warning(f"Reference distribution unavailable: {ex}")
            try:
                st.pyplot(shap_waterfall_fig(model, X1, cols), clear_figure=True)
                st.caption("Local SHAP contributions show how features influence the selected model output; they are not causal evidence.")
            except Exception as ex:
                st.warning(f"SHAP waterfall unavailable: {ex}")

        with tab_reliability:
            p1, p2 = st.columns(2)
            with p1:
                fig = calibration_with_current_fig(float(proba[1])) if proba is not None else None
                if fig is not None:
                    st.pyplot(fig, clear_figure=True)
                    st.caption("Calibration is a population-level evaluation; the vertical line marks this URL's model score.")
                else:
                    st.info("Calibration points are unavailable. Run train.py to generate evaluation artifacts.")
            with p2:
                threshold = float(art.get('threshold', .5))
                fig = threshold_tradeoff_fig(threshold)
                if fig is not None:
                    st.pyplot(fig, clear_figure=True)
                    st.caption("Threshold trade-offs are computed from validation data, not this single URL.")
                else:
                    st.info("Threshold analysis is unavailable. Run train.py to generate evaluation artifacts.")
            try:
                st.pyplot(risk_indicator_fig(thr), clear_figure=True)
                st.caption("This chart shows heuristic points and is intentionally separate from the model probability.")
            except Exception as ex:
                st.warning(f"Risk-indicator chart unavailable: {ex}")

        with tab_sensitivity:
            fig, warning_text = model_agreement_fig(art, X1)
            if fig is not None:
                st.pyplot(fig, clear_figure=True)
                st.caption("Comparison models are lightweight fits on a training-partition sample; this chart is qualitative disagreement only, not a benchmark comparison or calibrated confidence interval.")
            elif warning_text:
                st.info(warning_text)
            fig, warning_text = explanation_stability_fig(model, url, cols)
            if fig is not None:
                st.pyplot(fig, clear_figure=True)
                st.caption(warning_text)
            elif warning_text:
                st.info(warning_text)

        # ====================================================
        # FEATURE TABLE
        # ====================================================

        with st.expander(
            "📊 Full feature table"
        ):

            feats = extract_features(url)

            st.dataframe(
                pd.DataFrame(
                    feats.items(),
                    columns=[
                        "Feature",
                        "Value"
                    ]
                ),
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # FOOTER
    # ========================================================

    st.caption(
        "Explainable phishing URL analysis · URLs are not visited. DNS, certificate and server location are not verified."
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()