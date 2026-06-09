import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import shap

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocess import load_and_clean, get_features

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="AI Network Threat Detector",
    page_icon="🛡️",
    layout="wide"
)

# ─── Load or Train Model ───────────────────────────────────
@st.cache_resource
def get_model():
    sample_path = 'data/sample_traffic.csv'
    df = load_and_clean(sample_path)
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['Label'])
    X, _ = get_features(df)
    y = df['label_encoded']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    feature_names = list(X.columns)
    return model, le, feature_names

# ─── Header ────────────────────────────────────────────────
st.title("🛡️ AI Network Threat Detector")
st.markdown("**AI-powered network intrusion detection using Random Forest on CICIDS2017 dataset**")
st.markdown("---")

# ─── Load Model ────────────────────────────────────────────
with st.spinner("Loading AI model..."):
    model, le, feature_names = get_model()

st.success("✅ Model loaded successfully!")

# ─── Sidebar ───────────────────────────────────────────────

st.sidebar.markdown("""
### 📖 About
This dashboard detects network intrusions using **Random Forest ML** trained on the CICIDS2017 benchmark dataset.

**Attack types detected:**
- 🔴 DDoS (Distributed Denial of Service)

**AI Features:**
- SHAP Explainability
- Confidence scoring


**Dataset:** CICIDS2017 — Canadian Institute for Cybersecurity
""")
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("Upload your network traffic CSV or use sample data.")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_sample = st.sidebar.button("Use Sample Data")


# ─── Load Data ─────────────────────────────────────────────
df = None
if uploaded_file:
    df = load_and_clean(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")
    st.sidebar.info(f"📊 {len(df)} flows loaded")
elif use_sample:
    df = load_and_clean('data/sample_traffic.csv')
    st.sidebar.success("✅ Sample data loaded!")
else:
    df = load_and_clean('data/sample_traffic.csv')
    st.sidebar.info("📂 Using sample data by default")
st.sidebar.markdown("""
---
⚠️ **Upload format:** CSV must contain network flow features exported from CICFlowMeter tool (same format as CICIDS2017 dataset).
""")    
    

# ─── Predict ───────────────────────────────────────────────
if df is not None:
    # align features
    X_input = df.reindex(columns=feature_names, fill_value=0)
    X_input = X_input.select_dtypes(include=[np.number]).fillna(0)
    X_input = X_input.replace([np.inf, -np.inf], 0)

    predictions = model.predict(X_input)
    probabilities = model.predict_proba(X_input)
    pred_labels = le.inverse_transform(predictions)
    confidence = probabilities.max(axis=1) * 100

    df['Prediction'] = pred_labels
    df['Confidence %'] = confidence.round(2)

    # ─── Metrics ───────────────────────────────────────────
    st.subheader("📊 Detection Summary")
    col1, col2, col3, col4 = st.columns(4)

    total = len(df)
    attacks = (df['Prediction'] != 'BENIGN').sum()
    benign = (df['Prediction'] == 'BENIGN').sum()
    avg_conf = confidence.mean()

    col1.metric("Total Flows", total)
    col2.metric("🔴 Attacks Detected", attacks)
    col3.metric("🟢 Normal Traffic", benign)
    col4.metric("Avg Confidence", f"{avg_conf:.1f}%")

    st.markdown("---")
    
    
    # ─── Charts ────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader(" Traffic Breakdown")
        count_df = df['Prediction'].value_counts().reset_index()
        count_df.columns = ['Type', 'Count']
        fig1 = px.pie(count_df, names='Type', values='Count',
                      color_discrete_map={'BENIGN': '#00CC96', 'DDoS': '#EF553B'})
        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        st.subheader("📈 Confidence Distribution")
        fig2 = px.histogram(df, x='Confidence %', color='Prediction',
                            nbins=20, barmode='overlay',
                            color_discrete_map={'BENIGN': '#00CC96', 'DDoS': '#EF553B'})
        st.plotly_chart(fig2, use_container_width=True)

    # ─── Feature Importance ────────────────────────────────
    st.subheader("🧠 Top 10 Features (Explainable AI)")
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)

    fig3 = px.bar(importance_df, x='Importance', y='Feature',
                  orientation='h', color='Importance',
                  color_continuous_scale='Reds')
    fig3.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig3, use_container_width=True)

    # ─── Alert Table ───────────────────────────────────────
    st.subheader("🚨 Suspicious Flows")
    attacks_df = df[df['Prediction'] != 'BENIGN'][
        ['Confidence %', 'Prediction'] +
        [c for c in ['Flow Duration', 'Total Fwd Packets',
                     'Flow Bytes/s', 'Flow Packets/s'] if c in df.columns]
    ].head(50)

    if len(attacks_df) > 0:
        st.dataframe(attacks_df, use_container_width=True)
    else:
        st.success("No attacks detected in this traffic sample!")
        
   
    # ─── Individual Flow Explanation ───────────────────────
    st.markdown("---")
    st.subheader("🔬 Explain a Specific Flow")
    st.markdown("Select a suspicious flow to see exactly WHY the model flagged it as an attack.")

    attack_indices = df[df['Prediction'] != 'BENIGN'].index.tolist()
    
    if len(attack_indices) > 0:
        selected_idx = st.selectbox(
            "Select a suspicious flow to explain:",
            options=attack_indices[:20],
            format_func=lambda x: f"Flow #{x} — {df.loc[x, 'Prediction']} ({df.loc[x, 'Confidence %']:.1f}% confidence)"
        )
        
        flow_clean = X_input.loc[[selected_idx]].reindex(
            columns=feature_names, fill_value=0).fillna(0)
        flow_clean = flow_clean.replace([np.inf, -np.inf], 0)

        explainer = shap.TreeExplainer(model)
        shap_single = explainer.shap_values(flow_clean)

        # handle different SHAP output formats
        if isinstance(shap_single, list):
            # multi-class output — take class 1 (attack class)
            vals = np.array(shap_single[1]).flatten()
        elif hasattr(shap_single, 'values'):
            # newer SHAP Explanation object
            v = shap_single.values
            if v.ndim == 3:
                vals = v[0, :, 1]
            elif v.ndim == 2:
                vals = v[0]
            else:
                vals = v.flatten()
        else:
            vals = np.array(shap_single).flatten()

        # make sure lengths match
        min_len = min(len(feature_names), len(vals))
        single_df = pd.DataFrame({
            'Feature': feature_names[:min_len],
            'SHAP Value': vals[:min_len]
        }).sort_values('SHAP Value', key=abs, ascending=False).head(8)

        single_df['Direction'] = single_df['SHAP Value'].apply(
            lambda x: '🔴 Pushes toward ATTACK' if x > 0 else '🟢 Pushes toward NORMAL'
        )

        fig_single = px.bar(
            single_df, x='SHAP Value', y='Feature',
            orientation='h',
            color='SHAP Value',
            color_continuous_scale='RdYlGn_r',
            title=f'Why Flow #{selected_idx} was flagged as {df.loc[selected_idx, "Prediction"]}'
        )
        fig_single.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_single, use_container_width=True)

        top_reason = single_df.iloc[0]
        st.info(f"🎯 Strongest reason: **{top_reason['Feature']}** — {top_reason['Direction']}")
    else:
        st.success("No attack flows to explain!")  

    # ─── Download ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("⬇️ Export Results")
    csv = df[['Prediction', 'Confidence %']].to_csv(index=False)
    st.download_button(
        label="Download Alert Report as CSV",
        data=csv,
        file_name="threat_report.csv",
        mime="text/csv"
    )


