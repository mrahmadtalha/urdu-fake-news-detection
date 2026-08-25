import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page configuration
st.set_page_config(
    page_title="Urdu Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# Load models and vectorizers
@st.cache_resource
def load_models():
    with open('models/tfidf_urdu.pkl', 'rb') as f:
        tfidf_urdu = pickle.load(f)
    with open('models/tfidf_roman.pkl', 'rb') as f:
        tfidf_roman = pickle.load(f)
    with open('models/best_model_urdu.pkl', 'rb') as f:
        model_urdu = pickle.load(f)
    with open('models/best_model_roman_urdu.pkl', 'rb') as f:
        model_roman = pickle.load(f)
    return tfidf_urdu, tfidf_roman, model_urdu, model_roman

tfidf_urdu, tfidf_roman, model_urdu, model_roman = load_models()

# Load data for dashboard
@st.cache_data
def load_data():
    df_roman = pd.read_csv('data/processed/rufnd_features.csv')
    df_urdu = pd.read_csv('data/processed/urdu_news_features.csv')
    df_urdu['label'] = df_urdu['label'].str.strip()
    df_roman['label'] = df_roman['label'].str.strip()
    return df_roman, df_urdu

df_roman, df_urdu = load_data()
# Navigation
st.title("🔍 Urdu Fake News Detection System")
st.markdown("*Detecting misinformation in Urdu script and Roman Urdu*")

tab1, tab2 = st.tabs(["📊 Dashboard", "🔎 Live Prediction"])

# ─── TAB 1: DASHBOARD ───
with tab1:
    st.header("Project Overview")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", "14,240")
    col2.metric("Urdu Script F1", "0.9217")
    col3.metric("Roman Urdu F1", "0.9537")
    col4.metric("Best Model", "Linear SVM")

    st.divider()

    # Dataset distribution
    st.subheader("Dataset Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        df_urdu['label'].value_counts().plot(
            kind='bar', ax=ax,
            color=['#e74c3c', '#2ecc71'],
            edgecolor='black'
        )
        ax.set_title('Urdu News Dataset')
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=0)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        df_roman['label'].value_counts().plot(
            kind='bar', ax=ax,
            color=['#e74c3c', '#2ecc71'],
            edgecolor='black'
        )
        ax.set_title('RUFND Roman Urdu Dataset')
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=0)
        st.pyplot(fig)
        plt.close()

    st.divider()

    # Model comparison
    st.subheader("Model Performance Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Urdu News Dataset**")
        urdu_results = pd.DataFrame({
            'Model': ['Linear SVM', 'Random Forest', 'Logistic Regression', 'Naive Bayes'],
            'F1 (fake)': [0.9217, 0.9114, 0.9068, 0.8444],
            'Accuracy': [0.9377, 0.9301, 0.9248, 0.8793]
        })
        st.dataframe(urdu_results, hide_index=True)

    with col2:
        st.markdown("**Roman Urdu Dataset**")
        roman_results = pd.DataFrame({
            'Model': ['Linear SVM', 'Logistic Regression', 'Naive Bayes', 'Random Forest'],
            'F1 (fake)': [0.9537, 0.9440, 0.9407, 0.9195],
            'Accuracy': [0.9513, 0.9421, 0.9368, 0.9145]
        })
        st.dataframe(roman_results, hide_index=True)

    st.divider()

    # Cross evaluation
    st.subheader("Cross-Script Evaluation")
    st.markdown("*Testing each model on the other script's data*")

    cross_eval = pd.DataFrame({
        'Model': ['Urdu News Model', 'Roman Urdu Model'],
        'Own Test F1': [0.9217, 0.9537],
        'Cross Dataset F1': [0.6803, 0.0793],
        'Performance Drop': [0.2414, 0.8744]
    })
    st.dataframe(cross_eval, hide_index=True)

    st.warning("""
    **Key Finding:** Models trained on one script fail to generalize to the other. 
    The Roman Urdu model collapses to F1 0.08 on Urdu script data — nearly random. 
    Pakistani fake news detection requires script-agnostic or multilingual approaches.
    """)
    # ─── TAB 2: LIVE PREDICTION ───
with tab2:
    st.header("Live Fake News Detection")
    st.markdown("Enter a news headline or article in Urdu script or Roman Urdu")

    # Script selection
    script = st.radio(
        "Select script:",
        ["Roman Urdu (e.g. Imran Khan ne press conference ki)", 
         "Urdu Script (e.g. عمران خان نے پریس کانفرنس کی)"],
        horizontal=True
    )

    # Text input
    user_input = st.text_area(
        "Enter news text here:",
        height=150,
        placeholder="Type or paste news headline or article..."
    )
    st.caption("💡 Tip: Longer text (2-3 sentences minimum) gives more accurate predictions.")

    if st.button("🔍 Detect", type="primary"):
        if user_input.strip() == "":
            st.error("Please enter some text first.")
        else:
            if "Roman Urdu" in script:
                vector = tfidf_roman.transform([user_input])
                prediction = model_roman.predict(vector)[0]
                model_name = "Roman Urdu Model (Linear SVM)"
            else:
                vector = tfidf_urdu.transform([user_input])
                prediction = model_urdu.predict(vector)[0]
                model_name = "Urdu Script Model (Linear SVM)"

            st.divider()

            if prediction == 'fake':
                st.error(f"🚨 FAKE NEWS DETECTED")
            else:
                st.success(f"✅ REAL NEWS")

            st.caption(f"Model used: {model_name}")

            st.divider()
            st.info("""
            **Note:** This model was trained on Pakistani news datasets. 
            Results should be interpreted as probabilistic indicators, 
            not definitive verdicts. Always verify news from multiple sources.
            """)