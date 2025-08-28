# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# Set page configuration for a wider layout and custom title
st.set_page_config(
    layout="wide",
    page_title="Heart Health Predictor 💖",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://www.streamlit.io/help',
        'Report a bug': "https://github.com/streamlit/streamlit/issues",
        'About': "# Heart Health Predictor"
    }
)

# --- Custom CSS for a unique and beautiful look ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* --- Base Styles --- */
    html, body, [class*="st-emotion-cache"] {
        font-family: 'Inter', sans-serif;
    }

    /* --- App Layout & Backgrounds --- */
    .stApp {
        background: linear-gradient(135deg, #f0f2f6 0%, #e0e5ec 100%) !important;
    }

    /* Main content area styling */
    .main-content {
        background: linear-gradient(135deg, #ffffff 0%, #f0f8ff 100%) !important;
        border-radius: 1.5rem;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.1);
        padding: 3rem;
        border: 1px solid #e0eaf6;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e0e0e0;
        box-shadow: 5px 0 15px rgba(0,0,0,0.08);
        padding-top: 2rem;
    }

    /* --- ROBUST TEXT VISIBILITY FIXES --- */
    /* Set a strong default dark color for all text */
    body, .stApp, .stApp *, [class*="st-emotion-cache"] {
        color: #212529 !important;
    }
    
    /* Fix for dropdown menu items */
    div[data-baseweb="popover"] ul li {
        color: #212529 !important;
        background-color: #ffffff !important;
    }
    div[data-baseweb="popover"] ul li:hover {
        background-color: #f0f2f6 !important;
    }
    
    /* Fix for selected item in selectbox to make it visible */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #e9ecef !important; /* Light grey background for selected item */
        color: #212529 !important;
    }
    
    /* --- Headers (Restored Colors) --- */
    .main-header {
        font-size: 3.8em;
        color: #D32F2F !important;
        text-align: center;
        font-weight: 800;
    }
    .subheader {
        font-size: 1.6em;
        color: #4A55A2 !important;
        text-align: center;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] h2 {
        color: #0D47A1 !important;
        font-size: 2em;
        font-weight: 700;
    }

    /* --- Sidebar Widgets --- */
    section[data-testid="stSidebar"] .stSlider, 
    section[data-testid="stSidebar"] .stSelectbox, 
    section[data-testid="stSidebar"] .stRadio {
        background-color: #f0f0f0 !important;
        border-radius: 0.8rem;
        padding: 0.7em;
        border: 1px solid #d0d0d0 !important;
    }

    /* --- Prediction Button --- */
    .stButton>button {
        background: linear-gradient(45deg, #4CAF50 0%, #66BB6A 100%);
        color: #FFFFFF !important; /* Force pure white text */
        font-weight: bold;
        padding: 1em 2em;
        border-radius: 1rem;
        border: none;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4);
        font-size: 1.2em;
        margin-top: 2rem; /* Adds space above the button */
        width: 100%; /* Makes button full width */
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #388E3C 0%, #4CAF50 100%);
        transform: translateY(-4px);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Model and Data Loading ---
@st.cache_resource
def load_model():
    """Loads the pickled model file."""
    try:
        with open('heart_disease_pred.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Error: Model file 'heart_disease_pred.pkl' not found. Please ensure it's in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_dataframe():
    """Loads the dataset from the URL for min/max values."""
    url = 'https://github.com/ankitmisk/Heart_Disease_Prediction_ML_Model/blob/main/heart.csv?raw=true'
    try:
        df = pd.read_csv(url)
        numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                df[col].fillna(df[col].mean(), inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading dataset from URL: {e}")
        return None

model = load_model()
df = load_dataframe()

# --- Sidebar Content ---
st.sidebar.header('Patient Features')
st.sidebar.image('https://upload.wikimedia.org/wikipedia/ps/1/13/HeartBeat.gif', caption='Heartbeat Monitor', use_container_width=True)

with st.sidebar.expander("Patient Overview (Dummy Data)"):
    st.write("Total Patients: **1,250**")
    st.write("Avg. Age: **54.5 years**")
    st.write("Gender Ratio (M/F): **65% / 35%**")

with st.sidebar.expander("Risk Factors (Dummy Data)"):
    st.write("Avg. Cholesterol: **245 mg/dL**")
    st.write("Avg. Blood Pressure: **128 mmHg**")
    st.write("Patients with Angina: **30%**")

st.sidebar.markdown("---")

# --- User Inputs ---
input_values = {}
if df is not None:
    features_config = {
        'age': {'type': 'slider', 'min': 29, 'max': 77, 'default': 50, 'step': 1, 'label': 'Age'},
        'sex': {'type': 'radio', 'options': {0: 'Female', 1: 'Male'}, 'default': 1, 'label': 'Sex'},
        'cp': {'type': 'selectbox', 'options': {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-Anginal Pain', 3: 'Asymptomatic'}, 'default': 0, 'label': 'Chest Pain Type'},
        'trestbps': {'type': 'slider', 'min': 94, 'max': 200, 'default': 120, 'step': 1, 'label': 'Resting Blood Pressure (trestbps)'},
        'chol': {'type': 'slider', 'min': 126, 'max': 564, 'default': 240, 'step': 1, 'label': 'Cholesterol (chol)'},
        'fbs': {'type': 'radio', 'options': {0: 'False (<120 mg/dl)', 1: 'True (>120 mg/dl)'}, 'default': 0, 'label': 'Fasting Blood Sugar > 120 mg/dl'},
        'restecg': {'type': 'selectbox', 'options': {0: 'Normal', 1: 'ST-T wave abnormality', 2: 'Left ventricular hypertrophy'}, 'default': 0, 'label': 'Resting ECG Results'},
        'thalach': {'type': 'slider', 'min': 71, 'max': 202, 'default': 150, 'step': 1, 'label': 'Max Heart Rate Achieved (thalach)'},
        'exang': {'type': 'radio', 'options': {0: 'No', 1: 'Yes'}, 'default': 0, 'label': 'Exercise Induced Angina'},
        'oldpeak': {'type': 'slider', 'min': 0.0, 'max': 6.2, 'default': 1.0, 'step': 0.1, 'label': 'ST Depression (oldpeak)'},
        'slope': {'type': 'selectbox', 'options': {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}, 'default': 0, 'label': 'Slope of Peak ST Segment'},
        'ca': {'type': 'slider', 'min': 0, 'max': 4, 'default': 0, 'step': 1, 'label': 'Number of Major Vessels (0-3)'},
        'thal': {'type': 'selectbox', 'options': {0: 'Normal', 1: 'Fixed Defect', 2: 'Reversible Defect'}, 'default': 0, 'label': 'Thalassemia'},
    }

    for feature, config in features_config.items():
        label = config.get('label', feature.replace("_", " ").title())
        if config['type'] == 'slider':
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            input_values[feature] = st.sidebar.slider(
                f'**{label}**', min_val, max_val, float(config['default']), float(config['step'])
            )
        elif config['type'] == 'radio':
            input_values[feature] = st.sidebar.radio(
                f'**{label}**', list(config['options'].keys()), format_func=lambda x: config['options'][x],
                index=list(config['options'].keys()).index(config['default'])
            )
        elif config['type'] == 'selectbox':
            input_values[feature] = st.sidebar.selectbox(
                f'**{label}**', list(config['options'].keys()), format_func=lambda x: config['options'][x],
                index=list(config['options'].keys()).index(config['default'])
            )

# --- Main App Layout ---
# The entire main section is now wrapped in a single container with the "main-content" class
st.markdown('<div class="main-content">', unsafe_allow_html=True)

st.markdown('<p class="main-header">❤️ Heart Disease Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Leveraging Machine Learning for Proactive Health Insights</p>', unsafe_allow_html=True)

st.image("https://itdesigners.org/wp-content/uploads/2024/02/heart-1024x576.jpg", 
         caption="Predicting Heart Health", 
         use_container_width=True)

description = """
Heart disease prevention is critical, and data-driven prediction systems can significantly aid in early diagnosis and treatment. Machine Learning offers accurate prediction capabilities, enhancing healthcare outcomes. This project analyzes a heart disease dataset with appropriate preprocessing. Multiple classification algorithms were implemented in Python using Scikit-learn and Keras to predict the presence of heart disease.

**Algorithms Used:**
* Logistic Regression
* K-Nearest Neighbors
"""
st.write(description)

# --- Prediction and Output ---
if model is not None and df is not None:
    final_input_array = np.array([list(input_values.values())])

    if st.button('Predict Heart Disease Likelihood'):
        with st.spinner('Analyzing data...'):
            time.sleep(1.5)
            prediction = model.predict(final_input_array)[0]

        if prediction == 0:
            st.success('✅ Prediction: Low Likelihood of Heart Disease. Keep up the good work!')
        else:
            st.warning('⚠️ Prediction: High Likelihood of Heart Disease. Consider consulting a healthcare professional.')

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center;'>Developed by <b>Dhruv Sharma</b></div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
