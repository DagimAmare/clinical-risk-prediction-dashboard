"""
Streamlit Dashboard for 30-Day Hospital Readmission Risk Prediction

This interactive dashboard allows clinicians to input patient data
and receive real-time readmission risk predictions with clinical recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import ReadmissionPredictor

# Page configuration
st.set_page_config(
    page_title="Hospital Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'readmission_model.pkl')
    try:
        return ReadmissionPredictor.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Header
st.markdown('<div class="main-header">🏥 30-Day Hospital Readmission Risk Predictor</div>',
           unsafe_allow_html=True)
st.markdown("**Powered by Machine Learning | Designed for Clinical Decision Support**")
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Risk Prediction", "📊 Model Insights", "ℹ️ About"])

with tab1:
    st.markdown('<div class="sub-header">Patient Risk Assessment</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Demographics")

        age = st.selectbox("Age Group",
                          options=[5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
                          format_func=lambda x: f"{x-5}-{x+5} years",
                          index=6)

        gender = st.radio("Gender", options=["Male", "Female"])

        st.markdown("### Clinical Measures")

        time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 5)
        num_lab_procedures = st.slider("Number of Lab Procedures", 0, 132, 40)
        num_procedures = st.slider("Number of Procedures", 0, 6, 1)
        num_medications = st.slider("Number of Medications", 1, 81, 15)
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 7)

    with col2:
        st.markdown("### Medical History")

        number_inpatient = st.number_input("Prior Inpatient Visits (past year)",
                                          min_value=0, max_value=21, value=0)
        number_emergency = st.number_input("Prior Emergency Visits (past year)",
                                          min_value=0, max_value=76, value=0)
        number_outpatient = st.number_input("Prior Outpatient Visits (past year)",
                                           min_value=0, max_value=42, value=0)

        st.markdown("### Treatment Information")

        diabetesMed = st.selectbox("Diabetes Medication Prescribed?",
                                   options=["Yes", "No"])
        insulin = st.selectbox("Insulin Prescribed?",
                              options=["No", "Down", "Steady", "Up"])

        emergency_admit = st.checkbox("Emergency Admission?")

    # Calculate engineered features
    elderly = 1 if age >= 65 else 0
    polypharmacy = 1 if num_medications >= 10 else 0
    high_comorbidity = 1 if number_diagnoses >= 7 else 0
    long_stay = 1 if time_in_hospital > 7 else 0
    high_utilization = 1 if (num_procedures > 3 or num_lab_procedures > 50) else 0
    prior_utilization = number_inpatient + number_emergency + number_outpatient
    uncontrolled_diabetes = 1  # Simplified - in real app would use A1C data
    total_med_changes = 0  # Simplified

    # Prediction button
    st.markdown("---")
    if st.button("🔮 Predict Readmission Risk", type="primary", use_container_width=True):
        if model is None:
            st.error("Model not loaded. Please check model file.")
        else:
            # Create feature vector matching training data
            features = pd.DataFrame({
                'age_numeric': [age],
                'elderly': [elderly],
                'gender': [1 if gender == "Male" else 0],
                'time_in_hospital': [time_in_hospital],
                'num_lab_procedures': [num_lab_procedures],
                'num_procedures': [num_procedures],
                'num_medications': [num_medications],
                'number_diagnoses': [number_diagnoses],
                'polypharmacy': [polypharmacy],
                'high_comorbidity': [high_comorbidity],
                'long_stay': [long_stay],
                'high_utilization': [high_utilization],
                'uncontrolled_diabetes': [uncontrolled_diabetes],
                'emergency_admit': [1 if emergency_admit else 0],
                'prior_utilization': [prior_utilization],
                'total_med_changes': [total_med_changes],
                'number_inpatient': [number_inpatient],
                'number_emergency': [number_emergency],
                'number_outpatient': [number_outpatient],
                'diabetesMed': [1 if diabetesMed == "Yes" else 0],
                'insulin': [0 if insulin == "No" else 1]
            })

            # Predict
            risk_prob = model.predict_proba(features)[0]
            risk_class = model.predict(features)[0]

            # Display results
            st.markdown("---")
            st.markdown('<div class="sub-header">📊 Prediction Results</div>',
                       unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Readmission Risk", f"{risk_prob*100:.1f}%")

            with col2:
                risk_category = "HIGH RISK" if risk_class == 1 else "LOW RISK"
                st.metric("Risk Category", risk_category)

            with col3:
                color = "🔴" if risk_class == 1 else "🟢"
                st.metric("Status", color)

            # Risk gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Readmission Risk Score (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if risk_prob > 0.5 else "darkgreen"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

            # Clinical recommendations
            st.markdown('<div class="sub-header">💡 Clinical Recommendations</div>',
                       unsafe_allow_html=True)

            if risk_class == 1:
                st.error("**HIGH RISK patient - Consider the following interventions:**")
                recommendations = [
                    "Schedule follow-up appointment within 7 days of discharge",
                    "Coordinate with care management team for discharge planning",
                    "Ensure comprehensive patient education on disease management",
                    "Consider home health services or telehealth monitoring",
                    "Review and optimize medication regimen",
                    "Address social determinants (transportation, support system)"
                ]
                for rec in recommendations:
                    st.write(f"• {rec}")
            else:
                st.success("**LOW RISK patient - Standard discharge protocol:**")
                recommendations = [
                    "Standard discharge instructions and patient education",
                    "Routine follow-up within 2-4 weeks",
                    "Ensure prescription fulfillment",
                    "Provide emergency contact information"
                ]
                for rec in recommendations:
                    st.write(f"• {rec}")

with tab2:
    st.markdown('<div class="sub-header">📈 Model Performance Metrics</div>',
               unsafe_allow_html=True)

    # Load metrics
    metrics_path = os.path.join(os.path.dirname(__file__), '..', 'visualizations', 'static_plots', 'model_metrics.csv')
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path)

        col1, col2, col3 = st.columns(3)

        with col1:
            roc_auc = metrics[metrics['Metric'] == 'ROC-AUC']['Value'].values[0]
            st.metric("ROC-AUC Score", f"{roc_auc:.4f}")
        with col2:
            f1 = metrics[metrics['Metric'] == 'F1 Score']['Value'].values[0]
            st.metric("F1 Score", f"{f1:.4f}")
        with col3:
            acc = metrics[metrics['Metric'] == 'Accuracy']['Value'].values[0]
            st.metric("Accuracy", f"{acc:.2%}")

    # Display visualizations
    viz_path = os.path.join(os.path.dirname(__file__), '..', 'visualizations', 'static_plots')

    col1, col2 = st.columns(2)

    with col1:
        cm_path = os.path.join(viz_path, 'confusion_matrix.png')
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix", use_container_width=True)

    with col2:
        roc_path = os.path.join(viz_path, 'roc_curve.png')
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC Curve", use_container_width=True)

    fi_path = os.path.join(viz_path, 'feature_importance.png')
    if os.path.exists(fi_path):
        st.image(fi_path, caption="Top Clinical Features", use_container_width=True)

with tab3:
    st.markdown("""
    ### About This Tool

    This dashboard uses machine learning to predict the risk of 30-day hospital readmission
    for diabetic patients. It was developed as part of a health data science project.

    **Clinical Context:**
    - 30-day readmissions are a key quality metric in healthcare
    - Medicare penalizes hospitals with high readmission rates
    - Early identification of high-risk patients enables targeted interventions

    **Model Details:**
    - Algorithm: XGBoost (Gradient Boosting)
    - Training Data: 100,000+ diabetic patient encounters from UCI ML Repository
    - Features: 20+ clinical and demographic variables
    - Validation: 80-20 train-test split with SMOTE class balancing

    **Dataset Source:**
    - Diabetes 130-US Hospitals for Years 1999-2008
    - UCI Machine Learning Repository
    - 101,766 patient encounters across 130 US hospitals

    **Disclaimer:**
    This tool is for educational and research purposes only. It should not replace
    clinical judgment or be used for actual patient care without proper validation.

    ---
    **Developer:** Dagim Amare, MD | MSc Health Data Science Candidate
    **LinkedIn:** [linkedin.com/in/dagim-amare-md](https://linkedin.com/in/dagim-amare-md)
    **GitHub:** [github.com/dagimamare](https://github.com/dagimamare)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    Built with Streamlit | Machine Learning for Healthcare<br>
    © 2025 Dagim Amare | MIT License
</div>
""", unsafe_allow_html=True)
