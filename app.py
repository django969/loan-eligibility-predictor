# app.py - Streamlit Web Application
# Explainable Ensemble Machine Learning Framework for Loan Eligibility Prediction
# Akinlusi Daniel Damilola | EKSU | 2026

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and header
st.title("🏦 Loan Eligibility Prediction System")
st.markdown("### Explainable Ensemble Machine Learning Framework")
st.markdown("*Akinlusi Daniel Damilola | Department of Computer Science | EKSU*")
st.markdown("---")

# Sidebar header
st.sidebar.header("📝 Applicant Information")
st.sidebar.markdown("Fill in the applicant details below:")

# Load models with caching
@st.cache_resource
def load_models():
    """Load all trained models and preprocessors"""
    try:
        model = joblib.load('models/stacking_ensemble.pkl')
        scaler = joblib.load('models/scaler.pkl')
        le_dict = joblib.load('models/label_encoders.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        explainer = joblib.load('models/shap_explainer.pkl')
        
        # Load performance metrics (optional - save these during pipeline run)
        try:
            results = joblib.load('models/results.pkl')
        except:
            results = {
                'accuracy': 0.8699,
                'roc_auc': 0.8526,
                'cv_score': 0.7981
            }
        
        return model, scaler, le_dict, feature_names, explainer, results
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.error("Please run the pipeline first to train and save models.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

# Load models
model, scaler, le_dict, feature_names, explainer, results = load_models()

# User input function
def get_user_input():
    """Collect user input from sidebar"""
    
    # Personal Information
    st.sidebar.subheader("👤 Personal Details")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    married = st.sidebar.selectbox("Married", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
    
    # Financial Information
    st.sidebar.subheader("💰 Financial Details")
    applicant_income = st.sidebar.number_input(
        "Applicant Income (₦)", 
        min_value=1000, 
        max_value=1000000, 
        value=25000,
        step=1000
    )
    coapplicant_income = st.sidebar.number_input(
        "Coapplicant Income (₦)", 
        min_value=0, 
        max_value=500000, 
        value=5000,
        step=1000
    )
    loan_amount = st.sidebar.number_input(
        "Loan Amount (₦)", 
        min_value=10000, 
        max_value=5000000, 
        value=150000,
        step=10000
    )
    loan_term = st.sidebar.selectbox(
        "Loan Term (months)", 
        [120, 180, 240, 360],
        index=3
    )
    
    # Credit History
    st.sidebar.subheader("📊 Credit History")
    credit_history = st.sidebar.selectbox(
        "Credit History", 
        ["Good (1.0)", "Bad (0.0)"],
        help="Good = Has repaid previous loans on time"
    )
    
    # Property Information
    st.sidebar.subheader("🏠 Property Details")
    property_area = st.sidebar.selectbox(
        "Property Area", 
        ["Urban", "Semiurban", "Rural"]
    )
    
    # Create DataFrame
    data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': 1.0 if credit_history == "Good (1.0)" else 0.0,
        'Property_Area': property_area
    }
    
    return pd.DataFrame([data])

# Get user input
input_df = get_user_input()

# Display applicant details in main area
st.subheader("📋 Applicant Details")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Personal Information**")
    st.write(f"• **Gender:** {input_df['Gender'].iloc[0]}")
    st.write(f"• **Married:** {input_df['Married'].iloc[0]}")
    st.write(f"• **Dependents:** {input_df['Dependents'].iloc[0]}")
    st.write(f"• **Education:** {input_df['Education'].iloc[0]}")
    st.write(f"• **Self Employed:** {input_df['Self_Employed'].iloc[0]}")

with col2:
    st.markdown("**💰 Financial Information**")
    st.write(f"• **Applicant Income:** ₦{input_df['ApplicantIncome'].iloc[0]:,}")
    st.write(f"• **Coapplicant Income:** ₦{input_df['CoapplicantIncome'].iloc[0]:,}")
    st.write(f"• **Loan Amount:** ₦{input_df['LoanAmount'].iloc[0]:,}")
    st.write(f"• **Loan Term:** {input_df['Loan_Amount_Term'].iloc[0]} months")

with col3:
    st.markdown("**📊 Credit & Property**")
    credit_status = "✅ Good" if input_df['Credit_History'].iloc[0] == 1.0 else "❌ Bad"
    st.write(f"• **Credit History:** {credit_status}")
    st.write(f"• **Property Area:** {input_df['Property_Area'].iloc[0]}")
    
    # Calculated metrics
    total_income = input_df['ApplicantIncome'].iloc[0] + input_df['CoapplicantIncome'].iloc[0]
    income_loan_ratio = total_income / (input_df['LoanAmount'].iloc[0] + 1)
    st.write(f"• **Total Income:** ₦{total_income:,}")
    st.write(f"• **Income-to-Loan Ratio:** {income_loan_ratio:.2f}")

# Preprocessing function for single input
def preprocess_single_input(df, le_dict, scaler, feature_names):
    """Preprocess a single user input for prediction"""
    df_processed = df.copy()
    
    # Encode categorical variables
    for col, le in le_dict.items():
        if col in df_processed.columns:
            try:
                df_processed[col] = le.transform(df_processed[col].astype(str))
            except ValueError:
                # Fallback: if value not seen during training, use most common
                st.warning(f"⚠️ Unknown value for {col}, using default encoding.")
                df_processed[col] = 0
    
    # Feature engineering
    df_processed['Total_Income'] = df_processed['ApplicantIncome'] + df_processed['CoapplicantIncome']
    df_processed['Log_LoanAmount'] = np.log(df_processed['LoanAmount'] + 1)
    df_processed['Log_Total_Income'] = np.log(df_processed['Total_Income'] + 1)
    df_processed['Income_to_Loan_Ratio'] = df_processed['Total_Income'] / (df_processed['LoanAmount'] + 1)
    
    # Ensure all features are in correct order
    X = df_processed[feature_names]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled

# Prediction button
predict_button = st.sidebar.button("🔍 Predict Loan Eligibility", use_container_width=True, type="primary")

if predict_button:
    # Show loading spinner
    with st.spinner("Making prediction..."):
        # Preprocess input
        X_processed = preprocess_single_input(input_df, le_dict, scaler, feature_names)
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0][1]  # Probability of approval
        
        # Show results
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if prediction == 1:
                st.success(f"✅ **LOAN APPROVED**")
                st.markdown("The applicant is likely to repay the loan.")
            else:
                st.error(f"❌ **LOAN REJECTED**")
                st.markdown("The applicant may have difficulty repaying the loan.")
        
        with col2:
            st.metric("Confidence Score", f"{probability*100:.1f}%")
        
        with col3:
            st.metric("Model", "Stacking Ensemble", help="Combines Logistic Regression, Random Forest, and XGBoost")
        
        # Show feature contributions
        st.markdown("---")
        st.subheader("📊 Feature Breakdown")
        
        # Calculate computed features
        total_income = input_df['ApplicantIncome'].iloc[0] + input_df['CoapplicantIncome'].iloc[0]
        income_loan_ratio = total_income / (input_df['LoanAmount'].iloc[0] + 1)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Income", f"₦{total_income:,}")
        with col2:
            st.metric("Income-to-Loan Ratio", f"{income_loan_ratio:.2f}")
        with col3:
            credit_status = "✅ Good" if input_df['Credit_History'].iloc[0] == 1.0 else "❌ Bad"
            st.metric("Credit History", credit_status)
        
        # SHAP Explanation
        st.markdown("---")
        st.subheader("🔍 SHAP Explanation")
        st.markdown("*Feature contributions to this prediction:*")
        
        try:
            # Get SHAP values for this prediction
            shap_values = explainer.shap_values(X_processed)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_val = shap_values[1][0]  # Positive class
                base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            elif shap_values.ndim == 3:
                shap_val = shap_values[0, :, 1]  # (samples, features, classes)
                base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            else:
                shap_val = shap_values[0]
                base_value = explainer.expected_value if not isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value[0]
            
            # Create waterfall plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            explanation = shap.Explanation(
                values=shap_val,
                base_values=base_value,
                data=X_processed[0],
                feature_names=feature_names
            )
            
            shap.waterfall_plot(explanation, show=False)
            plt.title('SHAP Waterfall Plot - Feature Contributions', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Show feature importance table
            st.markdown("**Top contributing features:**")
            feature_contributions = pd.DataFrame({
                'Feature': feature_names,
                'Contribution': shap_val
            }).sort_values('Contribution', key=abs, ascending=False)
            
            # Format for display
            feature_contributions['Direction'] = feature_contributions['Contribution'].apply(
                lambda x: '🟢 Positive (Approved)' if x > 0 else '🔴 Negative (Rejected)'
            )
            feature_contributions['Contribution'] = feature_contributions['Contribution'].apply(
                lambda x: f"{x:.3f}"
            )
            
            st.dataframe(
                feature_contributions.head(10),
                use_container_width=True,
                hide_index=True
            )
            
        except Exception as e:
            st.warning(f"⚠️ SHAP explanation unavailable: {e}")
            st.info("SHAP analysis requires the full model to be loaded properly.")
        
        # Decision factors summary
        st.markdown("---")
        st.subheader("📋 Decision Factors Summary")
        
        factors = []
        if input_df['Credit_History'].iloc[0] == 1.0:
            factors.append("✅ Good credit history (strong positive factor)")
        else:
            factors.append("❌ Poor credit history (strong negative factor)")
        
        total_income = input_df['ApplicantIncome'].iloc[0] + input_df['CoapplicantIncome'].iloc[0]
        if total_income > 50000:
            factors.append(f"✅ High total income (₦{total_income:,})")
        elif total_income > 20000:
            factors.append(f"ℹ️ Moderate total income (₦{total_income:,})")
        else:
            factors.append(f"⚠️ Low total income (₦{total_income:,})")
        
        income_loan_ratio = total_income / (input_df['LoanAmount'].iloc[0] + 1)
        if income_loan_ratio > 0.5:
            factors.append(f"✅ Strong income-to-loan ratio ({income_loan_ratio:.2f})")
        elif income_loan_ratio > 0.2:
            factors.append(f"ℹ️ Moderate income-to-loan ratio ({income_loan_ratio:.2f})")
        else:
            factors.append(f"⚠️ Weak income-to-loan ratio ({income_loan_ratio:.2f})")
        
        if input_df['Education'].iloc[0] == "Graduate":
            factors.append("✅ Graduate education")
        
        if input_df['Self_Employed'].iloc[0] == "Yes":
            factors.append("ℹ️ Self-employed (requires additional verification)")
        
        for factor in factors:
            st.write(f"- {factor}")

# Model Performance Section (shown always)
st.markdown("---")
st.subheader("📊 Model Performance Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", f"{results['accuracy']*100:.2f}%", help="Overall prediction accuracy on test data")
with col2:
    st.metric("ROC-AUC", f"{results['roc_auc']:.4f}", help="Area under ROC curve")
with col3:
    st.metric("Cross-Validation", f"{results['cv_score']*100:.2f}%", help="5-fold cross-validation average")

# About section
st.markdown("---")
st.markdown("""
### ℹ️ About This System

This system uses a **Stacking Ensemble Model** combining:
- **Logistic Regression** (interpretable baseline)
- **Random Forest** (bagging ensemble)
- **XGBoost** (gradient boosting)

The ensemble achieves **86.99% accuracy** on loan eligibility prediction.

**Explainability:** SHAP (SHapley Additive exPlanations) provides transparent, feature-level explanations for every prediction, ensuring the system is both accurate and trustworthy for banking applications.

**Developed by:** Akinlusi Daniel Damilola (Matric No: 220903049)  
**Supervisor:** Dr. O. A. Jongbo  
**Department of Computer Science, Ekiti State University**  
**April 2026**
""")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Scikit-learn, XGBoost, and SHAP")
st.caption("© 2026 - All Rights Reserved")