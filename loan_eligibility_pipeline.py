"""
Explainable Ensemble Machine Learning Framework for Loan Eligibility Prediction
Author: Akinlusi Daniel Damilola
Supervisor: Dr. O. A. Jongbo
Department of Computer Science, Ekiti State University
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score, roc_curve)

import xgboost as xgb
import shap

# ============================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================

def load_data():
    df = pd.read_csv('loan_prediction.csv')
    return df
    """Load the Loan Prediction dataset"""
    # Note: You'll need to download from Kaggle or use the provided CSV
    # For this implementation, I'll use the dataset from the project
    # Replace with your actual file path
    
    # Sample data - you should load your actual CSV
    # df = pd.read_csv('loan_prediction_dataset.csv')
    
    # Creating sample data structure (remove this when you have actual data)
    data = {
        'Loan_ID': [f'LP00{i}' for i in range(614)],
        'Gender': np.random.choice(['Male', 'Female'], 614),
        'Married': np.random.choice(['Yes', 'No'], 614, p=[0.65, 0.35]),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], 614, p=[0.45, 0.25, 0.2, 0.1]),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], 614, p=[0.75, 0.25]),
        'Self_Employed': np.random.choice(['Yes', 'No'], 614, p=[0.15, 0.85]),
        'ApplicantIncome': np.random.randint(1000, 50000, 614),
        'CoapplicantIncome': np.random.randint(0, 20000, 614),
        'LoanAmount': np.random.randint(50, 500, 614),
        'Loan_Amount_Term': np.random.choice([120, 180, 240, 360], 614, p=[0.2, 0.2, 0.3, 0.3]),
        'Credit_History': np.random.choice([0, 1], 614, p=[0.15, 0.85]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], 614, p=[0.35, 0.4, 0.25]),
        'Loan_Status': np.random.choice(['Y', 'N'], 614, p=[0.68, 0.32])
    }
    df = pd.DataFrame(data)
    
    print("✅ Dataset loaded successfully!")
    print(f"📊 Shape: {df.shape}")
    print(f"📋 Columns: {df.columns.tolist()}")
    
    return df

# ============================================
# STEP 2: DATA PREPROCESSING
# ============================================

def preprocess_data(df):
    """
    Complete preprocessing pipeline:
    - Handle ALL missing values (numeric → median, categorical → mode)
    - Encode categorical variables
    - Feature engineering
    - Feature scaling
    """
    df_clean = df.copy()
    
    # 1. Handle ALL missing values – go column by column
    print("\n🔍 Handling Missing Values...")
    for col in df_clean.columns:
        if col == 'Loan_ID' or col == 'Loan_Status':
            continue  # skip these
        if df_clean[col].dtype in ['int64', 'float64']:
            # Numeric column → fill with median
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)   # <-- FIXED: assignment
            print(f"  {col}: filled numeric with median ({median_val})")
        else:
            # Categorical column → fill with mode
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)     # <-- FIXED: assignment
            print(f"  {col}: filled categorical with mode ({mode_val})")
    
    # 2. Drop Loan_ID
    if 'Loan_ID' in df_clean.columns:
        df_clean.drop('Loan_ID', axis=1, inplace=True)
        print("  Dropped Loan_ID column")
    
    # 3. Encode Categorical Variables
    print("\n🔤 Encoding Categorical Variables...")
    le_dict = {}
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                        'Self_Employed', 'Property_Area', 'Loan_Status']
    for col in categorical_cols:
        if col in df_clean.columns:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            le_dict[col] = le
            print(f"  {col}: encoded")
    
    # 4. Feature Engineering
    print("\n⚙️ Feature Engineering...")
    df_clean['Total_Income'] = df_clean['ApplicantIncome'] + df_clean['CoapplicantIncome']
    df_clean['Log_LoanAmount'] = np.log(df_clean['LoanAmount'] + 1)
    df_clean['Log_Total_Income'] = np.log(df_clean['Total_Income'] + 1)
    df_clean['Income_to_Loan_Ratio'] = df_clean['Total_Income'] / (df_clean['LoanAmount'] + 1)
    print("  Created Total_Income, Log_LoanAmount, Log_Total_Income, Income_to_Loan_Ratio")
    
    # 5. Split features and target
    X = df_clean.drop('Loan_Status', axis=1)
    y = df_clean['Loan_Status']
    
    # 6. Sanity check – ensure NO NaNs remain
    if X.isnull().any().any():
        print("\n⚠️ WARNING: There are still NaN values in X!")
        print(X.isnull().sum())
        # Fallback: fill any remaining NaNs with 0 (should not happen)
        X.fillna(0, inplace=True)
        print("  Filled remaining NaNs with 0.")
    else:
        print("\n✅ No NaN values remain in X.")
    
    # 7. Scale features
    print("\n📏 Scaling Features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    print("  Features scaled using StandardScaler")
    
    return X_scaled, y, le_dict, scaler
# ============================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# ============================================

def perform_eda(df_original):
    """Perform exploratory data analysis and create visualizations"""
    
    # Create figures directory if it doesn't exist
    import os
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # 3.1 Loan Status Distribution
    plt.figure(figsize=(8, 5))
    status_counts = df_original['Loan_Status'].value_counts()
    status_counts.plot(kind='bar', color=['green', 'red'])
    plt.title('Loan Status Distribution')
    plt.xlabel('Loan Status')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('figures/loan_status_distribution.png')
    plt.close()
    print("✅ Saved: loan_status_distribution.png")
    
    # 3.2 Gender vs Loan Status
    plt.figure(figsize=(8, 5))
    pd.crosstab(df_original['Gender'], df_original['Loan_Status']).plot(kind='bar')
    plt.title('Gender vs Loan Status')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.legend(title='Loan Status')
    plt.tight_layout()
    plt.savefig('figures/gender_vs_loan_status.png')
    plt.close()
    print("✅ Saved: gender_vs_loan_status.png")
    
    # 3.3 Credit History vs Loan Status
    plt.figure(figsize=(8, 5))
    pd.crosstab(df_original['Credit_History'], df_original['Loan_Status']).plot(kind='bar')
    plt.title('Credit History vs Loan Status')
    plt.xlabel('Credit History')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.legend(title='Loan Status')
    plt.tight_layout()
    plt.savefig('figures/credit_history_vs_loan_status.png')
    plt.close()
    print("✅ Saved: credit_history_vs_loan_status.png")
    
    # 3.4 Applicant Income Distribution
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    df_original['ApplicantIncome'].hist(bins=50, color='blue', alpha=0.7)
    plt.title('Applicant Income Distribution')
    plt.xlabel('Income')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    df_original['ApplicantIncome'].hist(bins=50, color='red', alpha=0.7)
    plt.xscale('log')
    plt.title('Applicant Income (Log Scale)')
    plt.xlabel('Income (log)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('figures/income_distribution.png')
    plt.close()
    print("✅ Saved: income_distribution.png")

# ============================================
# STEP 4: MODEL DEVELOPMENT
# ============================================

def develop_models(X, y):
    """
    Develop and train all models:
    1. Logistic Regression
    2. Random Forest
    3. XGBoost
    4. Stacking Ensemble
    """
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 4.1 Logistic Regression
    print("\n" + "="*50)
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_roc = roc_auc_score(y_test, lr_proba)
    print(f"  Accuracy: {lr_acc:.4f}")
    print(f"  ROC-AUC: {lr_roc:.4f}")
    
    # 4.2 Random Forest
    print("\n" + "="*50)
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_roc = roc_auc_score(y_test, rf_proba)
    print(f"  Accuracy: {rf_acc:.4f}")
    print(f"  ROC-AUC: {rf_roc:.4f}")
    
    # 4.3 XGBoost
    print("\n" + "="*50)
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_roc = roc_auc_score(y_test, xgb_proba)
    print(f"  Accuracy: {xgb_acc:.4f}")
    print(f"  ROC-AUC: {xgb_roc:.4f}")
    
    # 4.4 Stacking Ensemble
    print("\n" + "="*50)
    print("Training Stacking Ensemble...")
    
    base_learners = [
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'))
    ]
    
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    
    stacking = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        stack_method='predict_proba'
    )
    
    stacking.fit(X_train, y_train)
    stack_pred = stacking.predict(X_test)
    stack_proba = stacking.predict_proba(X_test)[:, 1]
    stack_acc = accuracy_score(y_test, stack_pred)
    stack_roc = roc_auc_score(y_test, stack_proba)
    print(f"  Accuracy: {stack_acc:.4f}")
    print(f"  ROC-AUC: {stack_roc:.4f}")
    
    # 4.5 Cross-validation
    print("\n" + "="*50)
    print("Cross-Validation Results (5-fold):")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    lr_cv = cross_val_score(lr, X, y, cv=cv, scoring='accuracy').mean()
    rf_cv = cross_val_score(rf, X, y, cv=cv, scoring='accuracy').mean()
    xgb_cv = cross_val_score(xgb_model, X, y, cv=cv, scoring='accuracy').mean()
    stack_cv = cross_val_score(stacking, X, y, cv=cv, scoring='accuracy').mean()
    
    print(f"  Logistic Regression CV: {lr_cv:.4f}")
    print(f"  Random Forest CV: {rf_cv:.4f}")
    print(f"  XGBoost CV: {xgb_cv:.4f}")
    print(f"  Stacking Ensemble CV: {stack_cv:.4f}")
    
    # 4.6 Classification Reports
    print("\n" + "="*50)
    print("Classification Report - Stacking Ensemble:")
    print(classification_report(y_test, stack_pred, target_names=['Rejected', 'Approved']))
    
    # Store all models and results
    models = {
        'logistic_regression': lr,
        'random_forest': rf,
        'xgboost': xgb_model,
        'stacking_ensemble': stacking
    }
    
    results = {
        'lr': {'accuracy': lr_acc, 'roc_auc': lr_roc, 'cv': lr_cv},
        'rf': {'accuracy': rf_acc, 'roc_auc': rf_roc, 'cv': rf_cv},
        'xgb': {'accuracy': xgb_acc, 'roc_auc': xgb_roc, 'cv': xgb_cv},
        'stack': {'accuracy': stack_acc, 'roc_auc': stack_roc, 'cv': stack_cv}
    }
    
    # Predictions for visualization
    predictions = {
        'y_test': y_test,
        'lr_pred': lr_pred, 'lr_proba': lr_proba,
        'rf_pred': rf_pred, 'rf_proba': rf_proba,
        'xgb_pred': xgb_pred, 'xgb_proba': xgb_proba,
        'stack_pred': stack_pred, 'stack_proba': stack_proba
    }
    
    return models, results, predictions, X_train, X_test, y_train, y_test

# ============================================
# STEP 5: VISUALIZATION
# ============================================

def visualize_results(results, predictions, X, y):
    """Create performance comparison visualizations"""
    
    import os
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # 5.1 Model Accuracy Comparison
    plt.figure(figsize=(10, 6))
    model_names = ['Logistic\nRegression', 'Random\nForest', 'XGBoost', 'Stacking\nEnsemble']
    accuracies = [results['lr']['accuracy'], results['rf']['accuracy'], 
                  results['xgb']['accuracy'], results['stack']['accuracy']]
    cv_scores = [results['lr']['cv'], results['rf']['cv'], 
                 results['xgb']['cv'], results['stack']['cv']]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, accuracies, width, label='Test Accuracy', color='steelblue')
    bars2 = plt.bar(x + width/2, cv_scores, width, label='CV Accuracy', color='lightcoral')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, model_names)
    plt.legend()
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=300)
    plt.close()
    print("✅ Saved: model_comparison.png")
    
    # 5.2 ROC Curves
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'green', 'orange', 'red']
    labels = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Stacking Ensemble']
    
    for i, (key, color, label) in enumerate(zip(['lr', 'rf', 'xgb', 'stack'], colors, labels)):
        fpr, tpr, _ = roc_curve(predictions['y_test'], predictions[f'{key}_proba'])
        auc = results[key]['roc_auc']
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/roc_curves.png', dpi=300)
    plt.close()
    print("✅ Saved: roc_curves.png")
    
    # 5.3 Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    models_conf = [('Logistic Regression', predictions['lr_pred']),
                   ('Random Forest', predictions['rf_pred']),
                   ('XGBoost', predictions['xgb_pred']),
                   ('Stacking Ensemble', predictions['stack_pred'])]
    
    for ax, (name, pred) in zip(axes.flat, models_conf):
        cm = confusion_matrix(predictions['y_test'], pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('figures/confusion_matrices.png', dpi=300)
    plt.close()
    print("✅ Saved: confusion_matrices.png")

# ============================================
# STEP 6: SHAP EXPLAINABILITY
# ============================================

def shap_analysis(model, X_train, X_test, feature_names):
    """
    Perform SHAP analysis for model explainability
    """
    print("\n" + "="*50)
    print("🔍 SHAP Explainability Analysis")
    
    # Use Random Forest for SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Handle multi-output (binary classification)
    # shap_values can be list of arrays, or a single array with 3 dims
    if isinstance(shap_values, list):
        # For binary classification, list length 2
        if len(shap_values) == 2:
            shap_values_class1 = shap_values[1]  # positive class
            base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            print("Using SHAP values for positive class (Approved) from list")
        else:
            shap_values_class1 = shap_values[0]
            base_value = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            print("Using SHAP values for first class from list")
    else:
        # If it's a single array, check dimensions
        if shap_values.ndim == 3:
            # Shape: (samples, features, classes) or (samples, classes, features)
            if shap_values.shape[2] == 2:
                shap_values_class1 = shap_values[:, :, 1]  # (samples, features)
                base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                print("Using SHAP values for positive class (Approved) from 3D array (last dim)")
            elif shap_values.shape[1] == 2:
                shap_values_class1 = shap_values[:, 1, :]  # (samples, features)
                base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                print("Using SHAP values for positive class (Approved) from 3D array (second dim)")
            else:
                shap_values_class1 = shap_values[:, :, 0]
                base_value = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                print("Using SHAP values for first class from 3D array")
        else:
            # 2D: (samples, features)
            shap_values_class1 = shap_values
            base_value = explainer.expected_value if not isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value[0]
            print("SHAP values are 2D, using as is.")
    
    # Now shap_values_class1 should be (samples, features)
    print(f"shap_values shape after selection: {np.shape(shap_values_class1)}")
    print(f"Number of features: {len(feature_names)}")
    
    # Compute mean absolute SHAP per feature
    shap_mean = np.abs(shap_values_class1).mean(axis=0)
    if shap_mean.ndim > 1:
        shap_mean = shap_mean.flatten()
    print(f"shap_mean length: {len(shap_mean)}")
    
    # Ensure lengths match
    if len(shap_mean) != len(feature_names):
        print(f"⚠️ Length mismatch! shap_mean={len(shap_mean)}, features={len(feature_names)}")
        min_len = min(len(shap_mean), len(feature_names))
        shap_mean = shap_mean[:min_len]
        feature_names_trimmed = feature_names[:min_len]
    else:
        feature_names_trimmed = feature_names
    
    # Global Feature Importance
    print("\n📊 Global Feature Importance (Mean |SHAP|):")
    feature_importance = pd.DataFrame({
        'feature': feature_names_trimmed,
        'shap_value': shap_mean
    }).sort_values('shap_value', ascending=False)
    print(feature_importance)
    
    # SHAP Summary Plot (use shap_values_class1)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_class1, X_test, feature_names=feature_names, show=False)
    plt.title('SHAP Feature Importance Summary', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: shap_summary.png")
    
    # SHAP Bar Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_class1, X_test, feature_names=feature_names, 
                     plot_type='bar', show=False)
    plt.title('SHAP Feature Importance (Bar)', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/shap_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: shap_bar.png")
    
    # SHAP Dot Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_class1, X_test, feature_names=feature_names, 
                     plot_type='dot', show=False)
    plt.title('SHAP Feature Impact (Dot Plot)', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/shap_dot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: shap_dot.png")
    
    # Individual Prediction Explanation (Waterfall)
    sample_idx = 0
    plt.figure(figsize=(12, 8))
    # Use the selected shap_values_class1 for the sample
    sample_shap = shap_values_class1[sample_idx]  # 1D array of length features
    shap.waterfall_plot(shap.Explanation(values=sample_shap,
                                        base_values=base_value,
                                        data=X_test.iloc[sample_idx],
                                        feature_names=feature_names), show=False)
    plt.title(f'Individual Prediction Explanation (Sample {sample_idx})', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/shap_waterfall.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: shap_waterfall.png")
    
    return shap_values_class1, explainer


# ============================================
# STEP 8: MAIN EXECUTION
# ============================================

def save_models(models, scaler, le_dict, feature_names, shap_explainer):
    """Save all trained models and preprocessing objects"""
    import joblib
    import os
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save stacking ensemble
    joblib.dump(models['stacking_ensemble'], 'models/stacking_ensemble.pkl')
    print("✅ Saved: stacking_ensemble.pkl")
    
    # Save individual models
    joblib.dump(models['logistic_regression'], 'models/logistic_regression.pkl')
    joblib.dump(models['random_forest'], 'models/random_forest.pkl')
    joblib.dump(models['xgboost'], 'models/xgboost.pkl')
    print("✅ Saved: individual models")
    
    # Save preprocessors
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le_dict, 'models/label_encoders.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    print("✅ Saved: preprocessors")
    
    # Save SHAP explainer
    if shap_explainer:
        joblib.dump(shap_explainer, 'models/shap_explainer.pkl')
        print("✅ Saved: shap_explainer.pkl")

def main():
    """Main execution function"""
    print("="*60)
    print("🏦 EXPLAINABLE ENSEMBLE MACHINE LEARNING FRAMEWORK")
    print("Loan Eligibility Prediction in Commercial Banks")
    print("Akinlusi Daniel Damilola | EKSU | 2026")
    print("="*60)
    
    # Step 1: Load data
    print("\n📂 LOADING DATA...")
    df = load_data()
    
    # Step 2: EDA
    print("\n📊 PERFORMING EDA...")
    perform_eda(df)
    
    # Step 3: Preprocess
    print("\n🔄 PREPROCESSING DATA...")
    X, y, le_dict, scaler = preprocess_data(df)
    feature_names = X.columns.tolist()
    print(f"✅ Final feature set: {len(feature_names)} features")
    
    # Step 4: Develop models
    print("\n🤖 DEVELOPING MODELS...")
    models, results, predictions, X_train, X_test, y_train, y_test = develop_models(X, y)
    
    # Step 5: Visualize results
    print("\n📈 VISUALIZING RESULTS...")
    visualize_results(results, predictions, X, y)
    
    # Step 6: SHAP Analysis
    print("\n🔍 PERFORMING SHAP ANALYSIS...")
    shap_values, shap_explainer = shap_analysis(models['random_forest'], X_train, X_test, feature_names)
    
    # Step 7: Save models
    print("\n💾 SAVING MODELS...")
    save_models(models, scaler, le_dict, feature_names, shap_explainer)
    
    # Step 8: Summary
    print("\n" + "="*60)
    print("✅ PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📊 Best Model: Stacking Ensemble")
    print(f"   Accuracy: {results['stack']['accuracy']*100:.2f}%")
    print(f"   ROC-AUC: {results['stack']['roc_auc']:.4f}")
    print(f"\n📁 Files saved:")
    print("   - figures/ (visualizations)")
    print("   - models/ (trained models and preprocessors)")
    print("\n🚀 To run the web app:")
    print("   streamlit run app.py")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()