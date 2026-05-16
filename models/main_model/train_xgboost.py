import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve
from data_cleaning import clean_and_load_data
import pickle

def engineer_features(df):
    # Mapping dictionaries
    binary_map = {'Yes': 1, 'No': 0, 'May be': np.nan, 'I do not know': -1}
    abnormal_map = {'Abnormal': 1, 'Normal': 0, 'I do not know': -1}
    ordinal_map = {'Low': 0, 'Normal': 1, 'Medium': 1, 'High': 2, 'Extreme': 3, 'I do not know': -1}
    period_map = {'Irregular': 1, 'Regular': 0, 'I do not know': -1}
    age_map = {'Youth': 0, 'Young adult': 1, 'adult': 2, 'I do not know': -1}
    
    # Apply mappings manually 
    binary_cols = ['Family_background', 'Cyst_ovary', 'Overweight', 'Hormonal_imbalance', 'Gain_weight',
                   'Excess_facial_hair', 'Excess_body_hair', 'Dark_area', 'Pimple_face', 'Hormonal_acne_face',
                   'Fast_food', 'Losing_hair', 'Depress', 'Mental_stress', 'Insomnia', 'Slow_activity']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: binary_map.get(str(x).strip(), np.nan) if not pd.isna(x) else x)

    abnormal_cols = ['LH_hormone', 'FSH_hormone', 'Diabetes_measurment', 'TSH_hormone', 'Prolactin_hormone', 'Hemoglobin_level']
    for col in abnormal_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: abnormal_map.get(str(x).strip(), np.nan) if not pd.isna(x) else x)

    ordinal_cols = ['Blood_pressure', 'Mood_swing_period', 'Mood_swing_normal', 'Craving_PT']
    for col in ordinal_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: ordinal_map.get(str(x).strip(), np.nan) if not pd.isna(x) else x)
            
    if 'Period_type' in df.columns:
        df['Period_type'] = df['Period_type'].map(lambda x: period_map.get(str(x).strip(), np.nan) if not pd.isna(x) else x)
        
    if 'Diagnosis_age' in df.columns:
        df['Diagnosis_age'] = df['Diagnosis_age'].map(lambda x: age_map.get(str(x).strip(), np.nan) if not pd.isna(x) else x)
        
    if 'Ex_pain_abdominal' in df.columns:
        df['Ex_pain_abdominal'] = df['Ex_pain_abdominal'].map({'Yes': 1, 'Sometimes': 0.5, 'No': 0, 'I do not know': -1})

    # Ensure everything is float 
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Feature Engineering Aggregates
    df['Metabolic_Score'] = df[['Overweight', 'Fast_food', 'Gain_weight']].replace(-1, np.nan).sum(axis=1, min_count=1)
    df['Hormonal_Symptom_Score'] = df[['Excess_facial_hair', 'Excess_body_hair', 'Hormonal_acne_face', 'Pimple_face']].replace(-1, np.nan).sum(axis=1, min_count=1)

    return df

def train_baseline_model():
    print("Loading and cleaning data...")
    df = clean_and_load_data()
    print("Engineering features...")
    df = engineer_features(df)
    
    # Features and Target
    X = df.drop(columns=['PCOS'])
    y = df['PCOS']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class prior (PCOS=1): {y.mean():.2f}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Class imbalance weight natively handled by XGBoost
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    # XGBoost configuration
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='aucpr',
        use_label_encoder=False
    )
    
    print("Training XGBClassifier...")
    model.fit(X_train, y_train)
    
    print("\n--- Evaluation on Test Set ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    
    pr_auc = average_precision_score(y_test, y_prob)
    print(f"PR-AUC Score: {pr_auc:.3f}")
    
    # Save the model
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    print("Model saved to xgb_model.pkl")
    
    return model, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    train_baseline_model()
