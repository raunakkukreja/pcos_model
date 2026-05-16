import pickle
import pandas as pd
import numpy as np

# Base profile with all features the model expects
BASE_PROFILE = {
    'Family_background': 0, 'LH_hormone': 0, 'FSH_hormone': 0, 
    'Diabetes_measurment': 0, 'TSH_hormone': 0, 'Prolactin_hormone': 0,
    'Hemoglobin_level': 0, 'Cyst_ovary': 0, 'Diagnosis_age': 1, 
    'Overweight': 0, 'Period_type': 0, 'Hormonal_imbalance': 0, 
    'Gain_weight': 0, 'Excess_facial_hair': 0, 'Excess_body_hair': 0, 
    'Dark_area': 0, 'Pimple_face': 0, 'Hormonal_acne_face': 0, 
    'Blood_pressure': 1, 'Fast_food': 0, 'Losing_hair': 0, 
    'Ex_pain_abdominal': 0, 'Mood_swing_period': 1, 'Mood_swing_normal': 1, 
    'Craving_PT': 1, 'Depress': 0, 'Mental_stress': 0, 'Insomnia': 0, 
    'Slow_activity': 0, 'Weight(kg)': 60.0, 'Height(m)': 1.65, 
    'BMI(kg/m*m)': 22.0, 'Metabolic_Score': 0.0, 'Hormonal_Symptom_Score': 0.0
}

# Key risk factors to ask the user
INTERACTIVE_FEATURES = [
    ("Period_type", "Are periods irregular?", "1 = Yes, 0 = No", int),
    ("Cyst_ovary", "History of ovarian cysts?", "1 = Yes, 0 = No", int),
    ("Gain_weight", "Recent unusual weight gain?", "1 = Yes, 0 = No", int),
    ("Excess_facial_hair", "Excess facial hair?", "1 = Yes, 0 = No", int),
    ("Fast_food", "Frequent fast food consumption?", "1 = Yes, 0 = No", int),
    ("Weight(kg)", "Weight in kg", "e.g., 65.0", float),
    ("Height(m)", "Height in meters", "e.g., 1.65", float),
]

def get_user_input():
    print("--- PCOS Prediction Demo ---")
    print("Please answer a few key questions (Press ENTER to use healthy defaults).")
    
    user_data = BASE_PROFILE.copy()
    
    for key, desc, prompt, dtype in INTERACTIVE_FEATURES:
        while True:
            val = input(f"  {desc} ({prompt}): ").strip()
            if val == "":
                break # use default
            try:
                user_data[key] = dtype(val)
                break
            except ValueError:
                print(f"  [!] Invalid input. Please enter a value as: {prompt}")

    # Auto-calculate derived features
    if 'Weight(kg)' in user_data and 'Height(m)' in user_data and user_data['Height(m)'] > 0:
        user_data['BMI(kg/m*m)'] = user_data['Weight(kg)'] / (user_data['Height(m)'] ** 2)

    # Calculate engineered scores exactly how `engineer_features()` in train_xgboost.py computes them
    user_data['Metabolic_Score'] = sum([
         user_data.get('Overweight', 0), 
         user_data.get('Fast_food', 0), 
         user_data.get('Gain_weight', 0)
    ])
    
    user_data['Hormonal_Symptom_Score'] = sum([
         user_data.get('Excess_facial_hair', 0), 
         user_data.get('Excess_body_hair', 0), 
         user_data.get('Hormonal_acne_face', 0), 
         user_data.get('Pimple_face', 0)
    ])

    # Model's expected feature order
    features_order = [
        'Family_background', 'LH_hormone', 'FSH_hormone', 'Diabetes_measurment', 'TSH_hormone', 'Prolactin_hormone',
        'Hemoglobin_level', 'Cyst_ovary', 'Diagnosis_age', 'Overweight', 'Period_type', 'Hormonal_imbalance',
        'Gain_weight', 'Excess_facial_hair', 'Excess_body_hair', 'Dark_area', 'Pimple_face', 'Hormonal_acne_face',
        'Blood_pressure', 'Fast_food', 'Losing_hair', 'Ex_pain_abdominal', 'Mood_swing_period', 'Mood_swing_normal',
        'Craving_PT', 'Depress', 'Mental_stress', 'Insomnia', 'Slow_activity', 'Weight(kg)', 'Height(m)', 'BMI(kg/m*m)',
        'Metabolic_Score', 'Hormonal_Symptom_Score'
    ]
    
    # Return as pandas DataFrame
    return pd.DataFrame([user_data])[features_order]

def main():
    try:
        with open("xgb_model.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Model file 'xgb_model.pkl' not found. Please train the model first.")
        return

    X = get_user_input()

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    
    print("\n" + "="*35)
    print(f"Prediction result: {'PCOS Risk Detected (1)' if prediction == 1 else 'Low PCOS Risk (0)'}")
    print(f"Risk Probability:  {probability:.1%}")
    print("="*35)
    
    try:
        import shap
        print("\nAnalyzing factors (SHAP)...")
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)[0]
        
        feature_names = X.columns
        contributions = list(zip(feature_names, shap_vals, X.iloc[0]))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("\n--- Key Risk Explanations ---")
        for feature_name, shap_val, feature_val in contributions[:5]:
            if shap_val > 0:
                print(f"  [↑] Increases Risk: {feature_name} = {feature_val} (Impact: +{shap_val:.2f})")
            else:
                print(f"  [↓] Decreases Risk: {feature_name} = {feature_val} (Impact: {shap_val:.2f})")
        print("-----------------------------\n")
    except ImportError:
        pass

if __name__ == "__main__":
    main()