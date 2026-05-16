import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import pickle
import os
from data_cleaning import clean_and_load_data
from train_xgboost import engineer_features

def explain_model():
    print("Loading model and data for SHAP analysis...")
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    df = clean_and_load_data()
    
    
    df = engineer_features(df)
    
    X = df.drop(columns=['PCOS'])
    
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    output_dir = "artifacts"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating Global Summary Plot...")
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(f"{output_dir}/shap_summary_plot.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Generating Local Waterfall Plot for a high-risk individual...")
   
    X_reset = X.reset_index(drop=True)
    df_reset = df.reset_index(drop=True)
    
    high_risk_idx = df_reset[df_reset['PCOS'] == 1].index[0]
    
    explanation = explainer(X_reset)
    
    plt.figure()
    shap.plots.waterfall(explanation[high_risk_idx], show=False)
    plt.savefig(f"{output_dir}/shap_waterfall_plot.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    print("SHAP artifacts saved successfully.")

if __name__ == "__main__":
    explain_model()
