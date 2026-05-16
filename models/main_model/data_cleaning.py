import pandas as pd
import numpy as np

def clean_and_load_data(filepath="c:/coding/my projects/PCOS/models/PCOS DataSet.csv"):
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Standardize column names
    df.columns = df.columns.str.strip()
    
    # Ensure target variable 'PCOS' is cleanly numeric 0/1
    df['PCOS'] = pd.to_numeric(df['PCOS'], errors='coerce')
    df = df.dropna(subset=['PCOS'])
    df['PCOS'] = df['PCOS'].astype(int)
    
    # Process text/categorical columns
    cat_columns = df.select_dtypes(include=['object']).columns
    for col in cat_columns:
        # Strip extra whitespace
        df[col] = df[col].astype(str).str.strip()
        
        # Unify representations of "I do not know"
        df[col] = df[col].replace({
            'i do not know': 'I do not know', 
            'I Do Not Know': 'I do not know',
            "I don't know": 'I do not know'
        })
        
        # Fix known typos or numerical glitches in text fields
        df[col] = df[col].replace({
            '2': np.nan, 
            'nan': np.nan, 
            'NaN': np.nan, 
            'None': np.nan
        })
    
    # Convert numerical columns back to float/int if they were accidentally string typed
    # For example, Height, Weight, BMI might have "I do not know" or NaNs
    numerical_cols = ['Weight(kg)', 'Height(m)', 'BMI(kg/m*m)']
    for col in numerical_cols:
        if col in df.columns:
            # Replace "I do not know" with NaN
            df[col] = df[col].replace('I do not know', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

if __name__ == "__main__":
    df = clean_and_load_data()
    print("Cleaned data shape:", df.shape)
    print("PCOS class distribution:\n", df['PCOS'].value_counts())
