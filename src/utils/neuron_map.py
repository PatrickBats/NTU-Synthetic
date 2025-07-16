import pandas as pd

def read_dissect_csv(csv_path):
    """Read neuron descriptions CSV file without strict layer prefix validation"""
    df_dissect = pd.read_csv(csv_path)
    
    # Basic validation
    required_columns = ['layer', 'unit']
    if not all(col in df_dissect.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    # Use prep_net_dis_desc if 'description' column is not found
    if 'description' not in df_dissect.columns:
        df_dissect = _prep_net_dis_desc(df_dissect)
    
    # Type conversion for consistency
    df_dissect['unit'] = df_dissect['unit'].astype(int)
    df_dissect['layer'] = df_dissect['layer'].astype(str)
    
    return df_dissect

def _prep_net_dis_desc(df):
    # Create a copy of the DataFrame to avoid warnings
    df_clean = df[['layer', 'unit', 'label', 'score']].copy()
    
    # Use .loc for assignments
    df_clean.loc[:, 'label'] = df_clean['label'].apply(lambda x: x[:-2] if x.endswith('-c') or x.endswith('-s') else x)
    df_clean = df_clean.rename(columns={'label': 'description', 'score': 'similarity'})
    df_clean.loc[:, 'description'] = df_clean['description'].str.replace(r'(-[cs])\b|\t', '', regex=True).str.strip()
    
    return df_clean
