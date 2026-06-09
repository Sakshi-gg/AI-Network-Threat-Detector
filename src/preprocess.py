import pandas as pd
import numpy as np

def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def get_features(df, label_col='Label'):
    drop_cols = [label_col, 'Flow ID', 'Source IP',
                 'Destination IP', 'Timestamp']
    drop_cols = [c for c in drop_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[label_col]
    return X, y
