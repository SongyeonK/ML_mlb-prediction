import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """CSV 파일에서 데이터를 로드합니다."""
    return pd.read_csv(filepath)

def check_missing_values(df):
    """결측치를 확인하고 비율을 반환합니다."""
    missing_ratio = df.isnull().sum() / len(df) * 100
    return missing_ratio[missing_ratio > 0]

def scale_features(df, features):
    """특정 열을 스케일링합니다."""
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df
