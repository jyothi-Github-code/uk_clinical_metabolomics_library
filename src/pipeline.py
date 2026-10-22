import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPipeline:
    def __init__(self, filepath):
        self.filepath = filepath
        self.scaler = StandardScaler()
        self.features = ['patient_age', 'serum_creatinine_mg_dl', 'drug_dosage_mg', 'peak_abundance_counts', 'fragment_count', 'uk_health_board_code_NHS-NW']

    def load_and_inspect(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Missing file path at: {self.filepath}")
        df = pd.read_csv(self.filepath)
        print(f"\n[CLINICAL INGESTION] Loaded {df.shape[0]} patient records.")
        return df

    def clear_anomalies(self, df):
        df = df.drop_duplicates(subset=['sample_id'], keep='first')
        if df['serum_creatinine_mg_dl'].isnull().sum() > 0:
            df['serum_creatinine_mg_dl'] = df.groupby('target_compound_class')['serum_creatinine_mg_dl'].transform(lambda x: x.fillna(x.median()))
        df = df[df['peak_abundance_counts'] >= 0]
        return df

    def transform_and_engineer(self, df):
        df = pd.get_dummies(df, columns=['uk_health_board_code'], drop_first=True)
        # Ensure dummy column consistency across validation matrix splits
        for col in self.features:
            if col not in df.columns and col != 'uk_health_board_code_NHS-NW':
                pass
        if 'uk_health_board_code_NHS-NW' not in df.columns:
            df['uk_health_board_code_NHS-NW'] = 0
        return df

    def run_validation_gates(self, df):
        assert df['peak_abundance_counts'].min() >= 0, "QC Gate Violation: Negative counts."
        assert df['serum_creatinine_mg_dl'].isnull().sum() == 0, "QC Gate Violation: Missing values."
        print("[+] Re-runnable clinical data validation gates passed successfully.")

    def split_and_scale(self, df):
        X = df[self.features]
        y = df['target_compound_class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=self.features)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=self.features)
        return X_train_scaled, X_test_scaled, y_train, y_test
