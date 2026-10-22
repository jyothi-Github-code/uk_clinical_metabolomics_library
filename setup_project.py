import os

# Define the repository directory layouts
directories = ["data", "src"]
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[+] Directory created: {directory}")

# 1. Structural contents for src/pipeline.py
pipeline_code = """import os
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
        print(f"\\n[CLINICAL INGESTION] Loaded {df.shape[0]} patient records.")
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
"""

# 2. Structural contents for src/models.py
models_code = """import numpy as np
from sklearn.ensemble import RandomForestClassifier

def run_legacy_baseline(X_test):
    predictions = []
    for _, row in X_test.iterrows():
        if row['serum_creatinine_mg_dl'] > 0.5 and row['drug_dosage_mg'] > 0.5:
            predictions.append('Toxicity_Risk')
        elif row['serum_creatinine_mg_dl'] > 0.0:
            predictions.append('Adverse_Reaction')
        else:
            predictions.append('Stable_Response')
    return np.array(predictions)

def train_ai_classifier(X_train, y_train):
    model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model
"""

# 3. Structural contents for main.py
main_code = """from src.pipeline import DataPipeline
from src.models import run_legacy_baseline, train_ai_classifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("\\n" + "="*60)
    print("🚀 EXECUTING RE-CONFIGURED UK CLINICAL MEDICATION GUARD")
    print("="*60)
    
    pipe = DataPipeline(filepath="data/uk_clinical_metabolomics_library.csv")
    raw_data = pipe.load_and_inspect()
    cleansed_data = pipe.clear_anomalies(raw_data)
    engineered_data = pipe.transform_and_engineer(cleansed_data)
    
    pipe.run_validation_gates(engineered_data)
    X_train, X_test, y_train, y_test = pipe.split_and_scale(engineered_data)
    
    baseline_predictions = run_legacy_baseline(X_test)
    print(f"\\n[BENCHMARK] Legacy Rules Accuracy: {accuracy_score(y_test, baseline_predictions):.2%}")
    
    print("[MODEL] Training Multi-Variable Random Forest...")
    ai_model = train_ai_classifier(X_train, y_train)
    ai_predictions = ai_model.predict(X_test)
    
    print(f"[MODEL SUCCESS] Random Forest Test Accuracy: {accuracy_score(y_test, ai_predictions):.2%}")
    print("\\n" + "-"*40 + "\\n📋 IMPERFECT REAL-WORLD EVALUATION REPORT\\n" + "-"*40)
    print(classification_report(y_test, ai_predictions, zero_division=0))
    print("="*60 + "\\n[PIPELINE TERMINATED] Ready for portfolio risk analysis.\\n")

if __name__ == "__main__":
    main()
"""

# 4. Environmental requirements and runtime configs
requirements_text = "pandas\nscikit-learn\nnumpy\n"

dockerfile_text = """FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY main.py .
COPY data/ ./data/
CMD ["python", "main.py"]
"""

# Write all structured strings out to their corresponding file objects
files_to_write = {
    "src/__init__.py": "",
    "src/pipeline.py": pipeline_code,
    "src/models.py": models_code,
    "main.py": main_code,
    "requirements.txt": requirements_text,
    "Dockerfile": dockerfile_text
}

for filepath, content in files_to_write.items():
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[+] Document created and populated: {filepath}")

print("\n[SUCCESS] Environment files successfully structured via native Python execution rules.")
