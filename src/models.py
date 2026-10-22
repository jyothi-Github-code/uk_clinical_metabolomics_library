import numpy as np
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
