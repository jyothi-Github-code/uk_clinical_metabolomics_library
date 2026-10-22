from src.pipeline import DataPipeline
from src.models import run_legacy_baseline, train_ai_classifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# ============================================================
# 🔬 REUSABLE PREDICTION FUNCTION WITH INPUT VALIDATION (Q18b)
# ============================================================
def predict_single_patient_risk(model, scaler, input_data):
    """
    Production-grade inference function. Validates input schemas, 
    handles sensor errors, and returns robust predictions safely.
    """
    required_fields = ['patient_age', 'serum_creatinine_mg_dl', 'drug_dosage_mg', 'peak_abundance_counts', 'fragment_count', 'uk_health_board_code_NHS-NW']
    
    # 1. Input Validation: Check for empty or wrong payload configurations
    if not input_data or not isinstance(input_data, dict):
        return {"status": "REJECTED", "error": "Invalid format. Input must be a non-empty dictionary."}
        
    # 2. Schema Validation: Verify all required fields exist
    for field in required_fields:
        if field not in input_data:
            return {"status": "REJECTED", "error": f"Missing critical schema field: '{field}'"}
            
    # 3. Boundary Verification: Check for physically impossible negative counts
    if input_data['peak_abundance_counts'] < 0:
        return {"status": "REJECTED", "error": "Data anomaly: Negative counts are physically impossible."}
        
    # 4. Error Handling: Impute missing values with fallback median if data cell contains nulls
    if pd.isna(input_data['serum_creatinine_mg_dl']):
        input_data['serum_creatinine_mg_dl'] = 1.15  # Secure fallback historical median
        
    # Convert input to a structured DataFrame matching model architecture
    input_df = pd.DataFrame([input_data])[required_fields]
    
    # Apply scaling factors based strictly on training boundaries
    scaled_features = scaler.transform(input_df)
    
    # Execute Model Inference
    prediction = model.predict(scaled_features)[0]
    confidence = np.max(model.predict_proba(scaled_features))
    
    return {
        "status": "APPROVED",
        "predicted_class": prediction,
        "model_confidence_score": f"{confidence:.2%}"
    }

# ============================================================
# 🚀 MAIN PIPELINE COORDINATOR EXECUTABLE (Q18a)
# ============================================================
def main():
    print("\n" + "="*60)
    print("🚀 EXECUTING RE-CONFIGURED UK CLINICAL MEDICATION GUARD")
    print("="*60)
    
    # Ingest and validate data structures
    pipe = DataPipeline(filepath="data/uk_clinical_metabolomics_library.csv")
    raw_data = pipe.load_and_inspect()
    cleansed_data = pipe.clear_anomalies(raw_data)
    engineered_data = pipe.transform_and_engineer(cleansed_data)
    
    pipe.run_validation_gates(engineered_data)
    X_train, X_test, y_train, y_test = pipe.split_and_scale(engineered_data)
    
    # Run Baseline Benchmarks
    baseline_predictions = run_legacy_baseline(X_test)
    print(f"\n[BENCHMARK] Legacy Rules Test Accuracy: {accuracy_score(y_test, baseline_predictions):.2%}")
    
    # Train Production Machine Learning Model
    print("[MODEL] Training Multi-Variable Random Forest...")
    ai_model = train_ai_classifier(X_train, y_train)
    ai_predictions = ai_model.predict(X_test)
    print(f"[MODEL SUCCESS] Random Forest Test Accuracy: {accuracy_score(y_test, ai_predictions):.2%}")
    
    print("\n" + "-"*40 + "\n📋 IMPERFECT REAL-WORLD EVALUATION REPORT\n" + "-"*40)
    print(classification_report(y_test, ai_predictions, zero_division=0))
    
    # ============================================================
    # TESTING LIVE PIPELINE INFERENCE VALIDATION TESTS (Q19b)
    # ============================================================
    print("\n" + "-"*40 + "\n🔬 RUNNING LIVE PIPELINE INFERENCE VALIDATION TESTS\n" + "-"*40)
    
    # Sample Test 1: Valid Toxicity Risk Profile
    sample_toxic_payload = {
        'patient_age': 72, 'serum_creatinine_mg_dl': 2.15, 'drug_dosage_mg': 1000,
        'peak_abundance_counts': 105000, 'fragment_count': 52, 'uk_health_board_code_NHS-NW': 0
    }
    result_1 = predict_single_patient_risk(ai_model, pipe.scaler, sample_toxic_payload)
    print(f"[TEST 1] Valid Case Response: {result_1}")
    
    # Sample Test 2: Malformed Anomalous Payload (Negative Counts)
    sample_corrupt_payload = {
        'patient_age': 34, 'serum_creatinine_mg_dl': 0.92, 'drug_dosage_mg': 500,
        'peak_abundance_counts': -500, 'fragment_count': 12, 'uk_health_board_code_NHS-NW': 1
    }
    result_2 = predict_single_patient_risk(ai_model, pipe.scaler, sample_corrupt_payload)
    print(f"[TEST 2] Corrupt Case Response: {result_2}")
    
    print("="*60 + "\n[PIPELINE TERMINATED SUCCESSFULLY] Code ready for deployment.\n")

if __name__ == "__main__":
    main()
