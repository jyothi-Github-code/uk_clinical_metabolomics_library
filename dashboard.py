import pandas as pd
import numpy as np

def run_pipeline_monitoring_dashboard(filepath="data/uk_clinical_metabolomics_library.csv"):
    print("\n" + "="*50)
    print("📊 LIVE CI/CD PERFORMANCE & DATA DRIFT DASHBOARD")
    print("="*50)
    
    df = pd.read_csv(filepath)
    
    # 1. Track Data Quality Metrics
    total_records = len(df)
    missing_creatinine = df['serum_creatinine_mg_dl'].isnull().sum()
    missing_rate = missing_creatinine / total_records
    
    # 2. Monitor Data Drift (Tracking if the incoming mean age shifts radically)
    historical_mean_age = 55.0
    current_mean_age = df['patient_age'].mean()
    age_drift = abs(current_mean_age - historical_mean_age)
    
    # Display Live Dashboard Metrics
    print(f"[MONITOR] Active Pipeline Log Volume : {total_records} records.")
    print(f"[MONITOR] Missing Data Cell Rate    : {missing_rate:.2%}")
    print(f"[MONITOR] Calculated Age Metric Drift: {age_drift:.2f} Years")
    
    # 3. System Alerting Threshold Gates
    print("\n--- Active System Alert Status ---")
    if missing_rate > 0.10:
        print("🚨 ALERT: Missing data rate has broken the 10% threshold! Check sensor integrity.")
    else:
        print("✅ SYSTEM HEALTH: Data missing rate sits within safe boundaries.")
        
    if age_drift > 5.0:
        print("🚨 ALERT: Data Drift Detected! The incoming patient population profile has shifted significantly.")
    else:
        print("✅ SYSTEM HEALTH: Population drift metrics sit within normal parameters.")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_pipeline_monitoring_dashboard()
