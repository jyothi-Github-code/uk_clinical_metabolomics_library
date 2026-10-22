import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def main():
    print("\n" + "="*50)
    print("[*] LOADING UK CLINICAL MONITORING DATA REGISTRY")
    print("="*50)
    filepath = "data/uk_clinical_metabolomics_library.csv"
    
    if not os.path.exists(filepath):
        print(f"[ERROR] Could not find data file at {filepath}.")
        return
        
    df = pd.read_csv(filepath)
    
    # Core quality controls to clean missing/negative markers
    df = df.drop_duplicates(subset=['sample_id'], keep='first')
    if df['serum_creatinine_mg_dl'].isnull().sum() > 0:
        df['serum_creatinine_mg_dl'] = df.groupby('target_compound_class')['serum_creatinine_mg_dl'].transform(lambda x: x.fillna(x.median()))
    df = df[df['peak_abundance_counts'] >= 0]
    
    # ==========================================
    # 📑 TASK 1: GENERATE EXPLORATORY CHARTS (Q7)
    # ==========================================
    print("[*] Generating exploratory visualization panels...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Chart 1: Top-Left Quadrant [0, 0]
    sns.histplot(data=df, x='serum_creatinine_mg_dl', hue='target_compound_class', kde=True, ax=axes[0, 0], multiple="stack")
    axes[0, 0].set_title('1. Serum Creatinine Distribution by Patient Class')

    # Chart 2: Top-Right Quadrant [0, 1]
    sns.scatterplot(data=df, x='patient_age', y='drug_dosage_mg', hue='target_compound_class', ax=axes[0, 1], s=80)
    axes[0, 1].set_title('2. Patient Age vs. Drug Dosage Interaction Bounds')

    # Chart 3: Bottom-Left Quadrant [1, 0]
    sns.boxplot(data=df, x='target_compound_class', y='fragment_count', ax=axes[1, 0], palette='Set2')
    axes[1, 0].set_title('3. Fragment Counts Across Patient Target Classes')

    # Chart 4: Bottom-Right Quadrant [1, 1]
    numeric_cols = ['patient_age', 'serum_creatinine_mg_dl', 'drug_dosage_mg', 'peak_abundance_counts', 'fragment_count']
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1, 1])
    axes[1, 1].set_title('4. Multi-Variable Feature Correlation Matrix')

    plt.tight_layout()
    plt.savefig('clinical_exploratory_charts.png', dpi=300)
    print("[SUCCESS] 4 charts compiled into: 'clinical_exploratory_charts.png'")
    
    # ==========================================
    # 📊 TASK 2: EXECUTE HYPOTHESIS TESTING (Q8)
    # ==========================================
    print("\n" + "-"*50)
    print("📋 COMPUTING STATISTICAL HYPOTHESIS TESTING (WELCH'S T-TEST)")
    print("-"*50)
    
    # Filter out our target contrast groups
    stable_group = df[df['target_compound_class'] == 'Stable_Response']['serum_creatinine_mg_dl']
    toxicity_group = df[df['target_compound_class'] == 'Toxicity_Risk']['serum_creatinine_mg_dl']
    
    # Execute Independent Samples T-Test with Welch's Correction (equal_var=False)
    t_stat, p_val = stats.ttest_ind(stable_group, toxicity_group, equal_var=False)
    
    # Calculate Cohen's d to determine the physical magnitude of the effect size
    n1, n2 = len(stable_group), len(toxicity_group)
    var1, var2 = np.var(stable_group, ddof=1), np.var(toxicity_group, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (np.mean(stable_group) - np.mean(toxicity_group)) / pooled_std
    
    print(f"[METRIC] Calculated T-Statistic : {t_stat:.4f}")
    print(f"[METRIC] Calculated P-Value     : {p_val:.4e}")
    print(f"[METRIC] Cohen's d Effect Size  : {abs(cohens_d):.4f}")
    
    print("-"*50)
    if p_val < 0.05:
        print("🚩 RESULT: Reject the Null Hypothesis (H0). The data is statistically significant.")
    else:
        print("🏳️ RESULT: Fail to Reject the Null Hypothesis (H0). No significance.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
