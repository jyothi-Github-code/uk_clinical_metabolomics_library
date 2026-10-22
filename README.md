# UK Clinical Medication Pipeline Guard

An automated, machine learning validation guard framework that replaces traditional, rigid conditional lookup checks within software integration pipelines.

## 🚀 Architectural Design Overview
This system processes telemetry data from clinical logs, automatically executing data quality checks, filtering missing values/sensor artifacts, and applying a cost-sensitive **Random Forest Classifier** model to ensure comprehensive fault isolation before deployment to production environments.

## 📁 Repository Layout
* `data/`: Protected directory containing sample clinical evaluation templates.
* `src/pipeline.py`: Production-grade quality control steps, feature transformation matrices, and leakage isolation functions.
* `src/models.py`: Structural definitions for the legacy benchmark model and the trained ensemble classifier.
* `main.py`: The single-point execution coordinator script used within CI environments.

## 🛠️ Step-by-Step Setup and Execution
To run this application locally inside VS Code, execute the following commands in your terminal workspace:

1. **Install required infrastructure libraries:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute the automated data cleaning and machine learning pipeline:**
   ```bash
   python main.py
   ```

## 🐋 Docker Compilation Framework
To run this application within an isolated container environment, run the following commands:
```bash
docker build -t clinical-ai-pipeline .
docker run --rm clinical-ai-pipeline
```
