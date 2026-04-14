# 🏥 Malnutrition Risk Prediction at Hospital Admission

> A machine learning screening tool to identify patients at high risk of protein-calorie malnutrition — enabling earlier dietitian involvement and better patient outcomes.

---

## 📌 Background

When patients are admitted to hospital, completing a comprehensive nutrition assessment at triage is often difficult. While a Subjective Global Assessment (SGA) is typically performed within 48 hours of admission, this relies on the ability of the patient or their proxies to provide accurate health histories.

This project develops a **supplemental screening tool** to flag patients at high risk of protein-calorie malnutrition at the point of admission — before a formal nutrition assessment can be completed. Earlier identification allows Registered Dietitians to be involved sooner, potentially reducing missed diagnoses, enabling earlier nutrition interventions, and also improving billing outcomes.

Data were obtained from the [MIMIC-IV dataset](https://physionet.org/content/mimiciv/), extracted via PostgreSQL queries *(link to SQL query text)*.



---

## 🗂️ Project Structure

malnutrition-prediction/ ├── .dvc/ # DVC configuration directory ├── .dvcignore # DVC ignore file ├── .github/ │ └── workflows/ │ └── ml-pipeline.yml # CI/CD workflow for ML pipeline ├── .gitignore ├── configs/ │ ├── features.yaml # Feature configuration │ └── model_params.yaml # Model parameters configuration ├── data/ # Data directory (tracked with DVC) ├── metrics/ # Metrics directory ├── models/ # Trained models directory ├── mlruns/ # MLflow experiment runs ├── mlflow.db # MLflow database ├── notebooks/ │ └── malnutrition_prediction.ipynb # Jupyter notebook for exploration ├── README.md ├── requirements.txt # Python dependencies ├── src/ │ ├── experiment.py # Experiment management │ ├── monitor_drift.py # Data drift monitoring │ ├── preprocessing.py # Data preprocessing utilities │ └── train.py # Model training script └── tests/ └── test_preprocessing.py # Unit tests for preprocessing

---

## 📊 Explored Features

| Category | Features |
|---|---|
| Demographics | Age, gender, race, marital status |
| Anthropometrics | Height, weight, BMI |
| Vitals | Systolic/diastolic BP |
| Labs | Hematocrit, albumin, alkaline phosphatase, AST |
| Admission | Admission type, location |
s
---

## 🤖 Model

_Add details about the final model choice, hyperparameters, and rationale here._

**Algorithm:** LightGBM  
**Tracking:** MLflow  
**Config:** `configs/model_params.yaml`

---

## 📈 Results

_Add final model performance metrics here._

| Metric | Value |
|---|---|
| Accuracy | 0.91 |
| Precision |0.62 |
| Recall | 0.71 |
| F1 Score | 0.67 |
| ROC-AUC | 0.86 |


---

## ⚠️ Limitations

Our project is currently in the 'proof of concept phase'. Original models were trained on the MIMIC-IV-demo dataset with additional entires synthesized. For continuous features, each variable was sampled from a normal distribution N(μ,σ2)\mathcal{N}(\mu, \sigma^2)
N(μ,σ2) where μ and σ were estimated directly from the 17 real malnutrition cases, then clipped to physiologically plausible bounds. Categorical variables were drawn from a multinomial distribution parameterized by the observed class proportions, and feature-level missingness was reintroduced stochastically via Bernoulli sampling to match the sparsity patterns of the real data.

---

## 🔭 Future Work

We are currently pending approve for the full MIMIC-IV dataset (~400,000 patient admit entries) and will retrain/tune our models at that point. 

---

## 📚 References

- Johnson, A. et al. MIMIC-IV. PhysioNet. 

---

## 👤 Author

**Raphael** — Registered Dietitian & ML practitioner  
[HuggingFace](https://huggingface.co/raphi-l) · [GitHub](https://github.com/raphi-l)
