# 🏥 Malnutrition Risk Prediction at Hospital Admission

> A machine learning screening tool to identify patients at high risk of protein-calorie malnutrition — enabling earlier dietitian involvement and better patient outcomes.

---

## 📌 Background

When patients are admitted to hospital, completing a comprehensive nutrition assessment at triage is often difficult. While a Subjective Global Assessment (SGA) is typically performed within 48 hours of admission, this relies on the ability of the patient or their proxies to provide accurate health histories.

This project develops a **supplemental screening tool** to flag patients at high risk of protein-calorie malnutrition at the point of admission — before a formal nutrition assessment can be completed. Earlier identification allows Registered Dietitians to be involved sooner, potentially reducing missed diagnoses, enabling earlier nutrition interventions, and also improving billing outcomes.

Data were obtained from the [MIMIC-IV dataset](https://physionet.org/content/mimiciv/), extracted via PostgreSQL queries *(link to SQL query text)*.



---

## 🗂️ Project Structure
```
malnutrition-prediction/
├── .dvc/                          
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml        
├── configs/
│   ├── features.yaml              
│   └── model_params.yaml          
├── data/                          
├── metrics/                       
├── models/                        
├── mlruns/                        
├── notebooks/
├── src/
│   ├── experiment.py              
│   ├── monitor_drift.py           
│   ├── preprocessing.py           
│   └── train.py                   
├── tests/
│   └── test_preprocessing.py
├── .gitignore
├── mlflow.db
├── README.md
└── requirements.txt
```
---

## 📊 Explored Features

| Category | Features |
|---|---|
| Demographics | age, gender, race, marital_status |
| Anthropometric | height_cm, weight_kg, bmi |
| Labs | glucose_admit, hematocrit_admit, hemoglobin_admit, potassium_admit |
| Vitals | systolic, diastolic |
| Admission Info | admission_type, admission_location |
| Missingness Indicators | alk_phos_ordered, ast_ordered |
---

## 🤖 Model

LightGBM was selected for its strong performance on tabular clinical data, where tree-based ensemble methods consistently outperform deep learning
approaches at this scale. Its native support for categorical features and built-in handling of missing values reduces preprocessing overhead. Probability outputs from the
classifier allows us to explore different thresholds for identifying malnutrition risk. While lower-thresholds allow up to maximize recall, we will be able to deploy real-time adjustments to reflect
the realities of patient census and nutrition-department staffing levels.

Tree-based models also provide feature importance scores add a layer of interpretability, surfacing the strongest predictors of malnutrition risk
and supporting clinical validation of the model's behaviour.

**Algorithm:** LightGBM  
**Tracking:** MLflow  
**Config:** `configs/model_params.yaml`

---

## 📈 Results

Current model version: 0_1

| Metric | Value |
|---|---|
| Accuracy | 0.91 |
| Precision |0.62 |
| Recall | 0.71 |
| F1 Score | 0.67 |
| ROC-AUC | 0.86 |

For simplicity, we maintained the default probability treshold of 0.5 for this exploratory model. With a recall of 71% we are able to identify a sizeable portion patients who were diagnosed with malnutrition at some point in their admit. At this point, it is unclear whether are precision score is indicative of true false-postives (i.e. patients who would have met clincal critria for malnutrition, but was discharaged before formal assessment/diagnosis). We will be able to address this once our model is deployed. 

---

## ⚠️ Limitations

Our project is currently in the 'proof of concept phase'. 
Original models were trained on the MIMIC-IV-demo dataset with additional entires synthesized. 

For continuous features, each variable was sampled from a normal distribution
$\mathcal{N}(\mu, \sigma^2)$ where $\mu$ and $\sigma$ were estimated directly
from the 17 real malnutrition cases, then clipped to physiologically plausible
bounds. Categorical variables were drawn from a multinomial distribution
parameterized by the observed class proportions, and feature-level missingness
was reintroduced stochastically via Bernoulli sampling to match the sparsity
patterns of the real data.

As our current model is based on retrospective data, it does not yet address patients who "fall through the cracks", the ultimate target of this model.

---

## 🔭 Future Work

We are currently pending approve for the full MIMIC-IV dataset (~400,000 patient admit entries) and will retrain/tune our models at that point. This will allow us to fine-tube hyperparameters and our detection treshold. 

Once deployed, we will also be able to compare the strength of this tool again existing hospital policy and procedures. Specifically, we will work to identify patients who were correctly identified by our model but would have been missed by extant hospital workflow. 

---

## 📚 References

- Johnson, A. et al. MIMIC-IV. PhysioNet. 

---

## 👤 Author

**Raphael** — Registered Dietitian & ML practitioner  
[HuggingFace](https://huggingface.co/raphi-l) · [GitHub](https://github.com/raphi-l)
