# 🏥 Malnutrition Risk Prediction at Hospital Admission

> A machine learning screening tool to identify patients at high risk of protein-calorie malnutrition — enabling earlier dietitian involvement and better patient outcomes.

---

## 📌 Background

When patients are admitted to hospital, completing a comprehensive nutrition assessment is often difficult. While a Subjective Global Assessment (SGA) is typically performed within 48 hours of admission, this relies on the ability of the patient or their proxies to provide accurate health histories.

This project develops a **supplemental screening tool** to flag patients at high risk of protein-calorie malnutrition at the point of admission — before a formal nutrition assessment can be completed. Earlier identification allows Registered Dietitians to be involved sooner, potentially reducing missed diagnoses, enabling earlier nutrition interventions, and improving billing outcomes.

Data were obtained from the [MIMIC-IV dataset](https://physionet.org/content/mimiciv/), extracted via PostgreSQL queries *(link to SQL query text)*.

---

## 🗂️ Project Structure
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

_Add limitations and caveats here — e.g. dataset size, generalizability, MIMIC population specifics._

---

## 🔭 Future Work

_Add planned improvements here._

---

## 📚 References

- Johnson, A. et al. MIMIC-IV. PhysioNet. 
- McClave et al. (2016). ASPEN/SCCM Clinical Guidelines.
- _(Add additional references)_

---

## 👤 Author

**Raphael** — Registered Dietitian & ML practitioner  
[HuggingFace](https://huggingface.co/raphi-l) · [GitHub](https://github.com/raphi-l)