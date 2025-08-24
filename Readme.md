# Credit Rating Prediction System (ML + Streamlit)

## Project Overview
This project predicts **credit ratings** for customers (Poor, Fair, Good, Very Good, Excellent) using a machine learning pipeline.  
It combines **financial attributes** (balance, loans, age, education, etc.) with a synthetic credit score to classify customer risk levels.  

The project includes:
- Data preprocessing & feature engineering
- Model training (Logistic Regression, Decision Tree, Random Forest)
- Deployment of a **Streamlit app** for interactive predictions
- Feature importance visualization for explainability

---

## Dataset
- Base dataset: Bank Marketing dataset (`bank.csv`)
- Engineered features:
  - `credit_score` (numeric 300–850 range)
  - `credit_rating` (categories: Poor → Excellent)

---

##  Tech Stack
- **Python** (pandas, numpy, scikit-learn)
- **Streamlit** (for deployment)
- **Matplotlib/Seaborn** (for visualization)
- **Joblib** (for saving/loading models)

---

##  How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/credit-scoring-ml.git
   cd credit-scoring-ml

# Install dependencies:
pip install -r requirements.txt

# Run the Streamlit app:
streamlit run app.py


