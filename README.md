# 📊 Credit Score Checking

🔗 **Live Demo:** [https://creditscorechecking-11.onrender.com/](https://creditscorechecking-11.onrender.com/)

---

## 📝 Project Overview

This project predicts creditworthiness using a **Logistic Regression** model trained on financial data. Users can input their financial information through a web interface and receive a **credit score prediction** along with high-risk customer identification.

---

## 🛠️ Technologies Used

- **Backend:** Flask  
- **Machine Learning:** scikit-learn, joblib  
- **Data Processing:** pandas, numpy  
- **Deployment:** Render  

---

## 🚀 Features

- User-friendly web interface for CSV input  
- Real-time credit score prediction  
- High-risk customer identification  
- Model retraining capability  

---

## 📂 Project Structure

.
├── app.py # Flask application
├── credit_scoring.py # Model training script
├── data/ # Dataset
├── outputs/ # Model and predictions
├── requirements.txt # Project dependencies
└── .gitignore # Git ignore rules

yaml
Copy code

---

## 📥 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/CreditScoreChecking.git
cd CreditScoreChecking
Set up a virtual environment

bash
Copy code
python -m venv env
# Activate on Linux/Mac
source env/bin/activate
# Activate on Windows
env\Scripts\activate
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the application

bash
Copy code
python app.py
🎯 Using the Web App
Upload your CSV file with financial data.

Click Submit to generate predictions.

Download the predictions CSV containing:

predicted_class → 0 = High Risk, 1 = Low Risk

predicted_proba → Probability of being low-risk (if available)


