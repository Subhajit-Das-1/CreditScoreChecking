📊 Project: Credit Score Checking

🔗 Live Demo: https://creditscorechecking-11.onrender.com/

This project predicts creditworthiness using a logistic regression model trained on financial data. It provides a web interface for users to input their financial information and receive a credit score prediction.

🛠️ Technologies Used

Backend: Flask

Machine Learning: scikit-learn, joblib

Data Processing: pandas, numpy

Deployment: Render

🚀 Features

User-friendly web interface for input

Real-time credit score prediction

High-risk customer identification

Model retraining capability

📂 Project Structure
.
├── app.py              # Flask application
├── credit_scoring.py   # Model training script
├── data/               # Dataset
├── outputs/            # Model and predictions
├── requirements.txt    # Project dependencies
└── .gitignore          # Git ignore rules

📥 Installation

Clone the repository:

git clone https://github.com/yourusername/CreditScoreChecking.git
cd CreditScoreChecking


Set up a virtual environment:

python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`


Install dependencies:

pip install -r requirements.txt


Run the application:

python app.py
