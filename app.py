from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib
import os
import tempfile

# Custom transformer used in the trained pipeline
def add_age_bin(X):
    X = X.copy()
    if 'age' in X.columns:
        X['age_bin'] = pd.cut(
            X['age'],
            bins=[0, 25, 35, 50, 65, 120],
            labels=['<25', '25-34', '35-49', '50-64', '65+']
        )
    return X

app = Flask(__name__)

# Load trained model once
MODEL_PATH = "outputs/best_model_LogisticRegression.joblib"
model = joblib.load(MODEL_PATH)

# Try to infer required columns from the model if possible
try:
    REQUIRED_COLUMNS = model.feature_names_in_.tolist()  # works for scikit-learn models
except AttributeError:
    REQUIRED_COLUMNS = []  # fallback if model doesn’t have feature_names_in_

@app.route("/", methods=["GET", "POST"])
def index():
    graphs = [
        "confusion_matrix_LogisticRegression.png",
        "roc_LogisticRegression.png"
    ]
    
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return render_template("index.html", graphs=graphs, error="No file uploaded.")
        
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return render_template("index.html", graphs=graphs, error=f"Failed to read CSV: {e}")
        
        # Check for required columns only if we know them
        if REQUIRED_COLUMNS:
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                return render_template(
                    "index.html",
                    graphs=graphs,
                    error=f"CSV is missing required columns: {', '.join(missing_cols)}"
                )
        
        # Apply custom transformation
        df = add_age_bin(df)
        
        # Predict
        preds = model.predict(df)
        df['predicted_class'] = preds
        
        # Save to a unique temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return send_file(temp_file.name, as_attachment=True, download_name="predictions.csv")

    return render_template("index.html", graphs=graphs)

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    app.run(debug=True)
