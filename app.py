from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib
import os
import tempfile
from utils import add_age_bin  # shared utility

app = Flask(__name__)

# Path to your deployed model
MODEL_PATH = os.path.join("outputs", "best_model_LogisticRegression.joblib")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Load trained pipeline
model = joblib.load(MODEL_PATH)
print(f"[+] Loaded model from {MODEL_PATH}")

# Infer required columns from pipeline (if possible)
try:
    REQUIRED_COLUMNS = model.feature_names_in_.tolist()
except AttributeError:
    REQUIRED_COLUMNS = []

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

        # Check for required columns if available
        if REQUIRED_COLUMNS:
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                return render_template(
                    "index.html",
                    graphs=graphs,
                    error=f"CSV is missing required columns: {', '.join(missing_cols)}"
                )

        # Apply age bin feature
        df = add_age_bin(df)

        # Make predictions
        preds = model.predict(df)
        df['predicted_class'] = preds

        # Optional: predicted probabilities if pipeline supports it
        try:
            y_proba = model.predict_proba(df)[:, 1]
            df['predicted_proba'] = y_proba
        except Exception:
            pass

        # Save predictions to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(temp_file.name, index=False)
        temp_file.close()

        return send_file(temp_file.name, as_attachment=True, download_name="predictions.csv")

    return render_template("index.html", graphs=graphs)

if __name__ == "__main__":
    app.run(debug=True)
