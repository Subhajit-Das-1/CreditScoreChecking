from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib
import os
import tempfile
from utils import add_age_bin  # <-- import from utils.py

app = Flask(__name__)

# Load trained model
MODEL_PATH = "outputs/best_model_LogisticRegression.joblib"
model = joblib.load(MODEL_PATH)

# Try to infer required columns
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

        # Check required columns
        if REQUIRED_COLUMNS:
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                return render_template(
                    "index.html",
                    graphs=graphs,
                    error=f"CSV is missing required columns: {', '.join(missing_cols)}"
                )

        # Apply transformation
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
    app.run(debug=True)
