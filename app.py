from flask import Flask, request, render_template, send_file
import os
import sys
import pandas as pd
import joblib
import tempfile

# Make utils importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import add_age_bin

app = Flask(__name__)
MODEL_PATH = os.path.join("outputs", "best_model.joblib")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
try:
    REQUIRED_COLUMNS = model.feature_names_in_.tolist()
except AttributeError:
    REQUIRED_COLUMNS = []

@app.route("/", methods=["GET", "POST"])
def index():
    graphs = ["confusion_matrix_LogisticRegression.png", "roc_LogisticRegression.png"]

    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return render_template("index.html", graphs=graphs, error="No file uploaded.")

        try:
            df = pd.read_csv(file)
        except Exception as e:
            return render_template("index.html", graphs=graphs, error=f"Failed to read CSV: {e}")

        if REQUIRED_COLUMNS:
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                return render_template("index.html", graphs=graphs, error=f"CSV missing columns: {', '.join(missing_cols)}")

        df = add_age_bin(df)
        df['predicted_class'] = model.predict(df)

        try:
            df['predicted_proba'] = model.predict_proba(df)[:, 1]
        except Exception:
            pass

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(temp_file.name, index=False)
        temp_file.close()

        return send_file(temp_file.name, as_attachment=True, download_name="predictions.csv")

    return render_template("index.html", graphs=graphs)

if __name__ == "__main__":
    app.run(debug=True)
