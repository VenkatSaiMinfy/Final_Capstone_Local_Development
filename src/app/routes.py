# src/app/routes.py

import os
import pandas as pd
from flask import Blueprint, request, render_template, jsonify
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────
# Import utility functions for prediction and CSV handling
# ─────────────────────────────────────────────
from .utils.prediction import predict_lead, predict_batch
from .utils.upload import handle_csv_upload

# ─────────────────────────────────────────────
# Initialize blueprint and configuration
# ─────────────────────────────────────────────
bp = Blueprint("routes", __name__)
ALLOWED_EXTENSIONS = {"csv"}


def allowed_file(filename: str) -> bool:
    """
    Check if the uploaded file has an allowed extension.
    
    Args:
        filename (str): Name of the file to check.
    
    Returns:
        bool: True if extension is allowed, False otherwise.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route("/", methods=["GET"])
def index():
    """
    Render the home page with forms for:
      - Single-lead JSON prediction
      - Batch CSV upload for bulk predictions
    """
    return render_template("index.html")


@bp.route("/predict", methods=["POST"])
def predict():
    """
    Handle single-lead prediction.
    
    Expects:
      - JSON payload in request body with lead feature key/value pairs.
    
    Returns:
      - JSON with { "conversion_probability": <float> }
      - 400 on invalid JSON
      - 500 on prediction error
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400

    try:
        proba = predict_lead(data)
        return jsonify({"conversion_probability": proba})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/upload", methods=["POST"])
def upload():
    """
    Handle batch predictions via CSV upload.
    
    Workflow:
      1) Validate file presence and extension.
      2) Save uploaded file to local 'uploads/' directory.
      3) Read CSV into DataFrame and validate contents.
      4) Persist raw CSV data to 'uploaded_leads' table.
      5) Generate batch predictions and append to DataFrame.
      6) Return JSON list of records with predictions.
    
    Returns:
      - JSON with { "predictions": [ {<row>..., "prediction": 0|1}, ... ] }
      - 400 on missing file, empty file, or wrong extension
      - 500 on server or processing errors
    """
    # 1) Check file part
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # 2) Validate extension and save
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Only .csv allowed"}), 400

    filename = secure_filename(file.filename)
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    upload_path = os.path.join(upload_dir, filename)
    file.save(upload_path)

    try:
        # 3) Load CSV into DataFrame
        df = pd.read_csv(upload_path)
        if df.empty:
            return jsonify({"error": "Uploaded file is empty."}), 400

        # 4) Save raw data to Postgres for monitoring/auditing
        handle_csv_upload(upload_path, table_name="uploaded_leads")

        # 5) Generate batch predictions
        predictions = predict_batch(df)
        df["prediction"] = predictions

        # 6) Prepare JSON-serializable output
        response_data = df.where(pd.notnull(df), None).to_dict(orient="records")
        return jsonify({"predictions": response_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
