from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import os
import numpy as np
import pandas as pd

from data_utils import allowed_file_extension, validate_dataset, save_uploaded_file
from model_utils import preprocess_data
from model_train import train_autoencoder_from_npy
from model_detect import detect_anomalies

app = Flask(__name__)
app.secret_key = "secret123"

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "model"
RESULTS_FOLDER = os.path.join("static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "POST":
        file = request.files.get("dataset")
        if not file or file.filename == "":
            flash("No file selected.", "warning")
            return redirect(url_for("train"))
        if not allowed_file_extension(file.filename):
            flash("File type not allowed. Please upload .csv or .xlsx.", "danger")
            return redirect(url_for("train"))

        saved_path = save_uploaded_file(file, UPLOAD_FOLDER)
        valid, df_clean, message = validate_dataset(saved_path)
        if not valid:
            flash(f"Validation failed: {message}", "danger")
            return redirect(url_for("train"))

        try:
            X_scaled, scaler = preprocess_data(df_clean, save_scaler=True)
            npy_path = os.path.join(UPLOAD_FOLDER, "X_train_scaled.npy")
            np.save(npy_path, X_scaled)
            flash("✅ Dataset preprocessed successfully! Training started...", "info")

            train_info = train_autoencoder_from_npy(
                npy_path,
                model_dir=MODEL_FOLDER,
                results_dir=RESULTS_FOLDER,
                epochs=50,
                batch_size=32,
                encoding_dim=8,
                patience=6,
                verbose=1
            )
            flash(f"✅ Training complete! Model saved.", "success")
            return redirect(url_for("results"))

        except Exception as e:
            flash(f"❌ Error during preprocessing or training: {e}", "danger")
            return redirect(url_for("train"))

    return render_template("train.html")


@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "POST":
        file = request.files.get("testdata")
        if not file or file.filename == "":
            flash("No file selected.", "warning")
            return redirect(url_for("detect"))
        if not allowed_file_extension(file.filename):
            flash("File type not allowed. Please upload .csv or .xlsx.", "danger")
            return redirect(url_for("detect"))

        saved_path = save_uploaded_file(file, UPLOAD_FOLDER)
        valid, df_clean, message = validate_dataset(saved_path)
        if not valid:
            flash(f"Validation failed: {message}", "danger")
            return redirect(url_for("detect"))

        cleaned_path = os.path.join(UPLOAD_FOLDER, "test_cleaned.csv")
        df_clean.to_csv(cleaned_path, index=False)

        try:
            
            detect_info = detect_anomalies(
                test_path=cleaned_path,
                model_dir=MODEL_FOLDER,
                results_dir=RESULTS_FOLDER
            )

            # Save detection summary
            summary_path = os.path.join(RESULTS_FOLDER, "detection_summary.txt")
            with open(summary_path, "w") as f:
                for k, v in detect_info.items():
                    if k != "top_anomalies":
                        f.write(f"{k}: {v}\n")

            flash(f"✅ Detection complete! {detect_info['n_anomalies']} anomalies found.", "success")
            # Prepare the results for the template
            template_vars = {
                'summary_text': open(summary_path, 'r', encoding='utf-8').read(),
                'detect_info': detect_info
            }
            
            # Add optional files if they exist
            if 'error_plot' in detect_info and os.path.exists(detect_info['error_plot']):
                template_vars['error_hist_file'] = os.path.relpath(detect_info['error_plot'], 'static')
            if 'results_csv' in detect_info and os.path.exists(detect_info['results_csv']):
                template_vars['csv_file'] = os.path.relpath(detect_info['results_csv'], 'static')
            if 'time_series_plot' in detect_info and os.path.exists(detect_info['time_series_plot']):
                template_vars['time_series_plot'] = os.path.relpath(detect_info['time_series_plot'], 'static')
            
            return render_template("results.html", **template_vars)

        except Exception as e:
            flash(f"❌ Detection failed: {e}", "danger")
            return redirect(url_for("detect"))

    return render_template("detect.html")


@app.route("/results")
def results():
    # --- Summary file ---
    summary_path = os.path.join(RESULTS_FOLDER, "detection_summary.txt")
    summary_text = open(summary_path).read() if os.path.exists(summary_path) else None

    # --- Reconstruction error histogram ---
    error_hist_file_path = os.path.join(RESULTS_FOLDER, "error_hist.png")
    error_hist_file = "results/error_hist.png" if os.path.exists(error_hist_file_path) else None

    # --- Full CSV results ---
    csv_file_path = os.path.join(RESULTS_FOLDER, "detection_results.csv")
    csv_file = "results/detection_results.csv" if os.path.exists(csv_file_path) else None

    # --- Top anomalies ---
    top_csv_path = os.path.join(RESULTS_FOLDER, "top_anomalies.csv")
    detect_info = None
    if os.path.exists(top_csv_path):
        top_df = pd.read_csv(top_csv_path)
        detect_info = {
            "top_anomalies": top_df.to_dict(orient="records")
        }

        # Calculate anomaly percentage
        anomaly_count = None
        total_count = None
        if summary_text:
            for line in summary_text.splitlines():
                if "Anomalies Detected" in line:
                    anomaly_count = int(line.split(":")[1].strip())
                    detect_info["n_anomalies"] = anomaly_count
                if "Total Records" in line:
                    total_count = int(line.split(":")[1].strip())
                    detect_info["total_samples"] = total_count
                    if anomaly_count is not None and total_count > 0:
                        detect_info["anomaly_percentage"] = (anomaly_count / total_count) * 100

    return render_template(
        "results.html",
        summary_text=summary_text,
        error_hist_file=error_hist_file,
        csv_file=csv_file,
        detect_info=detect_info
    )


@app.route('/download_results')
def download_results():
    csv_path = os.path.join(RESULTS_FOLDER, "detection_results.csv")
    if os.path.exists(csv_path):
        return send_file(
            csv_path,
            as_attachment=True,
            download_name="ddos_detection_results.csv",
            mimetype='text/csv'
        )
    flash("Results file not found. Please run detection first.", "error")
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
