import os
import time
import tempfile
import json
import numpy as np
import pandas as pd
import librosa
import torch
import joblib
from flask import Flask, request, jsonify, render_template, send_from_directory

# Initialize Flask
app = Flask(__name__, static_folder="static", template_folder="templates")

# Paths and Constants
TARGET_SR = 16000
DURATION = 10.0
DATA_DIR = os.path.join("data", "AUDIOS MACHINE LEARNING")
CACHE = os.path.join("outputs", "embeddings_v2.npz")
REPORTE = os.path.join("outputs", "reporte_filtrado_v2.csv")
MODEL_METRICS_PATH = os.path.join("outputs", "model_metrics.json")
EXPERIMENT_HISTORY_PATH = os.path.join("outputs", "experiment_history.json")

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Webapp running on device: {DEVICE}")

# Global model references
loaded_models = {}
processor = None
wav2vec2_model = None

# Model mapping for friendly names
MODEL_MAPPING = {
    "knn_k3": "KNN (k=3)",
    "knn_k5": "KNN (k=5)",
    "svm_lineal": "SVM lineal",
    "svm_rbf": "SVM RBF",
    "logreg": "Regresión Logística",
    "rf": "Random Forest"
}

ACTIVE_CLASSES = ["Enojo", "Tristeza", "Feliz"]
TOP_N_DATASET = 10
SCORE_COLUMNS = {
    "Enojo": "score_enojo",
    "Tristeza": "score_tristeza",
    "Feliz": "score_feliz",
}
CLASS_STYLES = {
    "Enojo": {"slug": "enojo", "label": "Enojo", "accent": "#ef4444", "accent_dark": "#b91c1c"},
    "Tristeza": {"slug": "tristeza", "label": "Tristeza", "accent": "#06b6d4", "accent_dark": "#0891b2"},
    "Feliz": {"slug": "feliz", "label": "Feliz", "accent": "#f59e0b", "accent_dark": "#d97706"},
}


def get_class_style(clase):
    return CLASS_STYLES.get(
        clase,
        {
            "slug": clase.lower(),
            "label": clase,
            "accent": "#a855f7",
            "accent_dark": "#7e22ce",
        },
    )


def load_model_metrics():
    if not os.path.exists(MODEL_METRICS_PATH):
        return {}
    with open(MODEL_METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_experiment_history():
    if not os.path.exists(EXPERIMENT_HISTORY_PATH):
        return {"default_experiment_id": None, "experiments": []}
    with open(EXPERIMENT_HISTORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def init_models():
    global loaded_models, processor, wav2vec2_model
    
    # 1. Load pre-trained models from joblib files
    model_dir = os.path.join("outputs", "modelos")
    print(f"Loading pre-trained classifiers from {model_dir}...")
    for model_key in MODEL_MAPPING.keys():
        path = os.path.join(model_dir, f"{model_key}.joblib")
        if os.path.exists(path):
            loaded_models[model_key] = joblib.load(path)
            print(f"  Loaded model: {MODEL_MAPPING[model_key]}")
        else:
            print(f"  WARNING: Model binary not found at {path}")
            
    # 2. Load wav2vec2 for inference
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    print("Loading facebook/wav2vec2-base...")
    t0 = time.time()
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE)
    wav2vec2_model.eval()
    print(f"wav2vec2 loaded in {time.time() - t0:.1f}s")

@app.route("/")
def index():
    metrics = load_model_metrics()
    experiment_history = load_experiment_history()
    model_metrics = metrics.get("models", {})
    best_model_key = metrics.get("best_model", "svm_lineal")
    best_model_label = MODEL_MAPPING.get(best_model_key, best_model_key)
    best_bal_acc = model_metrics.get(best_model_key, {}).get("balanced_accuracy")

    return render_template(
        "index.html",
        active_classes=ACTIVE_CLASSES,
        class_styles={clase: get_class_style(clase) for clase in ACTIVE_CLASSES},
        top_n_dataset=TOP_N_DATASET,
        model_mapping=MODEL_MAPPING,
        model_metrics=model_metrics,
        best_model_key=best_model_key,
        best_model_label=best_model_label,
        best_bal_acc=best_bal_acc,
        experiment_history=experiment_history,
    )

@app.route("/data/audio/<clase>/<filename>")
def serve_audio(clase, filename):
    directory = os.path.join(DATA_DIR, clase)
    return send_from_directory(directory, filename)

@app.route("/outputs/figuras/<path:filename>")
def serve_figuras(filename):
    return send_from_directory(os.path.join("outputs", "figuras"), filename)

@app.route("/api/dataset", methods=["GET"])
def get_dataset():
    if not os.path.exists(REPORTE):
        return jsonify({"error": "Report CSV not found"}), 404
        
    df_rep = pd.read_csv(REPORTE)
    
    frames = []
    for clase in ACTIVE_CLASSES:
        score_col = SCORE_COLUMNS.get(clase)
        if not score_col or score_col not in df_rep.columns:
            continue
        frames.append(
            df_rep[df_rep["clase_original"] == clase]
            .sort_values(score_col, ascending=False)
            .head(TOP_N_DATASET)
        )

    if not frames:
        return jsonify({"error": "No active classes available in report"}), 500

    df_balanced = pd.concat(frames, ignore_index=True)
    
    records = []
    for _, row in df_balanced.iterrows():
        archivo = row["archivo"]
        clase = row["clase_original"]
        score_col = SCORE_COLUMNS.get(clase)
        score = row[score_col] if score_col in row else None
        recolector = archivo[:2]
        style = get_class_style(clase)
        
        records.append({
            "archivo": archivo,
            "clase": clase,
            "slug": style["slug"],
            "recolector": recolector,
            "score": float(score) if score is not None else 0.0,
            "url": f"/data/audio/{clase}/{archivo}"
        })
        
    return jsonify(records)

def extract_features_and_predict(y, sr, model_key="svm_lineal"):
    global loaded_models, processor, wav2vec2_model
    
    # Fallback to svm_lineal if key is invalid
    if model_key not in loaded_models:
        model_key = "svm_lineal" if "svm_lineal" in loaded_models else list(loaded_models.keys())[0]
        
    clf = loaded_models[model_key]
    
    # 1. Extract wav2vec2 embedding
    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = wav2vec2_model(inputs.input_values.to(DEVICE))
    emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype(np.float32)
    
    # 2. Predict with selected model
    prediction = clf.predict([emb])[0]
    probs = clf.predict_proba([emb])[0]
    classes = clf.classes_
    
    prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
    
    # 3. Extract acoustic metrics for display
    # Pitch estimation using pyin (bounded for speech)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=50, fmax=400, sr=sr
    )
    f0_mean = float(np.nanmean(f0)) if not np.all(np.isnan(f0)) else 0.0
    f0_std = float(np.nanstd(f0)) if not np.all(np.isnan(f0)) else 0.0
    
    # RMS/Energy
    rms = librosa.feature.rms(y=y)
    rms_mean = float(np.mean(rms))
    rms_db = float(librosa.amplitude_to_db(np.array([rms_mean]))[0])
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y)
    zcr_mean = float(np.mean(zcr))
    
    return {
        "prediction": prediction,
        "probabilities": prob_dict,
        "metrics": {
            "duration": float(len(y) / sr),
            "pitch_mean_hz": round(f0_mean, 1),
            "pitch_std_hz": round(f0_std, 1),
            "energy_db": round(rms_db, 1),
            "zcr": round(zcr_mean, 4)
        }
    }

@app.route("/api/predict_dataset", methods=["POST"])
def predict_dataset():
    data = request.json
    clase = data.get("clase")
    archivo = data.get("archivo")
    model_key = data.get("model", "svm_lineal")
    
    if not clase or not archivo:
        return jsonify({"error": "Missing parameters"}), 400
        
    ruta = os.path.join(DATA_DIR, clase, archivo)
    if not os.path.exists(ruta):
        return jsonify({"error": "Audio file not found"}), 404
        
    try:
        y, sr = librosa.load(ruta, sr=TARGET_SR, mono=True, duration=DURATION)
        result = extract_features_and_predict(y, sr, model_key)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict_upload", methods=["POST"])
def predict_upload():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
        
    audio_file = request.files["audio"]
    model_key = request.form.get("model", "svm_lineal")
    
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
        
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_path = temp_audio.name
            audio_file.save(temp_path)
            
        try:
            y, sr = librosa.load(temp_path, sr=TARGET_SR, mono=True, duration=DURATION)
            if len(y) < TARGET_SR * 0.3:
                return jsonify({"error": "Audio is too short (minimum 0.3 seconds)"}), 400
                
            result = extract_features_and_predict(y, sr, model_key)
            return jsonify(result)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    init_models()
    app.run(host="0.0.0.0", port=5001, debug=True)
