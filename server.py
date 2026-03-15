from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

try:
    model      = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print("✅ model.pkl and vectorizer.pkl loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ ERROR: {e}")
    model      = None
    vectorizer = None

@app.route("/health", methods=["GET"])
def health():
    status = "ok" if model and vectorizer else "error: model not loaded"
    return jsonify({"status": status})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON in request body."}), 400
    news_text = data.get("text", "").strip()
    if not news_text:
        return jsonify({"error": "No text provided."}), 400
    if len(news_text) > 5000:
        return jsonify({"error": "Text too long. Maximum 5000 characters."}), 400
    try:
        vec        = vectorizer.transform([news_text])
        prediction = model.predict(vec)[0]
        proba      = model.predict_proba(vec)[0]
        fake_prob  = proba[0] * 100
        real_prob  = proba[1] * 100
        result     = "Real News" if prediction == 1 else "Fake News"
        confidence = real_prob if prediction == 1 else fake_prob
        return jsonify({
            "result":       result,
            "confidence":   f"{confidence:.2f}%",
            "fake_percent": round(fake_prob, 1),
            "real_percent": round(real_prob, 1),
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
