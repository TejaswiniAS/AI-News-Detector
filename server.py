# server.py - AI News Detector Backend
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load the trained model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        news_text = request.json["text"]
        if not news_text.strip():
            return jsonify({"error": "No text provided"})
        vec = vectorizer.transform([news_text])
        prediction = model.predict(vec)[0]
        confidence = model.predict_proba(vec).max()
        output = "Fake News" if prediction == 0 else "Real News"
        return jsonify({"result": output, "confidence": f"{confidence*100:.2f}%"})
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)