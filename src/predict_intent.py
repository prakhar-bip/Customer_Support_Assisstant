"""
Reproducible Inference Engine for Classical ML Intent Classifier.

Provides predict_intent() function and CLI interface using the trained
TF-IDF + Logistic Regression pipeline.
"""

import argparse
import json
import os
import sys
import joblib

DEFAULT_MODEL_PATH = "models/intent_tfidf_logistic_regression.joblib"

_cached_model = None

def get_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load and cache the trained scikit-learn pipeline."""
    global _cached_model
    if _cached_model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                f"Please run 'python src/train_classical_baseline.py' first."
            )
        _cached_model = joblib.load(model_path)
    return _cached_model

def predict_intent(text: str, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """
    Predict customer support intent for an incoming message.

    Args:
        text (str): Raw customer query.
        model_path (str): Path to serialized pipeline artifact.

    Returns:
        dict: {
            "text": input text,
            "predicted_intent": top predicted intent,
            "confidence": float probability in [0, 1],
            "all_probabilities": dict of intent -> probability sorted descending
        }
    """
    if not text or not isinstance(text, str) or not text.strip():
        return {
            "text": text,
            "predicted_intent": "UNKNOWN",
            "confidence": 0.0,
            "all_probabilities": {}
        }

    model = get_model(model_path)
    probs = model.predict_proba([text])[0]
    classes = model.classes_

    # Sort probabilities
    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    top_intent, top_conf = ranked[0]

    return {
        "text": text.strip(),
        "predicted_intent": top_intent,
        "confidence": round(float(top_conf), 4),
        "all_probabilities": {c: round(float(p), 4) for c, p in ranked}
    }

def main():
    parser = argparse.ArgumentParser(description="Classify customer support intent using Classical ML baseline.")
    parser.add_argument("--text", type=str, required=True, help="Customer message text to classify.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained model artifact.")
    parser.add_argument("--json", action="store_true", help="Output raw JSON.")
    args = parser.parse_args()

    try:
        result = predict_intent(args.text, model_path=args.model)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("=" * 60)
            print(f"Customer Text:     {result['text']}")
            print(f"Predicted Intent:  {result['predicted_intent']}")
            print(f"Confidence:        {result['confidence'] * 100:.2f}%")
            print("-" * 60)
            print("Top Intent Probabilities:")
            for intent, p in list(result["all_probabilities"].items())[:3]:
                print(f"  {intent:<38} : {p * 100:>6.2f}%")
            print("=" * 60)
    except Exception as e:
        print(f"Inference Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
