from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import shap
import numpy as np
import json

app = FastAPI()

# CORS - allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and encoders
model = None
stream_encoder = None
district_encoder = None
stream_course_map = {}  # Map stream to valid courses
explainer = None  # SHAP explainer
model_metrics = {}  # Evaluation metrics
feature_names = ['Z-Score', 'Stream', 'District']

# Request model
class PredictionRequest(BaseModel):
    zscore: float
    stream: str
    district: str

def load_or_train_model():
    """Load model if exists, otherwise train and save it"""
    global model, stream_encoder, district_encoder, stream_course_map, explainer, model_metrics
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, 'dataset.csv')
    model_path = os.path.join(script_dir, 'model.pkl')
    encoders_path = os.path.join(script_dir, 'encoders.pkl')
    metrics_path = os.path.join(script_dir, 'metrics.json')
    
    # Build stream-course mapping (needed for validation) - do this first
    dataset_temp = pd.read_csv(dataset_path)
    stream_course_map = {}
    for _, row in dataset_temp.iterrows():
        stream = row['Stream']
        course = row['Matched_Course_University']
        if stream not in stream_course_map:
            stream_course_map[stream] = set()
        stream_course_map[stream].add(course)
    stream_course_map = {k: list(v) for k, v in stream_course_map.items()}
    print(f"Built stream-course mapping for {len(stream_course_map)} streams")
    
    # Try to load existing model
    if os.path.exists(model_path) and os.path.exists(encoders_path):
        print("Loading existing model...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
            stream_encoder = encoders['stream']
            district_encoder = encoders['district']
        
        # Create SHAP explainer from loaded model
        explainer = shap.TreeExplainer(model)
        
        # Load saved metrics
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                model_metrics = json.load(f)
            print(f"Metrics loaded: accuracy={model_metrics.get('accuracy', 'N/A')}")
        
        print("Model loaded successfully!")
        return
    
    # Train new model
    print("Training new model...")
    dataset = pd.read_csv(dataset_path)
    
    # Clean Zscore column
    dataset['Zscore'] = pd.to_numeric(dataset['Zscore'], errors='coerce')
    dataset = dataset.dropna(subset=['Zscore'])
    
    if len(dataset) == 0:
        raise ValueError("No valid data after cleaning")
    
    # Encode categorical columns
    stream_encoder = LabelEncoder()
    district_encoder = LabelEncoder()
    dataset['Stream'] = stream_encoder.fit_transform(dataset['Stream'])
    dataset['District'] = district_encoder.fit_transform(dataset['District'])
    
    # Prepare features and target
    X = dataset[['Zscore', 'Stream', 'District']]
    y = dataset['Matched_Course_University']
    
    # Split data for evaluation (stratify might fail with too many classes, so try/except)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        # If stratify fails (too many unique classes), use regular split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model with better parameters for accuracy
    model = RandomForestClassifier(
        n_estimators=100,      # Increased for better accuracy
        max_depth=25,           # Increased depth for better learning
        min_samples_split=5,    # Prevent overfitting
        min_samples_leaf=2,     # Prevent overfitting
        max_features='sqrt',    # Better feature selection
        random_state=42,
        n_jobs=-1,
        class_weight='balanced' # Handle class imbalance
    )
    model.fit(X_train, y_train)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    print("SHAP explainer created!")
    
    # ====== Comprehensive Evaluation ======
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Top-3 accuracy
    top3_accuracy = 0
    for i, true_label in enumerate(y_test):
        top3_indices = y_pred_proba[i].argsort()[-3:][::-1]
        top3_labels = [model.classes_[idx] for idx in top3_indices]
        if true_label in top3_labels:
            top3_accuracy += 1
    top3_accuracy = top3_accuracy / len(y_test)
    
    # Top-5 accuracy
    top5_accuracy = 0
    for i, true_label in enumerate(y_test):
        top5_indices = y_pred_proba[i].argsort()[-5:][::-1]
        top5_labels = [model.classes_[idx] for idx in top5_indices]
        if true_label in top5_labels:
            top5_accuracy += 1
    top5_accuracy = top5_accuracy / len(y_test)
    
    # Feature importances from RandomForest
    feature_importances = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in zip(feature_names, model.feature_importances_)
    ]
    feature_importances.sort(key=lambda x: x['importance'], reverse=True)
    
    # Classification report as dict (top 15 classes by support)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    per_class = []
    for cls_name, metrics in report.items():
        if cls_name in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        per_class.append({
            "class": cls_name,
            "precision": round(metrics['precision'], 3),
            "recall": round(metrics['recall'], 3),
            "f1": round(metrics['f1-score'], 3),
            "support": int(metrics['support'])
        })
    per_class.sort(key=lambda x: x['support'], reverse=True)
    
    # Dataset stats
    dataset_raw = pd.read_csv(dataset_path)
    dataset_raw['Zscore'] = pd.to_numeric(dataset_raw['Zscore'], errors='coerce')
    
    # Build and store all metrics
    model_metrics = {
        "accuracy": round(float(accuracy), 4),
        "top3_accuracy": round(float(top3_accuracy), 4),
        "top5_accuracy": round(float(top5_accuracy), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "f1_macro": round(float(f1_macro), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_split": 0.2,
        "num_classes": len(model.classes_),
        "feature_importances": feature_importances,
        "per_class_top15": per_class[:15],
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 25,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": 42
        },
        "dataset_info": {
            "total_rows": len(dataset_raw),
            "rows_after_cleaning": len(dataset),
            "num_features": 3,
            "features": ["Z-Score (numeric)", "Stream (categorical, label-encoded)", "District (categorical, label-encoded)"],
            "target": "Matched_Course_University",
            "num_streams": int(dataset_raw['Stream'].nunique()),
            "num_districts": int(dataset_raw['District'].nunique()),
            "num_courses": int(dataset_raw['Matched_Course_University'].nunique()),
            "zscore_min": round(float(dataset_raw['Zscore'].min()), 2),
            "zscore_max": round(float(dataset_raw['Zscore'].max()), 2),
            "zscore_mean": round(float(dataset_raw['Zscore'].mean()), 2),
            "preprocessing": [
                "Converted Z-Score to numeric (coerced errors to NaN)",
                "Dropped rows with missing Z-Score values",
                "Label-encoded Stream and District columns",
                "Applied class_weight='balanced' to handle class imbalance"
            ]
        }
    }
    
    # Save metrics
    with open(metrics_path, 'w') as f:
        json.dump(model_metrics, f, indent=2)
    
    print(f"Model Accuracy: {accuracy:.2%}")
    print(f"Top-3 Accuracy: {top3_accuracy:.2%}")
    print(f"Top-5 Accuracy: {top5_accuracy:.2%}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(encoders_path, 'wb') as f:
        pickle.dump({
            'stream': stream_encoder,
            'district': district_encoder
        }, f)
    
    print("Model trained, evaluated, and saved!")

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_or_train_model()

@app.get("/")
def read_root():
    return {"message": "Prediction API is running"}

@app.get("/api/options")
def get_options():
    """Get unique streams and districts"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, 'dataset.csv')
    
    dataset = pd.read_csv(dataset_path)
    districts = sorted(dataset['District'].unique().tolist())
    streams = sorted(dataset['Stream'].unique().tolist())
    
    return {"districts": districts, "streams": streams}

@app.get("/api/model-info")
def get_model_info():
    """Get model evaluation metrics, hyperparameters, and dataset info"""
    if not model_metrics:
        return {"error": "Metrics not available. Model may need retraining."}
    return model_metrics

@app.post("/api/predict")
def predict(request: PredictionRequest):
    """Make a prediction with confidence scores, filtered by stream compatibility"""
    if model is None or stream_encoder is None or district_encoder is None:
        return {"error": "Model not loaded"}
    
    try:
        # Encode inputs
        stream_encoded = stream_encoder.transform([request.stream])[0]
        district_encoded = district_encoder.transform([request.district])[0]
        
        # Make prediction with probabilities
        input_data = [[request.zscore, stream_encoded, district_encoded]]
        probabilities = model.predict_proba(input_data)[0]
        
        # Get valid courses for this stream
        valid_courses = set(stream_course_map.get(request.stream, []))
        
        # Filter predictions to only include valid courses for this stream
        valid_predictions = []
        for idx, course in enumerate(model.classes_):
            if course in valid_courses:
                valid_predictions.append({
                    "course": course,
                    "confidence": float(probabilities[idx]),
                    "index": idx
                })
        
        # Sort by confidence
        valid_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        if not valid_predictions:
            return {"error": f"No valid courses found for stream: {request.stream}"}
        
        # Get top prediction and top 3
        top_prediction = valid_predictions[0]
        top_3 = valid_predictions[:3]
        
        return {
            "prediction": top_prediction['course'],
            "confidence": top_prediction['confidence'],
            "top_3": [
                {"course": p['course'], "confidence": p['confidence']}
                for p in top_3
            ]
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/explain")
def explain_prediction(request: PredictionRequest):
    """Explain a prediction using SHAP values"""
    if model is None or explainer is None:
        return {"error": "Model or explainer not loaded"}
    
    try:
        # Encode inputs
        stream_encoded = stream_encoder.transform([request.stream])[0]
        district_encoded = district_encoder.transform([request.district])[0]
        
        input_data = pd.DataFrame(
            [[request.zscore, stream_encoded, district_encoded]],
            columns=['Zscore', 'Stream', 'District']
        )
        
        # Get prediction
        prediction = model.predict(input_data)[0]
        
        # Get SHAP values for this prediction
        shap_values = explainer.shap_values(input_data)
        
        # Find the class index for the predicted course
        class_idx = list(model.classes_).index(prediction)
        
        # shap_values shape: (samples, features, classes) for RandomForest
        sv = shap_values[0]  # first sample, shape: (features, classes)
        
        # Get SHAP values for the predicted class
        sv_for_class = sv[:, class_idx]  # shape: (features,)
        base = explainer.expected_value[class_idx]
        
        # Build feature contributions
        contributions = []
        raw_feature_values = [request.zscore, request.stream, request.district]
        for i, name in enumerate(feature_names):
            contributions.append({
                "feature": name,
                "value": str(raw_feature_values[i]),
                "shap_value": round(float(sv_for_class[i]), 6),
                "impact": "positive" if sv_for_class[i] > 0 else "negative"
            })
        
        # Sort by absolute SHAP value (most important first)
        contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        # Load dataset stats for comparative explanations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_dir, 'dataset.csv')
        dataset_full = pd.read_csv(dataset_path)
        dataset_full['Zscore'] = pd.to_numeric(dataset_full['Zscore'], errors='coerce')
        
        # Get stats for the predicted course
        course_data = dataset_full[dataset_full['Matched_Course_University'] == prediction]
        overall_mean = dataset_full['Zscore'].mean()
        course_mean = course_data['Zscore'].mean() if len(course_data) > 0 else overall_mean
        course_min = course_data['Zscore'].min() if len(course_data) > 0 else 0
        
        # How many students from this stream got this course
        stream_course = course_data[course_data['Stream'] == request.stream]
        stream_total = dataset_full[dataset_full['Stream'] == request.stream]
        
        # How many students from this district got this course
        district_course = course_data[course_data['District'] == request.district]
        district_total = dataset_full[dataset_full['District'] == request.district]
        
        # Percentage contributions
        total_shap = sum(abs(c['shap_value']) for c in contributions)
        
        simple_explanations = []
        for c in contributions:
            pct = round((abs(c['shap_value']) / total_shap) * 100) if total_shap > 0 else 33
            influence = "supports" if c['shap_value'] > 0 else "opposes"
            
            if c['feature'] == 'Z-Score':
                # Round both to 2 decimals before comparing to avoid "below 1.60" when input is 1.6
                zscore_rounded = round(request.zscore, 2)
                mean_rounded = round(course_mean, 2)
                if abs(zscore_rounded - mean_rounded) < 0.01:
                    comparison = f"right at the average ({mean_rounded:.2f})"
                elif zscore_rounded > mean_rounded:
                    comparison = f"above the average ({mean_rounded:.2f})"
                else:
                    comparison = f"below the average ({mean_rounded:.2f})"
                simple_explanations.append({
                    "icon": "📊",
                    "text": f"Your Z-Score ({request.zscore}) is {comparison} for this course (min cutoff: {course_min:.2f}). This {influence} the prediction ({pct}% influence)."
                })
            elif c['feature'] == 'Stream':
                # Re-filter stream data to be safe
                actual_stream = request.stream.strip()
                stream_in_course = course_data[course_data['Stream'].str.strip() == actual_stream]
                stream_count = len(stream_in_course)
                stream_pct = round((stream_count / len(course_data)) * 100) if len(course_data) > 0 else 0
                if stream_pct == 0:
                    stream_desc = f"No students from {actual_stream} were historically admitted to this course."
                else:
                    stream_desc = f"{stream_pct}% of students admitted to this course were from {actual_stream} ({stream_count} out of {len(course_data)})."
                simple_explanations.append({
                    "icon": "📚",
                    "text": f"{stream_desc} The model {influence} this prediction based on stream ({pct}% influence)."
                })
            elif c['feature'] == 'District':
                # Re-filter here with the actual input district to be safe
                actual_district = request.district.strip()
                dist_in_course = course_data[course_data['District'].str.strip() == actual_district]
                dist_count = len(dist_in_course)
                total_in_course = len(course_data)
                dist_pct = round((dist_count / total_in_course) * 100) if total_in_course > 0 else 0
                if dist_count == 0:
                    dist_desc = f"No students from {actual_district} were historically admitted to this course."
                else:
                    dist_desc = f"{dist_count} out of {total_in_course} students admitted to this course were from {actual_district} ({dist_pct}%)."
                simple_explanations.append({
                    "icon": "📍",
                    "text": f"{dist_desc} This {influence} the prediction ({pct}% influence)."
                })
        
        # Overall summary
        top = contributions[0]
        summary = f"Predicted {prediction.split('(')[0].strip()} based on your inputs. Your {top['feature']} had the most influence ({round((abs(top['shap_value'])/total_shap)*100)}% of the decision)."
        
        return {
            "predicted_course": prediction,
            "base_value": round(float(base), 6),
            "contributions": contributions,
            "simple_explanations": simple_explanations,
            "summary": summary
        }
    except Exception as e:
        return {"error": str(e)}

