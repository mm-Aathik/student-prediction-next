from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import shap
import numpy as np

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
feature_names = ['Z-Score', 'Stream', 'District']

# Request model
class PredictionRequest(BaseModel):
    zscore: float
    stream: str
    district: str

def load_or_train_model():
    """Load model if exists, otherwise train and save it"""
    global model, stream_encoder, district_encoder, stream_course_map, explainer
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, 'dataset.csv')
    model_path = os.path.join(script_dir, 'model.pkl')
    encoders_path = os.path.join(script_dir, 'encoders.pkl')
    
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
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2%}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Show top predictions accuracy
    y_pred_proba = model.predict_proba(X_test)
    top3_accuracy = 0
    for i, true_label in enumerate(y_test):
        top3_indices = y_pred_proba[i].argsort()[-3:][::-1]
        top3_labels = [model.classes_[idx] for idx in top3_indices]
        if true_label in top3_labels:
            top3_accuracy += 1
    top3_accuracy = top3_accuracy / len(y_test)
    print(f"Top-3 Accuracy: {top3_accuracy:.2%}")
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(encoders_path, 'wb') as f:
        pickle.dump({
            'stream': stream_encoder,
            'district': district_encoder
        }, f)
    
    print("Model trained and saved!")

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
            direction = "toward" if c['shap_value'] > 0 else "away from"
            
            if c['feature'] == 'Z-Score':
                diff = request.zscore - course_mean
                if diff >= 0:
                    comparison = f"above the average ({course_mean:.2f}) for this course"
                else:
                    comparison = f"below the average ({course_mean:.2f}) for this course"
                simple_explanations.append({
                    "icon": "📊",
                    "text": f"Z-Score {request.zscore} is {comparison}. Min cutoff in data: {course_min:.2f}. Contributes {pct}% of the decision, pushing {direction} this course."
                })
            elif c['feature'] == 'Stream':
                stream_count = len(stream_course)
                stream_pct = round((stream_count / len(course_data)) * 100) if len(course_data) > 0 else 0
                simple_explanations.append({
                    "icon": "📚",
                    "text": f"{stream_pct}% of students admitted to this course were from {c['value']}. Contributes {pct}% of the decision, pushing {direction} this course."
                })
            elif c['feature'] == 'District':
                dist_count = len(district_course)
                dist_pct = round((dist_count / len(course_data)) * 100) if len(course_data) > 0 else 0
                simple_explanations.append({
                    "icon": "📍",
                    "text": f"{dist_count} out of {len(course_data)} students admitted to this course were from {c['value']} ({dist_pct}%). Contributes {pct}% of the decision, pushing {direction} this course."
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

