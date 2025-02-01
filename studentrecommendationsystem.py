import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#Loading the Pre-trained Models
def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Load all models
weak_areas_model = load_model("weak_areas_model.pt")
improvements_model = load_model("improvements_model.pt")
performance_gaps_model = load_model("performance_gaps_model.pt")
recommendations_model = load_model("recommendations_model.pt")
student_persona_model = load_model("student_persona_model.pt")

#Define Prediction Functions
def predict_weak_areas(data):
    cluster = weak_areas_model.predict(data)
    labels = {0: 'Focus More', 1: 'Revise', 2: 'Strong'}
    return labels[cluster[0]]

def predict_improvements(data):
    cluster = improvements_model.predict(data)
    labels = {0: 'Declining', 1: 'Consistent', 2: 'Improving'}
    return labels[cluster[0]]

def predict_performance_gaps(data):
    cluster = performance_gaps_model.predict(data)
    labels = {0: 'High Performer', 1: 'Needs Improvement'}
    return labels[cluster[0]]

def predict_recommendations(data):
    cluster = recommendations_model.predict(data)
    labels = {0: 'Explore New Topics', 1: 'Needs Practice', 2: 'Mastery'}
    return labels[cluster[0]]

def predict_student_persona(data):
    cluster = student_persona_model.predict(data)
    labels = {0: 'Slow but Steady', 1: 'Balanced', 2: 'Fast Learner'}
    return labels[cluster[0]]

# Input Data from Real-world Scenario
# Replace this with real-world input data (e.g., from a user form or API)
real_world_data = {
    "weak_areas": [[15, 10, 5]],  # Example data for Weak Areas
    "improvements": [[85, 3]],   # Example data for Improvements
    "performance_gaps": [[3, 20, 100, 5]],  # Example data for Performance Gaps
    "recommendations": [[75, 2]],  # Example data for Recommendations
    "student_persona": [[120, 5, 0.3333]]  # Example data for Student Persona
}

#Making Predictions
weak_area_prediction = predict_weak_areas(real_world_data["weak_areas"])
improvement_prediction = predict_improvements(real_world_data["improvements"])
performance_gap_prediction = predict_performance_gaps(real_world_data["performance_gaps"])
recommendation_prediction = predict_recommendations(real_world_data["recommendations"])
student_persona_prediction = predict_student_persona(real_world_data["student_persona"])

# Display Results
print("Predictions:")
print(f"Weak Areas: {weak_area_prediction}")
print(f"Improvements: {improvement_prediction}")
print(f"Performance Gaps: {performance_gap_prediction}")
print(f"Recommendations: {recommendation_prediction}")
print(f"Student Persona: {student_persona_prediction}")
