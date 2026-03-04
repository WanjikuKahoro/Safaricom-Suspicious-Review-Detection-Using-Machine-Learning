import joblib
import sklearn

print("Using sklearn version:", sklearn.__version__)

# Load your existing trained model
model = joblib.load("model/model_clean.joblib")   # adjust if your file name differs

# Save it again in the new environment
joblib.dump(model, "model/model_clean.joblib")

print("Model successfully re-saved.")