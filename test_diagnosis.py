import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.diagnosis_model import HospitalDiagnosisModel

def test_diagnosis_prediction():
    # Create a model instance
    model = HospitalDiagnosisModel(model_type='random_forest')
    
    # Load the trained model
    model.load_model('models/diagnosis_model.joblib')
    
    # Test patient data
    patient_data = {
        'Age': 50,
        'Length_of_Stay': 5,
        'Department': 'ER',
        'Medical_Condition': 'Cardiac'
    }
    
    # Make prediction
    try:
        diagnosis, probability = model.predict(patient_data)
        print(f"Prediction successful!")
        print(f"Predicted Diagnosis: {diagnosis}")
        print(f"Probability: {probability:.2%}")
        return True
    except Exception as e:
        print(f"Prediction failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_diagnosis_prediction()
    if success:
        print("Test passed!")
    else:
        print("Test failed!")
