import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils import load_and_preprocess_data

class SimpleDiagnosisModel:
    """
    A simple diagnosis model that doesn't rely on scikit-learn's feature names validation.
    """
    
    def __init__(self):
        """
        Initialize the diagnosis model.
        """
        self.diagnosis_probabilities = {
            'ER': {
                'Cardiac': {'Condition A': 0.3, 'Condition B': 0.3, 'Condition C': 0.4},
                'Neuro': {'Condition A': 0.4, 'Condition B': 0.3, 'Condition C': 0.3},
                'Injury': {'Condition A': 0.3, 'Condition B': 0.4, 'Condition C': 0.3}
            },
            'Ward': {
                'Cardiac': {'Condition A': 0.4, 'Condition B': 0.4, 'Condition C': 0.2},
                'Neuro': {'Condition A': 0.5, 'Condition B': 0.3, 'Condition C': 0.2},
                'Injury': {'Condition A': 0.3, 'Condition B': 0.4, 'Condition C': 0.3}
            },
            'ICU': {
                'Cardiac': {'Condition A': 0.2, 'Condition B': 0.3, 'Condition C': 0.5},
                'Neuro': {'Condition A': 0.2, 'Condition B': 0.3, 'Condition C': 0.5},
                'Injury': {'Condition A': 0.3, 'Condition B': 0.4, 'Condition C': 0.3}
            }
        }
        
        # Age and length of stay modifiers
        self.age_modifiers = {
            'young': {'Condition A': 1.2, 'Condition B': 0.9, 'Condition C': 0.9},
            'middle': {'Condition A': 1.0, 'Condition B': 1.1, 'Condition C': 0.9},
            'elderly': {'Condition A': 0.8, 'Condition B': 0.9, 'Condition C': 1.3}
        }
        
        self.los_modifiers = {
            'short': {'Condition A': 1.2, 'Condition B': 0.9, 'Condition C': 0.9},
            'medium': {'Condition A': 1.0, 'Condition B': 1.1, 'Condition C': 0.9},
            'long': {'Condition A': 0.8, 'Condition B': 0.9, 'Condition C': 1.3}
        }
    
    def train(self, data_path):
        """
        Train the model by calculating probabilities from data.
        
        Args:
            data_path (str): Path to the data file
            
        Returns:
            dict: Training results
        """
        # Load data
        df = load_and_preprocess_data(data_path)
        
        # Calculate probabilities for each combination
        for dept in df['Department'].unique():
            if dept not in self.diagnosis_probabilities:
                self.diagnosis_probabilities[dept] = {}
                
            dept_df = df[df['Department'] == dept]
            
            for cond in dept_df['Medical_Condition'].unique():
                if cond not in self.diagnosis_probabilities[dept]:
                    self.diagnosis_probabilities[dept][cond] = {}
                
                cond_df = dept_df[dept_df['Medical_Condition'] == cond]
                diagnosis_counts = cond_df['Diagnosis'].value_counts(normalize=True)
                
                for diagnosis, prob in diagnosis_counts.items():
                    self.diagnosis_probabilities[dept][cond][diagnosis] = prob
        
        # Calculate age modifiers
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 40, 60, 100], labels=['young', 'middle', 'elderly'])
        
        for age_group in df['AgeGroup'].unique():
            age_df = df[df['AgeGroup'] == age_group]
            diagnosis_counts = age_df['Diagnosis'].value_counts(normalize=True)
            
            # Calculate relative probabilities compared to overall
            overall_counts = df['Diagnosis'].value_counts(normalize=True)
            
            for diagnosis in diagnosis_counts.index:
                if diagnosis in overall_counts:
                    self.age_modifiers[age_group][diagnosis] = diagnosis_counts[diagnosis] / overall_counts[diagnosis]
        
        # Calculate length of stay modifiers
        df['LOSGroup'] = pd.cut(df['Length_of_Stay'], bins=[0, 5, 10, 15], labels=['short', 'medium', 'long'])
        
        for los_group in df['LOSGroup'].unique():
            los_df = df[df['LOSGroup'] == los_group]
            diagnosis_counts = los_df['Diagnosis'].value_counts(normalize=True)
            
            # Calculate relative probabilities compared to overall
            overall_counts = df['Diagnosis'].value_counts(normalize=True)
            
            for diagnosis in diagnosis_counts.index:
                if diagnosis in overall_counts:
                    self.los_modifiers[los_group][diagnosis] = diagnosis_counts[diagnosis] / overall_counts[diagnosis]
        
        return {
            'diagnosis_probabilities': self.diagnosis_probabilities,
            'age_modifiers': self.age_modifiers,
            'los_modifiers': self.los_modifiers
        }
    
    def predict(self, patient_data):
        """
        Predict diagnosis for a patient.
        
        Args:
            patient_data (dict): Patient data with Age, Length_of_Stay, Department, Medical_Condition
            
        Returns:
            tuple: (predicted_diagnosis, probability)
        """
        # Get patient attributes
        department = patient_data.get('Department', 'ER')
        condition = patient_data.get('Medical_Condition', 'Cardiac')
        age = patient_data.get('Age', 50)
        los = patient_data.get('Length_of_Stay', 5)
        
        # Determine age group
        if age < 40:
            age_group = 'young'
        elif age < 60:
            age_group = 'middle'
        else:
            age_group = 'elderly'
        
        # Determine length of stay group
        if los < 5:
            los_group = 'short'
        elif los < 10:
            los_group = 'medium'
        else:
            los_group = 'long'
        
        # Get base probabilities
        if department in self.diagnosis_probabilities and condition in self.diagnosis_probabilities[department]:
            base_probs = self.diagnosis_probabilities[department][condition]
        else:
            # Default probabilities if combination not found
            base_probs = {'Condition A': 0.33, 'Condition B': 0.33, 'Condition C': 0.34}
        
        # Apply modifiers
        modified_probs = {}
        for diagnosis, prob in base_probs.items():
            age_modifier = self.age_modifiers.get(age_group, {}).get(diagnosis, 1.0)
            los_modifier = self.los_modifiers.get(los_group, {}).get(diagnosis, 1.0)
            
            modified_probs[diagnosis] = prob * age_modifier * los_modifier
        
        # Normalize probabilities
        total = sum(modified_probs.values())
        if total > 0:
            for diagnosis in modified_probs:
                modified_probs[diagnosis] /= total
        
        # Get most likely diagnosis
        predicted_diagnosis = max(modified_probs, key=modified_probs.get)
        probability = modified_probs[predicted_diagnosis]
        
        return predicted_diagnosis, probability
    
    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'diagnosis_probabilities': self.diagnosis_probabilities,
            'age_modifiers': self.age_modifiers,
            'los_modifiers': self.los_modifiers
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a model from a file.
        
        Args:
            filepath (str): Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.diagnosis_probabilities = model_data['diagnosis_probabilities']
        self.age_modifiers = model_data['age_modifiers']
        self.los_modifiers = model_data['los_modifiers']
        
        print(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Create and train model
    model = SimpleDiagnosisModel()
    results = model.train('../data/hospital_patient_dataset.csv')
    
    # Save model
    model.save_model('../models/simple_diagnosis_model.joblib')
    
    # Test prediction
    patient_data = {
        'Age': 50,
        'Length_of_Stay': 5,
        'Department': 'ER',
        'Medical_Condition': 'Cardiac'
    }
    
    diagnosis, probability = model.predict(patient_data)
    print(f"Predicted Diagnosis: {diagnosis}")
    print(f"Probability: {probability:.2%}")
