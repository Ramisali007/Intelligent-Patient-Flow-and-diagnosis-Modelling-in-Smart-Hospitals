
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils import load_and_preprocess_data

class HospitalDiagnosisModel:
    """
    A machine learning model for predicting patient diagnoses based on various features.
    """

    def __init__(self, model_type='random_forest'):
        """
        Initialize the diagnosis model.

        Args:
            model_type (str): Type of model to use ('random_forest', 'logistic', 'svm')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.feature_importances = None
        self.categorical_values = {
            'Department': ['ER', 'Ward', 'ICU'],
            'Medical_Condition': ['Cardiac', 'Neuro', 'Injury']
        }

    def preprocess_data(self, df):
        """
        Preprocess the data for model training.

        Args:
            df (pd.DataFrame): Raw dataframe

        Returns:
            tuple: (X, y, feature_names)
        """
        # Create a copy to avoid modifying the original
        df_processed = df.copy()

        # Encode the target variable
        if 'Diagnosis' in df_processed.columns:
            df_processed['Diagnosis'] = self.label_encoder.fit_transform(df_processed['Diagnosis'])

        # One-hot encode categorical variables
        categorical_cols = ['Department', 'Medical_Condition']
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols)

        # Scale numerical features
        numerical_cols = ['Age', 'Length_of_Stay']
        if all(col in df_processed.columns for col in numerical_cols):
            df_processed[numerical_cols] = self.scaler.fit_transform(df_processed[numerical_cols])

        # Split features and target
        if 'Diagnosis' in df_processed.columns:
            X = df_processed.drop('Diagnosis', axis=1)
            y = df_processed['Diagnosis']
        else:
            X = df_processed
            y = None

        # Store feature names
        self.feature_names = X.columns.tolist()

        return X, y, self.feature_names

    def train(self, data_path, test_size=0.2, random_state=42, tune_hyperparams=False):
        """
        Train the diagnosis model.

        Args:
            data_path (str): Path to the data file
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            tune_hyperparams (bool): Whether to tune hyperparameters using GridSearchCV

        Returns:
            dict: Training results including accuracy and classification report
        """
        # Load and preprocess data
        df = load_and_preprocess_data(data_path)
        X, y, _ = self.preprocess_data(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Select model
        if self.model_type == 'random_forest':
            if tune_hyperparams:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                self.model = GridSearchCV(RandomForestClassifier(random_state=random_state),
                                         param_grid, cv=5, n_jobs=-1)
            else:
                self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)

        elif self.model_type == 'logistic':
            if tune_hyperparams:
                param_grid = {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga'],
                    'penalty': ['l1', 'l2']
                }
                self.model = GridSearchCV(LogisticRegression(random_state=random_state, max_iter=1000),
                                         param_grid, cv=5, n_jobs=-1)
            else:
                self.model = LogisticRegression(random_state=random_state, max_iter=1000)

        elif self.model_type == 'svm':
            if tune_hyperparams:
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.1, 0.01],
                    'kernel': ['rbf', 'linear']
                }
                self.model = GridSearchCV(SVC(random_state=random_state, probability=True),
                                         param_grid, cv=5, n_jobs=-1)
            else:
                self.model = SVC(random_state=random_state, probability=True)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Train model
        self.model.fit(X_train, y_train)

        # If GridSearchCV was used, get the best model
        if tune_hyperparams:
            print(f"Best parameters: {self.model.best_params_}")
            self.model = self.model.best_estimator_

        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Get feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances = dict(zip(self.feature_names, self.model.feature_importances_))

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5)

        # Return results
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importances': self.feature_importances
        }

        return results

    def predict(self, patient_data):
        """
        Predict diagnosis for a patient.

        Args:
            patient_data (dict or pd.DataFrame): Patient data

        Returns:
            tuple: (predicted_diagnosis, probability)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Convert dict to DataFrame if necessary
        if isinstance(patient_data, dict):
            patient_data = pd.DataFrame([patient_data])

        # Get the department and medical condition
        department = patient_data['Department'].iloc[0] if 'Department' in patient_data.columns else None
        medical_condition = patient_data['Medical_Condition'].iloc[0] if 'Medical_Condition' in patient_data.columns else None

        # Get numerical features
        age = patient_data['Age'].iloc[0] if 'Age' in patient_data.columns else 0
        length_of_stay = patient_data['Length_of_Stay'].iloc[0] if 'Length_of_Stay' in patient_data.columns else 0

        # Scale numerical features
        scaled_age, scaled_los = self.scaler.transform([[age, length_of_stay]])[0]

        # Create a dictionary to hold all features with default value 0
        feature_dict = {feature: 0 for feature in self.feature_names}

        # Set numerical features
        if 'Age' in self.feature_names:
            feature_dict['Age'] = scaled_age
        if 'Length_of_Stay' in self.feature_names:
            feature_dict['Length_of_Stay'] = scaled_los

        # Set categorical features
        if department is not None:
            dept_col = f'Department_{department}'
            if dept_col in self.feature_names:
                feature_dict[dept_col] = 1

        if medical_condition is not None:
            cond_col = f'Medical_Condition_{medical_condition}'
            if cond_col in self.feature_names:
                feature_dict[cond_col] = 1

        # Convert to DataFrame
        X = pd.DataFrame([feature_dict])

        # Ensure columns are in the right order
        X = X[self.feature_names]

        # Make prediction
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        # Convert prediction back to original label
        diagnosis = self.label_encoder.inverse_transform([prediction])[0]
        probability = probabilities[prediction]

        return diagnosis, probability

    def save_model(self, filepath):
        """
        Save the trained model to a file.

        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Get all possible values for categorical columns from the feature names
        categorical_values = {}
        for feature in self.feature_names:
            if '_' in feature:
                col, val = feature.split('_', 1)
                if col in ['Department', 'Medical_Condition']:
                    if col not in categorical_values:
                        categorical_values[col] = []
                    categorical_values[col].append(val)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances,
            'model_type': self.model_type,
            'categorical_values': categorical_values
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a trained model from a file.

        Args:
            filepath (str): Path to the saved model
        """
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.feature_importances = model_data['feature_importances']
        self.model_type = model_data['model_type']

        # Load categorical values if available
        if 'categorical_values' in model_data:
            self.categorical_values = model_data['categorical_values']
        else:
            # Extract from feature names if not available
            self.categorical_values = {}
            for feature in self.feature_names:
                if '_' in feature:
                    col, val = feature.split('_', 1)
                    if col in ['Department', 'Medical_Condition']:
                        if col not in self.categorical_values:
                            self.categorical_values[col] = []
                        self.categorical_values[col].append(val)

        print(f"Model loaded from {filepath}")

    def plot_feature_importance(self, top_n=10):
        """
        Plot feature importance.

        Args:
            top_n (int): Number of top features to show

        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if self.feature_importances is None:
            raise ValueError("Feature importances not available. Train a model that supports feature importances.")

        # Sort features by importance
        sorted_features = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_names = [f[0] for f in top_features]
        feature_values = [f[1] for f in top_features]

        sns.barplot(x=feature_values, y=feature_names, ax=ax)
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')

        return fig

    def plot_confusion_matrix(self, data_path, test_size=0.2, random_state=42):
        """
        Plot confusion matrix.

        Args:
            data_path (str): Path to the data file
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility

        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Load and preprocess data
        df = load_and_preprocess_data(data_path)
        X, y, _ = self.preprocess_data(df)

        # Split data
        _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Get class labels
        class_labels = self.label_encoder.inverse_transform(range(len(self.label_encoder.classes_)))

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        return fig

# Example usage
if __name__ == "__main__":
    # Create and train model
    model = HospitalDiagnosisModel(model_type='random_forest')
    results = model.train('../data/hospital_patient_dataset.csv', tune_hyperparams=False)

    # Print results
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Cross-validation mean accuracy: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
    print("\nClassification Report:")
    for class_name, metrics in results['classification_report'].items():
        if isinstance(metrics, dict):
            print(f"{class_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1-score={metrics['f1-score']:.4f}")

    # Save model
    model.save_model('../models/diagnosis_model.joblib')

    # Plot feature importance
    fig = model.plot_feature_importance(top_n=10)
    plt.tight_layout()
    plt.savefig('../models/feature_importance.png')

    # Plot confusion matrix
    fig = model.plot_confusion_matrix('../data/hospital_patient_dataset.csv')
    plt.tight_layout()
    plt.savefig('../models/confusion_matrix.png')
