import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

class BayesianPatientModel:
    """
    A simplified Bayesian Network model for patient diagnosis and treatment.

    This model uses a simplified approach to represent probabilistic relationships
    between patient symptoms, conditions, and treatments.
    """

    def __init__(self):
        """Initialize the Bayesian Network model."""
        # Define the network structure
        self.edges = [
            ('Age', 'HeartDisease'),
            ('Age', 'Diabetes'),
            ('Gender', 'HeartDisease'),
            ('Smoking', 'HeartDisease'),
            ('Smoking', 'LungDisease'),
            ('HeartDisease', 'ChestPain'),
            ('HeartDisease', 'Fatigue'),
            ('HeartDisease', 'ECGAbnormal'),
            ('LungDisease', 'Cough'),
            ('LungDisease', 'Fatigue'),
            ('LungDisease', 'ShortnessOfBreath'),
            ('Diabetes', 'Fatigue'),
            ('Diabetes', 'ExcessiveThirst'),
            ('ChestPain', 'EmergencyCare'),
            ('ECGAbnormal', 'EmergencyCare'),
            ('ShortnessOfBreath', 'EmergencyCare'),
            ('HeartDisease', 'CardiacMedication'),
            ('LungDisease', 'RespiratoryMedication'),
            ('Diabetes', 'InsulinTherapy')
        ]

        # Define nodes
        self.nodes = set([node for edge in self.edges for node in edge])

        # Define simple conditional probabilities
        self.probabilities = {
            # Prior probabilities
            'Age': {'Young': 0.3, 'Middle': 0.5, 'Elderly': 0.2},
            'Gender': {'Male': 0.5, 'Female': 0.5},
            'Smoking': {'No': 0.7, 'Yes': 0.3},

            # Disease probabilities given demographics
            'HeartDisease|Age=Young,Gender=Male,Smoking=No': {'No': 0.9, 'Yes': 0.1},
            'HeartDisease|Age=Young,Gender=Male,Smoking=Yes': {'No': 0.8, 'Yes': 0.2},
            'HeartDisease|Age=Young,Gender=Female,Smoking=No': {'No': 0.95, 'Yes': 0.05},
            'HeartDisease|Age=Young,Gender=Female,Smoking=Yes': {'No': 0.85, 'Yes': 0.15},
            'HeartDisease|Age=Middle,Gender=Male,Smoking=No': {'No': 0.8, 'Yes': 0.2},
            'HeartDisease|Age=Middle,Gender=Male,Smoking=Yes': {'No': 0.6, 'Yes': 0.4},
            'HeartDisease|Age=Middle,Gender=Female,Smoking=No': {'No': 0.85, 'Yes': 0.15},
            'HeartDisease|Age=Middle,Gender=Female,Smoking=Yes': {'No': 0.7, 'Yes': 0.3},
            'HeartDisease|Age=Elderly,Gender=Male,Smoking=No': {'No': 0.7, 'Yes': 0.3},
            'HeartDisease|Age=Elderly,Gender=Male,Smoking=Yes': {'No': 0.5, 'Yes': 0.5},
            'HeartDisease|Age=Elderly,Gender=Female,Smoking=No': {'No': 0.75, 'Yes': 0.25},
            'HeartDisease|Age=Elderly,Gender=Female,Smoking=Yes': {'No': 0.6, 'Yes': 0.4},

            'LungDisease|Smoking=No': {'No': 0.95, 'Yes': 0.05},
            'LungDisease|Smoking=Yes': {'No': 0.7, 'Yes': 0.3},

            'Diabetes|Age=Young': {'No': 0.9, 'Yes': 0.1},
            'Diabetes|Age=Middle': {'No': 0.8, 'Yes': 0.2},
            'Diabetes|Age=Elderly': {'No': 0.7, 'Yes': 0.3},

            # Symptom probabilities given diseases
            'ChestPain|HeartDisease=No': {'No': 0.9, 'Yes': 0.1},
            'ChestPain|HeartDisease=Yes': {'No': 0.3, 'Yes': 0.7},

            'ECGAbnormal|HeartDisease=No': {'No': 0.95, 'Yes': 0.05},
            'ECGAbnormal|HeartDisease=Yes': {'No': 0.4, 'Yes': 0.6},

            'Cough|LungDisease=No': {'No': 0.8, 'Yes': 0.2},
            'Cough|LungDisease=Yes': {'No': 0.2, 'Yes': 0.8},

            'ShortnessOfBreath|LungDisease=No': {'No': 0.9, 'Yes': 0.1},
            'ShortnessOfBreath|LungDisease=Yes': {'No': 0.2, 'Yes': 0.8},

            'Fatigue|HeartDisease=No,LungDisease=No,Diabetes=No': {'No': 0.95, 'Yes': 0.05},
            'Fatigue|HeartDisease=Yes,LungDisease=No,Diabetes=No': {'No': 0.7, 'Yes': 0.3},
            'Fatigue|HeartDisease=No,LungDisease=Yes,Diabetes=No': {'No': 0.7, 'Yes': 0.3},
            'Fatigue|HeartDisease=Yes,LungDisease=Yes,Diabetes=No': {'No': 0.4, 'Yes': 0.6},
            'Fatigue|HeartDisease=No,LungDisease=No,Diabetes=Yes': {'No': 0.7, 'Yes': 0.3},
            'Fatigue|HeartDisease=Yes,LungDisease=No,Diabetes=Yes': {'No': 0.4, 'Yes': 0.6},
            'Fatigue|HeartDisease=No,LungDisease=Yes,Diabetes=Yes': {'No': 0.4, 'Yes': 0.6},
            'Fatigue|HeartDisease=Yes,LungDisease=Yes,Diabetes=Yes': {'No': 0.1, 'Yes': 0.9},

            'ExcessiveThirst|Diabetes=No': {'No': 0.95, 'Yes': 0.05},
            'ExcessiveThirst|Diabetes=Yes': {'No': 0.3, 'Yes': 0.7},

            # Treatment probabilities given diseases
            'EmergencyCare|ChestPain=No,ECGAbnormal=No,ShortnessOfBreath=No': {'No': 0.99, 'Yes': 0.01},
            'EmergencyCare|ChestPain=Yes,ECGAbnormal=No,ShortnessOfBreath=No': {'No': 0.8, 'Yes': 0.2},
            'EmergencyCare|ChestPain=No,ECGAbnormal=Yes,ShortnessOfBreath=No': {'No': 0.8, 'Yes': 0.2},
            'EmergencyCare|ChestPain=Yes,ECGAbnormal=Yes,ShortnessOfBreath=No': {'No': 0.4, 'Yes': 0.6},
            'EmergencyCare|ChestPain=No,ECGAbnormal=No,ShortnessOfBreath=Yes': {'No': 0.8, 'Yes': 0.2},
            'EmergencyCare|ChestPain=Yes,ECGAbnormal=No,ShortnessOfBreath=Yes': {'No': 0.4, 'Yes': 0.6},
            'EmergencyCare|ChestPain=No,ECGAbnormal=Yes,ShortnessOfBreath=Yes': {'No': 0.4, 'Yes': 0.6},
            'EmergencyCare|ChestPain=Yes,ECGAbnormal=Yes,ShortnessOfBreath=Yes': {'No': 0.01, 'Yes': 0.99},

            'CardiacMedication|HeartDisease=No': {'No': 0.95, 'Yes': 0.05},
            'CardiacMedication|HeartDisease=Yes': {'No': 0.1, 'Yes': 0.9},

            'RespiratoryMedication|LungDisease=No': {'No': 0.9, 'Yes': 0.1},
            'RespiratoryMedication|LungDisease=Yes': {'No': 0.2, 'Yes': 0.8},

            'InsulinTherapy|Diabetes=No': {'No': 0.95, 'Yes': 0.05},
            'InsulinTherapy|Diabetes=Yes': {'No': 0.3, 'Yes': 0.7}
        }

    def query(self, evidence=None, variables=None):
        """
        Query the Bayesian Network for probabilities.

        Args:
            evidence (dict): Evidence variables and their values
            variables (list): Variables to query

        Returns:
            dict: Probability distributions for queried variables
        """
        if evidence is None:
            evidence = {}

        if variables is None:
            # Query all variables not in evidence
            variables = [node for node in self.nodes if node not in evidence]

        # Convert numeric evidence to string values
        evidence_str = {}
        for var, val in evidence.items():
            if var == 'Age':
                if val == 0:
                    evidence_str[var] = 'Young'
                elif val == 1:
                    evidence_str[var] = 'Middle'
                else:
                    evidence_str[var] = 'Elderly'
            elif var in ['Gender', 'Smoking', 'HeartDisease', 'LungDisease', 'Diabetes',
                        'ChestPain', 'ECGAbnormal', 'Cough', 'ShortnessOfBreath',
                        'Fatigue', 'ExcessiveThirst']:
                evidence_str[var] = 'Yes' if val == 1 else 'No'

        # Perform simple inference
        result = {}
        for var in variables:
            # For simplicity, we'll use a naive approach
            if var in ['Age', 'Gender', 'Smoking']:
                # Prior variables
                result[var] = self._get_prior_probability(var)
            elif var in ['HeartDisease', 'LungDisease', 'Diabetes']:
                # Disease variables
                result[var] = self._get_disease_probability(var, evidence_str)
            elif var in ['ChestPain', 'ECGAbnormal', 'Cough', 'ShortnessOfBreath', 'Fatigue', 'ExcessiveThirst']:
                # Symptom variables
                result[var] = self._get_symptom_probability(var, evidence_str)
            else:
                # Treatment variables
                result[var] = self._get_treatment_probability(var, evidence_str)

        return result

    def _get_prior_probability(self, var):
        """Get prior probability for a variable."""
        # Return probability array [P(var=0), P(var=1)]
        if var == 'Age':
            return np.array([self.probabilities['Age']['Young'],
                            self.probabilities['Age']['Middle'],
                            self.probabilities['Age']['Elderly']])
        elif var == 'Gender':
            return np.array([self.probabilities['Gender']['Male'],
                            self.probabilities['Gender']['Female']])
        elif var == 'Smoking':
            return np.array([self.probabilities['Smoking']['No'],
                            self.probabilities['Smoking']['Yes']])
        return np.array([0.5, 0.5])  # Default

    def _get_disease_probability(self, disease, evidence):
        """Get disease probability given evidence."""
        if disease == 'HeartDisease':
            # Check if we have all the parent variables
            if all(var in evidence for var in ['Age', 'Gender', 'Smoking']):
                key = f"HeartDisease|Age={evidence['Age']},Gender={evidence['Gender']},Smoking={evidence['Smoking']}"
                if key in self.probabilities:
                    return np.array([self.probabilities[key]['No'], self.probabilities[key]['Yes']])
            # If we don't have all parents, use a default
            return np.array([0.8, 0.2])  # Default

        elif disease == 'LungDisease':
            if 'Smoking' in evidence:
                key = f"LungDisease|Smoking={evidence['Smoking']}"
                if key in self.probabilities:
                    return np.array([self.probabilities[key]['No'], self.probabilities[key]['Yes']])
            return np.array([0.9, 0.1])  # Default

        elif disease == 'Diabetes':
            if 'Age' in evidence:
                key = f"Diabetes|Age={evidence['Age']}"
                if key in self.probabilities:
                    return np.array([self.probabilities[key]['No'], self.probabilities[key]['Yes']])
            return np.array([0.85, 0.15])  # Default

        return np.array([0.5, 0.5])  # Default

    def _get_symptom_probability(self, symptom, evidence):
        """Get symptom probability given evidence."""
        if symptom == 'ChestPain' and 'HeartDisease' in evidence:
            key = f"ChestPain|HeartDisease={evidence['HeartDisease']}"
            if key in self.probabilities:
                return np.array([self.probabilities[key]['No'], self.probabilities[key]['Yes']])

        elif symptom == 'ECGAbnormal' and 'HeartDisease' in evidence:
            key = f"ECGAbnormal|HeartDisease={evidence['HeartDisease']}"
            if key in self.probabilities:
                return np.array([self.probabilities[key]['No'], self.probabilities[key]['Yes']])

        elif symptom == 'Cough' and 'LungDisease' in evidence:
            key = f"Cough|LungDisease={evidence['LungDisease']}"
            if key in self.probabilities:
                return np.array([self.probabilities[key]['No'], self.probabilities[key]['Yes']])

        elif symptom == 'ShortnessOfBreath' and 'LungDisease' in evidence:
            key = f"ShortnessOfBreath|LungDisease={evidence['LungDisease']}"
            if key in self.probabilities:
                return np.array([self.probabilities[key]['No'], self.probabilities[key]['Yes']])

        elif symptom == 'Fatigue' and all(var in evidence for var in ['HeartDisease', 'LungDisease', 'Diabetes']):
            key = f"Fatigue|HeartDisease={evidence['HeartDisease']},LungDisease={evidence['LungDisease']},Diabetes={evidence['Diabetes']}"
            if key in self.probabilities:
                return np.array([self.probabilities[key]['No'], self.probabilities[key]['Yes']])

        elif symptom == 'ExcessiveThirst' and 'Diabetes' in evidence:
            key = f"ExcessiveThirst|Diabetes={evidence['Diabetes']}"
            if key in self.probabilities:
                return np.array([self.probabilities[key]['No'], self.probabilities[key]['Yes']])

        return np.array([0.5, 0.5])  # Default

    def _get_treatment_probability(self, treatment, evidence):
        """Get treatment probability given evidence."""
        if treatment == 'EmergencyCare' and all(var in evidence for var in ['ChestPain', 'ECGAbnormal', 'ShortnessOfBreath']):
            key = f"EmergencyCare|ChestPain={evidence['ChestPain']},ECGAbnormal={evidence['ECGAbnormal']},ShortnessOfBreath={evidence['ShortnessOfBreath']}"
            if key in self.probabilities:
                return np.array([self.probabilities[key]['No'], self.probabilities[key]['Yes']])

        elif treatment == 'CardiacMedication' and 'HeartDisease' in evidence:
            key = f"CardiacMedication|HeartDisease={evidence['HeartDisease']}"
            if key in self.probabilities:
                return np.array([self.probabilities[key]['No'], self.probabilities[key]['Yes']])

        elif treatment == 'RespiratoryMedication' and 'LungDisease' in evidence:
            key = f"RespiratoryMedication|LungDisease={evidence['LungDisease']}"
            if key in self.probabilities:
                return np.array([self.probabilities[key]['No'], self.probabilities[key]['Yes']])

        elif treatment == 'InsulinTherapy' and 'Diabetes' in evidence:
            key = f"InsulinTherapy|Diabetes={evidence['Diabetes']}"
            if key in self.probabilities:
                return np.array([self.probabilities[key]['No'], self.probabilities[key]['Yes']])

        return np.array([0.5, 0.5])  # Default

    def predict_diagnosis(self, symptoms):
        """
        Predict the probability of different diseases given symptoms.

        Args:
            symptoms (dict): Dictionary of symptoms and their values (0=No, 1=Yes)

        Returns:
            dict: Probability of each disease
        """
        # Define the diseases to predict
        diseases = ['HeartDisease', 'LungDisease', 'Diabetes']

        # Query the model
        result = self.query(evidence=symptoms, variables=diseases)

        # Format the results
        diagnosis = {}
        for disease in diseases:
            # Probability of having the disease (index 1 is "Yes")
            diagnosis[disease] = result[disease][1]

        return diagnosis

    def recommend_treatment(self, diagnosis):
        """
        Recommend treatments based on diagnosis.

        Args:
            diagnosis (dict): Dictionary of diseases and their values (0=No, 1=Yes)

        Returns:
            dict: Recommended treatments and their probabilities
        """
        # Define the treatments to recommend
        treatments = ['EmergencyCare', 'CardiacMedication', 'RespiratoryMedication', 'InsulinTherapy']

        # Query the model
        result = self.query(evidence=diagnosis, variables=treatments)

        # Format the results
        recommendations = {}
        for treatment in treatments:
            # Probability of needing the treatment (index 1 is "Yes")
            recommendations[treatment] = result[treatment][1]

        return recommendations

    def plot_network(self):
        """
        Plot the Bayesian Network structure.

        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Create a directed graph
        G = nx.DiGraph()

        # Add edges from the model
        G.add_edges_from(self.edges)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Define node colors by category
        node_colors = []
        for node in G.nodes():
            if node in ['Age', 'Gender', 'Smoking']:
                node_colors.append('lightblue')  # Demographics
            elif node in ['HeartDisease', 'LungDisease', 'Diabetes']:
                node_colors.append('salmon')  # Diseases
            elif node in ['ChestPain', 'ECGAbnormal', 'Cough', 'ShortnessOfBreath', 'Fatigue', 'ExcessiveThirst']:
                node_colors.append('lightgreen')  # Symptoms
            else:
                node_colors.append('lightyellow')  # Treatments

        # Define node positions using a layout algorithm
        pos = nx.spring_layout(G, seed=42)

        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, arrows=True, arrowsize=20)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

        # Add a legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=15, label='Demographics'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', markersize=15, label='Diseases'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15, label='Symptoms'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightyellow', markersize=15, label='Treatments')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        # Set title and remove axis
        plt.title('Bayesian Network for Patient Diagnosis and Treatment', fontsize=16)
        plt.axis('off')

        return fig

    def plot_diagnosis_probabilities(self, diagnosis):
        """
        Plot the diagnosis probabilities.

        Args:
            diagnosis (dict): Dictionary of diseases and their probabilities

        Returns:
            plotly.graph_objects.Figure: The figure object
        """
        # Create a DataFrame for plotting
        df = pd.DataFrame({
            'Disease': list(diagnosis.keys()),
            'Probability': list(diagnosis.values())
        })

        # Create a bar chart
        fig = px.bar(
            df,
            x='Disease',
            y='Probability',
            color='Probability',
            color_continuous_scale='Reds',
            title='Disease Probability Based on Symptoms',
            labels={'Probability': 'Probability of Disease'}
        )

        # Update layout
        fig.update_layout(
            xaxis_title='Disease',
            yaxis_title='Probability',
            yaxis_range=[0, 1],
            coloraxis_showscale=False
        )

        return fig

    @classmethod
    def create_default_model(cls):
        """
        Create a default Bayesian Network model.

        Returns:
            BayesianPatientModel: Configured model
        """
        return cls()
