import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the hospital patient dataset
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    df = pd.read_csv(file_path)
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        print(f"Dropped {df.isnull().sum().sum()} rows with missing values")
    
    return df

def exploratory_data_analysis(df):
    """
    Perform exploratory data analysis on the hospital patient dataset
    
    Args:
        df (pd.DataFrame): Hospital patient dataframe
        
    Returns:
        dict: Dictionary containing EDA results
    """
    results = {}
    
    # Basic statistics
    results['basic_stats'] = df.describe()
    
    # Department distribution
    results['department_counts'] = df['Department'].value_counts()
    
    # Medical condition distribution
    results['condition_counts'] = df['Medical_Condition'].value_counts()
    
    # Diagnosis distribution
    results['diagnosis_counts'] = df['Diagnosis'].value_counts()
    
    # Length of stay statistics by department
    results['los_by_dept'] = df.groupby('Department')['Length_of_Stay'].agg(['mean', 'median', 'std'])
    
    # Age statistics by department
    results['age_by_dept'] = df.groupby('Department')['Age'].agg(['mean', 'median', 'std'])
    
    return results

def create_transition_matrix(df, state_col='Department'):
    """
    Create a transition matrix from patient flow data
    
    Args:
        df (pd.DataFrame): Hospital patient dataframe
        state_col (str): Column name for the states
        
    Returns:
        tuple: (states, transition_matrix)
    """
    # Get unique states
    states = df[state_col].unique().tolist()
    
    # Add 'Discharged' state
    if 'Discharged' not in states:
        states.append('Discharged')
    
    # Initialize transition matrix with zeros
    n_states = len(states)
    transition_matrix = np.zeros((n_states, n_states))
    
    # Calculate transition probabilities based on data
    # This is a simplified approach - in reality, you would need sequential data
    # Here we're using department distribution as a proxy
    dept_counts = df[state_col].value_counts(normalize=True)
    
    for i, state_i in enumerate(states[:-1]):  # Exclude 'Discharged'
        # Set some probability to stay in the same state
        transition_matrix[i, i] = 0.2
        
        # Set probabilities to transition to other states
        remaining_prob = 0.8
        for j, state_j in enumerate(states):
            if i != j:
                if state_j == 'Discharged':
                    # Higher probability to be discharged from ER and Ward
                    if state_i in ['ER', 'Ward']:
                        transition_matrix[i, j] = 0.3
                    else:
                        transition_matrix[i, j] = 0.2
                    remaining_prob -= transition_matrix[i, j]
                elif j < len(states) - 1:  # Not 'Discharged'
                    # Use department distribution as a proxy for transition probabilities
                    if state_j in dept_counts:
                        transition_matrix[i, j] = remaining_prob * dept_counts[state_j] / (dept_counts.sum() - (dept_counts[state_i] if state_i in dept_counts else 0))
        
        # Normalize to ensure each row sums to 1
        transition_matrix[i, :] = transition_matrix[i, :] / transition_matrix[i, :].sum()
    
    # Set 'Discharged' state to stay in 'Discharged' with probability 1
    transition_matrix[-1, -1] = 1.0
    
    return states, transition_matrix

def plot_patient_flow_sankey(df):
    """
    Create a Sankey diagram of patient flow
    
    Args:
        df (pd.DataFrame): Hospital patient dataframe
        
    Returns:
        plotly.graph_objects.Figure: Sankey diagram
    """
    # Create source-target pairs
    source = []
    target = []
    value = []
    
    # Department to Medical Condition
    for dept in df['Department'].unique():
        for cond in df['Medical_Condition'].unique():
            count = len(df[(df['Department'] == dept) & (df['Medical_Condition'] == cond)])
            if count > 0:
                source.append(dept)
                target.append(cond)
                value.append(count)
    
    # Medical Condition to Diagnosis
    for cond in df['Medical_Condition'].unique():
        for diag in df['Diagnosis'].unique():
            count = len(df[(df['Medical_Condition'] == cond) & (df['Diagnosis'] == diag)])
            if count > 0:
                source.append(cond)
                target.append(diag)
                value.append(count)
    
    # Create node labels
    all_nodes = list(df['Department'].unique()) + list(df['Medical_Condition'].unique()) + list(df['Diagnosis'].unique())
    node_indices = {node: i for i, node in enumerate(all_nodes)}
    
    # Map source and target to indices
    source_indices = [node_indices[s] for s in source]
    target_indices = [node_indices[t] for t in target]
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=value
        )
    )])
    
    fig.update_layout(title_text="Patient Flow Sankey Diagram", font_size=10)
    return fig
