# Intelligent Patient Flow and Diagnosis Modeling in Smart Hospitals

This project implements stochastic processes and machine learning models to analyze and simulate patient flow and diagnosis in a hospital setting. It provides a comprehensive solution for understanding patient journeys through different hospital departments and predicting diagnoses based on patient characteristics.

## Overview

The Intelligent Patient Flow and Diagnosis Modeling system is a sophisticated analytics platform designed for modern healthcare facilities. It combines advanced stochastic processes, machine learning algorithms, and interactive visualizations to model, simulate, and analyze patient flow, waiting times, resource utilization, and diagnosis prediction in a hospital environment.

## Features

- **Patient Flow Simulation**: Uses Markov chains to model patient transitions between departments
- **Hidden Markov Models**: Models patient health states that are not directly observable
- **Poisson Arrival Process**: Models patient arrivals to the hospital using Poisson processes
- **Hospital Queuing Theory**: Models patient waiting times and service times in different departments
- **Bayesian Networks**: Models probabilistic relationships between symptoms and diseases
- **Diagnosis Prediction**: Employs machine learning models to predict patient diagnoses
- **Data Analysis**: Provides tools for exploring and visualizing hospital patient data
- **3D Interactive Visualizations**: Offers immersive 3D visualizations of patient flow and models
- **Interactive Dashboard**: Offers a user-friendly Streamlit interface for all functionalities

## Project Structure

```
.
├── app/
│   ├── markov_model.py         # Stochastic models for patient flow
│   ├── hidden_markov_model.py  # Hidden Markov Models for patient health states
│   ├── poisson_model.py        # Poisson process models for patient arrivals
│   ├── queuing_model.py        # Queuing theory models for hospital departments
│   ├── bayesian_network.py     # Bayesian network for diagnosis and treatment
│   ├── visualization_3d.py     # 3D visualization tools for models
│   ├── ui_theme.py             # UI theme and styling components
│   ├── streamlit_app.py        # Main Streamlit application
│   └── utils.py                # Utility functions for data analysis
├── data/
│   └── hospital_patient_dataset.csv  # Patient dataset
├── models/
│   ├── diagnosis_model.py      # Machine learning models for diagnosis prediction
│   └── simple_diagnosis_model.py # Simplified probabilistic diagnosis model
├── static/
│   ├── css/                    # CSS styling files
│   └── images/                 # Images and icons for the UI
└── requirements.txt            # Project dependencies
```

## Stochastic Processes Used

1. **Markov Chains**
   - Model patient transitions between departments (ER, Ward, ICU, Discharged)
   - Transition probabilities estimated from data or set manually
   - Used to simulate patient journeys through the hospital
   - Visualized in interactive 3D space for better understanding

2. **Hidden Markov Models (HMMs)**
   - Model patient health states that are not directly observable
   - Infer underlying health conditions from observable symptoms and vital signs
   - Three hidden states: Stable, Deteriorating, Critical
   - Four observable states: Normal Vitals, Mild Symptoms, Severe Symptoms, Critical Symptoms
   - Used for health state prediction and patient monitoring
   - Visualized in 3D with state transitions and emission probabilities

3. **Poisson Processes**
   - Model patient arrivals to the hospital
   - Exponential inter-arrival times with time-varying rates
   - Accounts for time-of-day and day-of-week variations
   - Used to simulate realistic hospital operations and staffing needs

4. **Queuing Theory**
   - Model waiting times and service times in different departments
   - Multiple service distributions (exponential, normal) for different departments
   - Priority-based and FIFO queue disciplines
   - Helps optimize resource allocation and reduce waiting times

5. **Bayesian Networks**
   - Model probabilistic relationships between symptoms, diseases, and treatments
   - Enables diagnostic inference and treatment recommendations
   - Accounts for patient demographics and risk factors
   - Visualized as an interactive directed acyclic graph

## Machine Learning Models

1. **Random Forest Classifier**
   - Ensemble learning method for diagnosis prediction
   - Feature importance analysis
   - High accuracy and robustness

2. **Logistic Regression**
   - Linear model for diagnosis prediction
   - Interpretable coefficients
   - Baseline model for comparison

3. **Support Vector Machine**
   - Non-linear classification for diagnosis prediction
   - Kernel methods for complex decision boundaries
   - Good performance on small to medium datasets

## Installation and Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Intelligent-Patient-Flow-and-diagnosis-Modelling-in-Smart-Hospitals
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run app/streamlit_app.py
   ```

## Usage

The Streamlit app provides multiple interactive pages:

1. **Home**: Overview of the project, features, and sample data
2. **Patient Flow Simulation**: Simulate patient journeys and hospital operations using Markov chains
   - Single Patient Simulation: Track an individual patient's journey through departments
   - Hospital Day Simulation: Simulate an entire day of hospital operations
   - Transition Matrix: View and understand the patient transition probabilities

3. **Hidden Markov Models**: Explore unobservable patient health states
   - Patient State Simulation: Simulate health state trajectories over time
   - Model Parameters: Visualize transition and emission matrices
   - State Prediction: Predict hidden health states from observed symptoms

4. **Poisson Arrival Process**: Model random patient arrivals
   - Arrival Simulation: Simulate patient arrivals over time
   - Arrival Patterns: Visualize time-varying arrival rates
   - Department Distribution: Configure arrival distribution across departments

5. **Hospital Queuing Theory**: Model waiting times and resource utilization
   - Simulation: Run queuing simulations and analyze results
   - Service Time Distributions: Visualize service time distributions by department
   - Department Configuration: View and understand department configurations

6. **Bayesian Networks**: Model probabilistic relationships for diagnosis
   - Diagnosis: Predict diseases based on symptoms and patient information
   - Network Visualization: Visualize the Bayesian network structure
   - Treatment Recommendations: Get treatment suggestions based on diagnoses

7. **Diagnosis Prediction**: Predict patient diagnoses using machine learning
   - Predict Diagnosis: Get diagnosis predictions for new patients
   - Model Performance: Evaluate model accuracy and performance
   - Feature Importance: Understand factors influencing diagnosis

8. **Data Analysis**: Explore patterns and insights from hospital patient data
   - Basic Statistics: View summary statistics and distributions
   - Patient Flow Analysis: Analyze patient flow patterns with Sankey diagrams
   - Length of Stay Analysis: Analyze factors affecting length of stay

9. **3D Visualizations**: Explore immersive 3D visualizations of models
   - Patient Flow 3D: Visualize patient transitions in 3D space
   - Queuing Model 3D: Visualize queue lengths and utilization in 3D
   - Hidden Markov Model 3D: Visualize the HMM structure in 3D

## Data

The project uses a synthetic dataset of hospital patients with the following features:

### Patient Demographics
- **Age**: Patient age in years
- **Gender**: Patient gender (Male, Female)
- **Smoking**: Smoking status (Smoker, Non-smoker)

### Hospital Information
- **Department**: Hospital department (ER, Ward, ICU)
- **Length_of_Stay**: Duration of hospital stay in days
- **Medical_Condition**: Patient's presenting condition (Cardiac, Neuro, Injury)
- **Diagnosis**: Target variable for prediction (Condition A, Condition B, Condition C)

### Symptoms and Vital Signs
- **Chest Pain**: Presence of chest pain (Yes/No)
- **Shortness of Breath**: Presence of breathing difficulty (Yes/No)
- **Fatigue**: Presence of fatigue (Yes/No)
- **Cough**: Presence of cough (Yes/No)
- **Excessive Thirst**: Presence of excessive thirst (Yes/No)
- **ECG Abnormal**: Abnormal electrocardiogram results (Yes/No)

### Simulation-Generated Data
The application also generates synthetic data during simulations:
- **Patient Paths**: Sequences of departments visited
- **Service Times**: Time spent in each department
- **Waiting Times**: Time spent waiting for service
- **Queue Lengths**: Number of patients waiting in each department
- **Resource Utilization**: Proportion of time resources are in use

## Key Technologies

- **Python**: Core programming language
- **Streamlit**: Interactive web application framework
- **NumPy/Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Static data visualization
- **Plotly**: Interactive 2D and 3D visualizations
- **SciPy**: Scientific computing and statistical distributions
- **Scikit-learn**: Machine learning algorithms
- **hmmlearn**: Hidden Markov Model implementation
- **NetworkX**: Graph-based modeling for Bayesian networks

## Mathematical Models

### Stochastic Processes
- **Markov Chains**: Discrete-time stochastic process with the Markov property
- **Hidden Markov Models**: Statistical Markov model with unobservable states
- **Poisson Process**: Count process with exponentially distributed inter-arrival times
- **Queuing Theory**: M/M/c, M/G/c, and priority queue models
- **Bayesian Networks**: Probabilistic graphical models representing conditional dependencies

### Machine Learning
- **Classification Models**: For diagnosis prediction
- **Feature Importance Analysis**: For understanding key diagnostic factors
- **Model Evaluation**: Using metrics like accuracy, precision, recall, and F1-score

## Future Improvements

- **Advanced Stochastic Models**: Incorporate semi-Markov processes and continuous-time Markov chains
- **Resource Optimization**: Add capacity planning and resource allocation optimization
- **Deep Learning Models**: Implement neural networks for more accurate diagnosis prediction
- **Real-time Integration**: Connect with hospital information systems for live data
- **Mobile Interface**: Develop a mobile application for on-the-go access
- **Predictive Analytics**: Add forecasting capabilities for patient volumes and resource needs
- **Multi-hospital Network**: Extend the model to handle patient transfers between facilities
- **Personalized Treatment Paths**: Develop individualized patient journey recommendations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was developed as part of a stochastic processes and healthcare analytics course
- Inspired by real-world hospital operations and patient flow modeling
- Special thanks to all contributors and healthcare professionals who provided domain expertise
