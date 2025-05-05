
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import time

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.markov_model import HospitalStochasticModel
from app.hidden_markov_model import HiddenMarkovPatientModel
from app.poisson_model import PoissonArrivalModel
from app.queuing_model import HospitalQueueingModel
from app.bayesian_network import BayesianPatientModel
from app.visualization_3d import HospitalVisualization3D
from app.ui_theme import ModernUITheme
from app.utils import load_and_preprocess_data, exploratory_data_analysis, plot_patient_flow_sankey
from models.simple_diagnosis_model import SimpleDiagnosisModel

# Set page configuration
st.set_page_config(
    page_title="Intelligent Patient Flow and Diagnosis Modeling",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize UI theme
ui_theme = ModernUITheme()
ui_theme.apply_theme()

# Initialize 3D visualization
vis_3d = HospitalVisualization3D()

# Initialize session state
if 'patient_flow_model' not in st.session_state:
    st.session_state.patient_flow_model = HospitalStochasticModel(data_path="data/hospital_patient_dataset.csv")

if 'diagnosis_model' not in st.session_state:
    st.session_state.diagnosis_model = SimpleDiagnosisModel()
    # Check if model file exists, otherwise train a new one
    model_path = "models/simple_diagnosis_model.joblib"
    if os.path.exists(model_path):
        st.session_state.diagnosis_model.load_model(model_path)
    else:
        with st.spinner("Training diagnosis model..."):
            st.session_state.diagnosis_model.train("data/hospital_patient_dataset.csv")
            st.session_state.diagnosis_model.save_model(model_path)

if 'hidden_markov_model' not in st.session_state:
    # Create a pre-configured HMM for patient health states
    st.session_state.hidden_markov_model = HiddenMarkovPatientModel.create_patient_hmm()

if 'poisson_model' not in st.session_state:
    # Create a pre-configured Poisson arrival model
    st.session_state.poisson_model = PoissonArrivalModel.create_default_model()

if 'queuing_model' not in st.session_state:
    # Create a pre-configured hospital queuing model
    st.session_state.queuing_model = HospitalQueueingModel.create_default_model()

if 'bayesian_network' not in st.session_state:
    # Create a pre-configured Bayesian Network model
    st.session_state.bayesian_network = BayesianPatientModel.create_default_model()

# Add loading animation
if 'first_load' not in st.session_state:
    with st.spinner("Loading Smart Hospital Analytics Dashboard..."):
        # Simulate loading time for better UX
        time.sleep(1)
    st.session_state.first_load = True

# Load data
@st.cache_data
def load_data():
    return load_and_preprocess_data("data/hospital_patient_dataset.csv")

df = load_data()

# Add logo to sidebar
ui_theme.add_logo()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Patient Flow Simulation",
    "Hidden Markov Models",
    "Poisson Arrival Process",
    "Hospital Queuing Theory",
    "Bayesian Networks",
    "Diagnosis Prediction",
    "Data Analysis",
    "3D Visualizations"
])

# Add footer to sidebar
ui_theme.add_footer()

# Home page
if page == "Home":
    st.title("🏥 Intelligent Patient Flow and Diagnosis Modeling in Smart Hospitals")

    # Create welcome cards directly without using the card template
    st.markdown("""
    <div style="background-color: #1E2130; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 1rem; overflow: hidden;">
        <div style="display: flex; align-items: center; padding: 1rem; border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
            <div style="font-size: 1.5rem; margin-right: 0.5rem;">🏥</div>
            <h3 style="margin: 0; color: #4F8BF9;">Welcome to the Smart Hospital Analytics Dashboard</h3>
        </div>
        <div style="padding: 1rem;">
            This advanced analytics dashboard provides comprehensive tools for modeling, simulating,
            and analyzing patient flow and diagnosis in a modern hospital setting.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Create feature cards in columns
    col1, col2 = st.columns(2)

    with col1:
        # Create feature card directly
        st.markdown("""
        <div style="background-color: #1E2130; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 1rem; overflow: hidden;">
            <div style="display: flex; align-items: center; padding: 1rem; border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
                <div style="font-size: 1.5rem; margin-right: 0.5rem;">📊</div>
                <h3 style="margin: 0; color: #4F8BF9;">Advanced Modeling Features</h3>
            </div>
            <div style="padding: 1rem;">
                <ul>
                    <li><strong>Patient Flow Simulation</strong>: Simulate patient movement using Markov chains</li>
                    <li><strong>Hidden Markov Models</strong>: Model unobservable patient health states</li>
                    <li><strong>Poisson Arrival Process</strong>: Model random patient arrivals</li>
                    <li><strong>Hospital Queuing Theory</strong>: Model waiting times and resource utilization</li>
                    <li><strong>Bayesian Networks</strong>: Model probabilistic relationships between symptoms and diseases</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Create visualization card directly
        st.markdown("""
        <div style="background-color: #1E2130; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 1rem; overflow: hidden;">
            <div style="display: flex; align-items: center; padding: 1rem; border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
                <div style="font-size: 1.5rem; margin-right: 0.5rem;">📈</div>
                <h3 style="margin: 0; color: #4F8BF9;">Visualization & Analysis Tools</h3>
            </div>
            <div style="padding: 1rem;">
                <ul>
                    <li><strong>3D Interactive Visualizations</strong>: Explore data in three dimensions</li>
                    <li><strong>Diagnosis Prediction</strong>: Predict diagnoses using machine learning</li>
                    <li><strong>Data Analysis</strong>: Discover patterns in hospital patient data</li>
                    <li><strong>Real-time Simulations</strong>: Watch patient flow in real-time</li>
                    <li><strong>Treatment Recommendations</strong>: Get AI-powered treatment suggestions</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Create info box for stochastic processes directly
    st.markdown("""
    <div style="display: flex; align-items: flex-start; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; background-color: rgba(51, 181, 229, 0.1); border-left: 4px solid #33B5E5;">
        <div style="font-size: 1.2rem; margin-right: 0.5rem;">📌</div>
        <div style="flex: 1;">
            <strong>Stochastic Processes Used:</strong>
            <ul>
                <li><strong>Markov Chains</strong>: Model patient transitions between departments</li>
                <li><strong>Hidden Markov Models</strong>: Model unobservable patient health states</li>
                <li><strong>Poisson Process</strong>: Model random patient arrivals</li>
                <li><strong>Queuing Theory</strong>: Model waiting lines and service times</li>
                <li><strong>Bayesian Networks</strong>: Model probabilistic causal relationships</li>
                <li><strong>Exponential & Normal Distributions</strong>: Model service times</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Create stat cards using Streamlit's native metric component
    st.subheader("Hospital Performance Metrics")

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.metric(
            label="Average Length of Stay",
            value="3.2 days",
            delta="-0.4 days"
        )

    with stat_col2:
        st.metric(
            label="Bed Utilization",
            value="78%",
            delta="5%"
        )

    with stat_col3:
        st.metric(
            label="ER Wait Time",
            value="42 min",
            delta="-8 min"
        )

    with stat_col4:
        st.metric(
            label="Readmission Rate",
            value="4.7%",
            delta="-0.3%"
        )

    # Display a sample of the data
    st.subheader("Sample Data")
    st.dataframe(df.head())

# Patient Flow Simulation page
elif page == "Patient Flow Simulation":
    st.title("🏥 Patient Flow Simulation")

    # Create tabs for different simulation options
    sim_tab1, sim_tab2, sim_tab3 = st.tabs(["Single Patient Simulation", "Hospital Day Simulation", "Transition Matrix"])

    with sim_tab1:
        st.subheader("Simulate a Single Patient's Journey")

        col1, col2 = st.columns(2)

        with col1:
            start_state = st.selectbox("Select Starting Department", ['ER', 'Ward', 'ICU'])
            steps = st.slider("Maximum Number of Steps", 1, 10, 5)
            include_times = st.checkbox("Include Service Times", value=True)

        if st.button("Simulate Patient Journey"):
            if include_times:
                path, times = st.session_state.patient_flow_model.simulate_patient_flow(start_state, steps, include_times=True)

                # Create a DataFrame for visualization
                journey_df = pd.DataFrame({
                    'Step': range(len(path)),
                    'Department': path,
                    'Hours': [0] + times  # Add 0 for the initial state
                })

                # Calculate cumulative time
                journey_df['Cumulative Hours'] = journey_df['Hours'].cumsum()

                # Display the journey
                st.subheader("Patient Journey")
                st.write(f"Path: {' ➡️ '.join(path)}")

                if len(times) > 0:
                    st.write(f"Total Length of Stay: {sum(times):.1f} hours ({sum(times)/24:.1f} days)")

                # Create a Gantt chart
                fig = px.timeline(
                    journey_df,
                    x_start='Cumulative Hours',
                    x_end=journey_df['Cumulative Hours'] + journey_df['Hours'],
                    y='Department',
                    color='Department',
                    title="Patient Journey Timeline"
                )
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig)

                # Display the journey details
                st.dataframe(journey_df)
            else:
                path = st.session_state.patient_flow_model.simulate_patient_flow(start_state, steps)
                st.write("Patient Path:", " ➡️ ".join(path))

    with sim_tab2:
        st.subheader("Simulate a Day in the Hospital")

        duration = st.slider("Simulation Duration (hours)", 6, 48, 24)

        if st.button("Run Hospital Simulation"):
            with st.spinner("Simulating hospital operations..."):
                results = st.session_state.patient_flow_model.simulate_hospital_day(duration_hours=duration)

            # Display summary statistics
            total_patients = len(results)
            avg_los = np.mean([r['total_los'] for r in results])

            st.write(f"Total Patients: {total_patients}")
            st.write(f"Average Length of Stay: {avg_los:.1f} hours ({avg_los/24:.1f} days)")

            # Create a DataFrame for all patients
            all_paths = []
            for i, r in enumerate(results):
                for j, (dept, time) in enumerate(zip(r['path'][:-1], r['service_times'])):
                    all_paths.append({
                        'Patient': i+1,
                        'Department': dept,
                        'Start Time': r['absolute_times'][j],
                        'End Time': r['absolute_times'][j+1],
                        'Duration': time
                    })

            if all_paths:
                paths_df = pd.DataFrame(all_paths)

                # Create a Gantt chart for all patients
                fig = px.timeline(
                    paths_df,
                    x_start='Start Time',
                    x_end='End Time',
                    y='Patient',
                    color='Department',
                    title="Hospital Patient Flow Simulation"
                )
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig)

                # Department occupancy over time
                st.subheader("Department Occupancy Over Time")

                # Create time points for the entire simulation
                time_points = np.linspace(0, duration, 100)
                dept_occupancy = {dept: [0] * len(time_points) for dept in ['ER', 'Ward', 'ICU']}

                # Calculate occupancy at each time point
                for t_idx, t in enumerate(time_points):
                    for _, row in paths_df.iterrows():
                        if row['Start Time'] <= t < row['End Time']:
                            dept = row['Department']
                            if dept in dept_occupancy:
                                dept_occupancy[dept][t_idx] += 1

                # Create occupancy DataFrame
                occupancy_df = pd.DataFrame({
                    'Time': time_points,
                    **dept_occupancy
                })

                # Plot occupancy
                fig = px.line(
                    occupancy_df,
                    x='Time',
                    y=['ER', 'Ward', 'ICU'],
                    title="Department Occupancy Over Time",
                    labels={'value': 'Number of Patients', 'variable': 'Department'}
                )
                st.plotly_chart(fig)

    with sim_tab3:
        st.subheader("Transition Matrix")

        # Display the transition matrix
        fig = st.session_state.patient_flow_model.plot_transition_matrix()
        st.pyplot(fig)

        # Explain the transition matrix
        st.markdown("""
        The transition matrix shows the probability of a patient moving from one department (rows) to another (columns).

        For example:
        - A patient in the ER has a probability of moving to the Ward, ICU, or being discharged
        - A patient in the ICU has a probability of moving to the Ward or being discharged
        - Once discharged, a patient stays discharged (probability 1.0)
        """)

# Hidden Markov Models page
elif page == "Hidden Markov Models":
    st.title("🔄 Hidden Markov Models for Patient Health States")

    st.markdown("""
    Hidden Markov Models (HMMs) are powerful tools for modeling systems where the true states are not directly
    observable but can be inferred from observable measurements. In healthcare, HMMs can model a patient's
    underlying health state based on observable symptoms and vital signs.
    """)

    # Create tabs for different HMM options
    hmm_tab1, hmm_tab2, hmm_tab3 = st.tabs(["Patient State Simulation", "Model Parameters", "State Prediction"])

    with hmm_tab1:
        st.subheader("Simulate Patient Health States")

        col1, col2 = st.columns(2)

        with col1:
            start_state = st.selectbox("Initial Health State", ['Stable', 'Deteriorating', 'Critical'])
            n_days = st.slider("Number of Days to Simulate", 5, 30, 10)

        if st.button("Simulate Patient Health"):
            # Map state names to indices
            state_map = {'Stable': 0, 'Deteriorating': 1, 'Critical': 2}
            start_state_idx = state_map[start_state]

            # Simulate patient trajectory
            states, observations = st.session_state.hidden_markov_model.simulate(n_steps=n_days, start_state=start_state_idx)

            # Create a DataFrame for visualization
            state_names = st.session_state.hidden_markov_model.state_names
            obs_names = st.session_state.hidden_markov_model.observation_names

            trajectory_df = pd.DataFrame({
                'Day': range(1, n_days + 1),
                'Hidden State': [state_names[s] for s in states],
                'Observation': [obs_names[o] for o in observations]
            })

            # Display the trajectory
            st.subheader("Patient Health Trajectory")
            st.dataframe(trajectory_df)

            # Create a line chart for the trajectory
            # Map states to numerical values for plotting
            state_values = {'Stable': 1, 'Deteriorating': 2, 'Critical': 3}
            trajectory_df['State Value'] = trajectory_df['Hidden State'].map(state_values)

            # Create a line chart
            fig = px.line(
                trajectory_df,
                x='Day',
                y='State Value',
                title="Patient Health State Trajectory",
                labels={'State Value': 'Health State (1=Stable, 2=Deteriorating, 3=Critical)'}
            )

            # Add markers for observations
            obs_values = {
                'Normal Vitals': 1,
                'Mild Symptoms': 2,
                'Severe Symptoms': 3,
                'Critical Symptoms': 4
            }
            trajectory_df['Observation Value'] = trajectory_df['Observation'].map(obs_values)

            fig.add_scatter(
                x=trajectory_df['Day'],
                y=trajectory_df['Observation Value'],
                mode='markers',
                name='Observations',
                marker=dict(size=10),
                yaxis='y2'
            )

            # Add second y-axis for observations
            fig.update_layout(
                yaxis2=dict(
                    title='Observation (1=Normal, 2=Mild, 3=Severe, 4=Critical)',
                    overlaying='y',
                    side='right'
                )
            )

            st.plotly_chart(fig)

    with hmm_tab2:
        st.subheader("Model Parameters")

        # Display transition matrix
        st.write("### Transition Matrix")
        st.write("Probability of transitioning between hidden health states:")
        fig = st.session_state.hidden_markov_model.plot_transition_matrix()
        st.pyplot(fig)

        # Display emission matrix
        st.write("### Emission Matrix")
        st.write("Probability of observing symptoms given the hidden health state:")
        fig = st.session_state.hidden_markov_model.plot_emission_matrix()
        st.pyplot(fig)

        # Explain the model
        st.markdown("""
        ### Model Explanation

        The Hidden Markov Model has three hidden states representing the patient's true health condition:

        1. **Stable**: Patient is in good health with stable vital signs
        2. **Deteriorating**: Patient's condition is worsening but not critical
        3. **Critical**: Patient is in critical condition requiring immediate attention

        These states are not directly observable but can be inferred from observable symptoms:

        1. **Normal Vitals**: Temperature, blood pressure, heart rate, etc. are normal
        2. **Mild Symptoms**: Some abnormal vital signs or mild symptoms
        3. **Severe Symptoms**: Multiple abnormal vital signs or severe symptoms
        4. **Critical Symptoms**: Life-threatening symptoms requiring immediate intervention

        The transition matrix shows the probability of moving from one health state to another,
        while the emission matrix shows the probability of observing certain symptoms given the
        underlying health state.
        """)

    with hmm_tab3:
        st.subheader("Predict Hidden Health State")

        st.write("Enter a sequence of observed symptoms to predict the most likely hidden health states:")

        # Allow user to input a sequence of observations
        obs_options = ['Normal Vitals', 'Mild Symptoms', 'Severe Symptoms', 'Critical Symptoms']

        # Create input for 5 days of observations
        obs_inputs = []
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                obs = col.selectbox(f"Day {i+1}", obs_options, key=f"obs_{i}")
                obs_inputs.append(obs)

        if st.button("Predict Health States"):
            # Convert observations to indices
            obs_map = {name: i for i, name in enumerate(st.session_state.hidden_markov_model.observation_names)}
            obs_indices = np.array([obs_map[obs] for obs in obs_inputs]).reshape(-1, 1)

            # Predict most likely state sequence
            states = st.session_state.hidden_markov_model.predict_states(obs_indices)

            # Create results dataframe
            results_df = pd.DataFrame({
                'Day': range(1, len(obs_inputs) + 1),
                'Observation': obs_inputs,
                'Predicted Health State': [st.session_state.hidden_markov_model.state_names[s] for s in states]
            })

            # Display results
            st.subheader("Prediction Results")
            st.dataframe(results_df)

            # Create a visualization
            fig = go.Figure()

            # Add observations
            fig.add_trace(go.Scatter(
                x=results_df['Day'],
                y=results_df['Observation'],
                mode='markers+lines',
                name='Observed Symptoms',
                marker=dict(size=12, symbol='circle')
            ))

            # Add predicted states
            fig.add_trace(go.Scatter(
                x=results_df['Day'],
                y=results_df['Predicted Health State'],
                mode='markers+lines',
                name='Predicted Health State',
                marker=dict(size=12, symbol='square')
            ))

            fig.update_layout(
                title="Observed Symptoms and Predicted Health States",
                xaxis_title="Day",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )

            st.plotly_chart(fig)

# Poisson Arrival Process page
elif page == "Poisson Arrival Process":
    st.title("📊 Poisson Process for Patient Arrivals")

    st.markdown("""
    The Poisson process is a stochastic process that models random arrivals over time.
    In a hospital setting, it's ideal for modeling patient arrivals, as patients typically
    arrive independently of each other at random times.
    """)

    # Create tabs for different Poisson process options
    poisson_tab1, poisson_tab2, poisson_tab3 = st.tabs(["Arrival Simulation", "Arrival Patterns", "Department Distribution"])

    with poisson_tab1:
        st.subheader("Simulate Patient Arrivals")

        col1, col2 = st.columns(2)

        with col1:
            arrival_rate = st.slider("Average Arrivals per Hour", 0.5, 10.0, 3.0, 0.5)
            duration = st.slider("Simulation Duration (hours)", 6, 48, 24, 6)

        with col2:
            start_hour = st.selectbox("Start Hour", list(range(24)), 8)  # Default to 8 AM
            day_of_week = st.selectbox("Day of Week",
                                      ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                      0)  # Default to Monday

        # Update the model with the new arrival rate
        st.session_state.poisson_model.set_arrival_rate(arrival_rate)

        if st.button("Simulate Arrivals"):
            # Simulate arrivals
            arrivals = st.session_state.poisson_model.simulate_arrivals(
                duration_hours=duration,
                start_hour=start_hour,
                day_of_week=day_of_week
            )

            # Display summary statistics
            st.subheader("Simulation Results")
            st.write(f"Total Arrivals: {len(arrivals)}")
            st.write(f"Average Arrivals per Hour: {len(arrivals) / duration:.2f}")

            # Create a DataFrame for visualization
            if arrivals:
                arrivals_df = pd.DataFrame(arrivals)

                # Count arrivals by hour
                hour_counts = arrivals_df.groupby('hour').size().reset_index(name='count')

                # Create a bar chart of arrivals by hour
                fig = px.bar(
                    hour_counts,
                    x='hour',
                    y='count',
                    title="Patient Arrivals by Hour",
                    labels={'hour': 'Hour of Day', 'count': 'Number of Arrivals'}
                )
                st.plotly_chart(fig)

                # Count arrivals by department
                dept_counts = arrivals_df['department'].value_counts().reset_index()
                dept_counts.columns = ['Department', 'Count']

                # Create a pie chart of arrivals by department
                fig = px.pie(
                    dept_counts,
                    values='Count',
                    names='Department',
                    title="Patient Arrivals by Department"
                )
                st.plotly_chart(fig)

                # Display the arrival data
                st.subheader("Arrival Data")
                st.dataframe(arrivals_df)

    with poisson_tab2:
        st.subheader("Patient Arrival Patterns")

        # Parameters for arrival pattern visualization
        duration = st.slider("Duration to Visualize (hours)", 24, 168, 72, 24)
        start_hour_pattern = st.selectbox("Start Hour for Pattern", list(range(24)), 8, key="pattern_start")
        day_of_week_pattern = st.selectbox("Start Day for Pattern",
                                         ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                         0, key="pattern_day")

        # Plot arrival pattern
        fig = st.session_state.poisson_model.plot_arrival_pattern(
            duration_hours=duration,
            start_hour=start_hour_pattern,
            day_of_week=day_of_week_pattern
        )
        st.pyplot(fig)

        # Explain the pattern
        st.markdown("""
        ### Arrival Pattern Explanation

        The graph above shows the expected number of patient arrivals per hour over time. The pattern is influenced by:

        1. **Time of Day**: Arrivals typically peak in the morning and evening, with fewer arrivals overnight.
        2. **Day of Week**: Weekdays generally have more arrivals than weekends.
        3. **Base Arrival Rate**: The overall average number of arrivals per hour.

        This pattern is modeled using a Poisson process with a time-varying rate parameter λ(t), which is the product of:
        - Base arrival rate
        - Hourly modifier (time-of-day effect)
        - Daily modifier (day-of-week effect)

        In a Poisson process, the number of arrivals in any time interval follows a Poisson distribution, and the inter-arrival times follow an exponential distribution.
        """)

    with poisson_tab3:
        st.subheader("Department Distribution")

        # Display current department probabilities
        st.write("Current Department Distribution:")
        dept_probs = st.session_state.poisson_model.department_probs
        dept_df = pd.DataFrame({
            'Department': list(dept_probs.keys()),
            'Probability': list(dept_probs.values())
        })
        st.dataframe(dept_df)

        # Allow user to modify department probabilities
        st.write("Modify Department Distribution:")

        col1, col2, col3 = st.columns(3)

        with col1:
            er_prob = st.slider("ER Probability", 0.0, 1.0, dept_probs.get('ER', 0.7), 0.05)

        with col2:
            ward_prob = st.slider("Ward Probability", 0.0, 1.0, dept_probs.get('Ward', 0.2), 0.05)

        with col3:
            icu_prob = st.slider("ICU Probability", 0.0, 1.0, dept_probs.get('ICU', 0.1), 0.05)

        # Normalize probabilities to sum to 1
        total = er_prob + ward_prob + icu_prob
        if total > 0:
            er_prob_norm = er_prob / total
            ward_prob_norm = ward_prob / total
            icu_prob_norm = icu_prob / total

            new_probs = {
                'ER': er_prob_norm,
                'Ward': ward_prob_norm,
                'ICU': icu_prob_norm
            }

            if st.button("Update Department Distribution"):
                try:
                    st.session_state.poisson_model.set_department_probs(new_probs)
                    st.success("Department distribution updated successfully!")

                    # Plot the updated distribution
                    fig = st.session_state.poisson_model.plot_department_distribution()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error updating department distribution: {e}")
        else:
            st.error("Total probability must be greater than 0")

# Hospital Queuing Theory page
elif page == "Hospital Queuing Theory":
    st.title("⏱️ Hospital Queuing Theory Model")

    st.markdown("""
    Queuing theory is a mathematical study of waiting lines or queues. In healthcare, it helps model
    patient flow through departments, waiting times, and resource utilization.
    """)

    # Create tabs for different queuing theory options
    queue_tab1, queue_tab2, queue_tab3 = st.tabs(["Simulation", "Service Time Distributions", "Department Configuration"])

    with queue_tab1:
        st.subheader("Simulate Hospital Queues")

        col1, col2 = st.columns(2)

        with col1:
            sim_duration = st.slider("Simulation Duration (hours)", 24, 168, 72, 24)
            arrival_rate_queue = st.slider("Patient Arrival Rate (per hour)", 0.5, 10.0, 3.0, 0.5)

        if st.button("Run Simulation"):
            with st.spinner("Running simulation..."):
                # Run the simulation
                results = st.session_state.queuing_model.simulate(
                    duration=sim_duration,
                    arrival_rate=arrival_rate_queue
                )

            # Display results
            st.subheader("Simulation Results")

            # Create a DataFrame for the summary statistics
            summary_data = []
            for dept, stats in results.items():
                summary_data.append({
                    'Department': dept,
                    'Avg. Waiting Time (hours)': stats['avg_waiting_time'],
                    'Avg. Service Time (hours)': stats['avg_service_time'],
                    'Avg. Queue Length': stats['avg_queue_length'],
                    'Avg. Utilization (%)': stats['avg_utilization'] * 100
                })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df)

            # Create visualizations

            # Waiting time by department
            fig = px.bar(
                summary_df,
                x='Department',
                y='Avg. Waiting Time (hours)',
                title="Average Waiting Time by Department",
                color='Department'
            )
            st.plotly_chart(fig)

            # Utilization by department
            fig = px.bar(
                summary_df,
                x='Department',
                y='Avg. Utilization (%)',
                title="Resource Utilization by Department",
                color='Department'
            )
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig)

            # Queue length by department
            fig = px.bar(
                summary_df,
                x='Department',
                y='Avg. Queue Length',
                title="Average Queue Length by Department",
                color='Department'
            )
            st.plotly_chart(fig)

    with queue_tab2:
        st.subheader("Service Time Distributions")

        # Display service time distributions
        fig = st.session_state.queuing_model.plot_service_time_distributions()
        st.pyplot(fig)

        # Explain the distributions
        st.markdown("""
        ### Service Time Distribution Explanation

        Different hospital departments have different service time distributions:

        1. **ER (Emergency Room)**: Service times follow an **exponential distribution**, which is memoryless.
           This means the probability of a patient being discharged in the next hour is independent of how long
           they've already been there. This is appropriate for emergency care where some patients are treated
           quickly while others may require extended care.

        2. **Ward**: Service times follow a **normal distribution**, reflecting more predictable lengths of stay
           for general hospital care. Most patients stay for a duration close to the mean, with fewer patients
           having very short or very long stays.

        3. **ICU (Intensive Care Unit)**: Service times also follow a **normal distribution** but with a higher
           mean and standard deviation, reflecting the longer and more variable stays typical of intensive care.

        These distributions are used in the queuing model to simulate realistic patient flow through the hospital.
        """)

    with queue_tab3:
        st.subheader("Department Configuration")

        # Display current department configuration
        dept_config = st.session_state.queuing_model.departments

        for dept, config in dept_config.items():
            st.write(f"### {dept} Department")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"Number of Servers: {config['servers']}")
                st.write(f"Service Time Distribution: {config['service_time_dist'].capitalize()}")

                if config['service_time_dist'] == 'exponential':
                    st.write(f"Mean Service Time: {config['service_time_params']['scale']:.1f} hours")
                elif config['service_time_dist'] == 'normal':
                    st.write(f"Mean Service Time: {config['service_time_params']['loc']:.1f} hours")
                    st.write(f"Standard Deviation: {config['service_time_params']['scale']:.1f} hours")

            with col2:
                st.write(f"Queue Discipline: {config['queue_discipline'].upper()}")

                if config['queue_discipline'] == 'priority':
                    st.write("Priority-based queuing (higher severity patients are treated first)")
                else:
                    st.write("First-in, first-out queuing (patients are treated in order of arrival)")

            st.markdown("---")

        # Explain queuing theory concepts
        st.markdown("""
        ### Queuing Theory Concepts

        The hospital queuing model uses several key concepts from queuing theory:

        1. **Arrival Process**: Patients arrive according to a Poisson process with rate λ.

        2. **Service Process**: Each department has its own service time distribution.

        3. **Number of Servers**: Each department has multiple servers (doctors, beds, etc.).

        4. **Queue Discipline**: The order in which patients are served:
           - FIFO (First-In, First-Out): Patients are served in order of arrival
           - Priority: Patients with higher severity are served first

        5. **Performance Metrics**:
           - Waiting Time: Time spent in queue before service
           - Service Time: Time spent receiving care
           - Queue Length: Number of patients waiting
           - Utilization: Proportion of time servers are busy

        This model helps hospital administrators optimize resource allocation and improve patient flow.
        """)

# Bayesian Networks page
elif page == "Bayesian Networks":
    st.title("🔄 Bayesian Networks for Patient Diagnosis")

    # Create info box explaining Bayesian Networks
    bayesian_info = """
    <strong>Bayesian Networks</strong> are probabilistic graphical models that represent a set of variables and their
    conditional dependencies via a directed acyclic graph (DAG). In healthcare, they can model the relationships
    between patient symptoms, diseases, and treatments, allowing for probabilistic inference and decision making.
    """

    st.markdown(ui_theme.create_info_box(bayesian_info, type='info'), unsafe_allow_html=True)

    # Create tabs for different Bayesian Network options
    bayesian_tab1, bayesian_tab2, bayesian_tab3 = st.tabs(["Diagnosis", "Network Visualization", "Treatment Recommendations"])

    with bayesian_tab1:
        st.subheader("Patient Diagnosis using Bayesian Networks")

        # Create columns for patient information and symptoms
        col1, col2 = st.columns(2)

        with col1:
            st.write("### Patient Demographics")

            age_group = st.selectbox(
                "Age Group",
                options=["Young (0-30)", "Middle-aged (31-60)", "Elderly (61+)"],
                index=1
            )

            gender = st.radio("Gender", ["Male", "Female"])

            smoking = st.radio("Smoking Status", ["Non-smoker", "Smoker"])

            # Map selections to model values
            age_map = {
                "Young (0-30)": 0,
                "Middle-aged (31-60)": 1,
                "Elderly (61+)": 2
            }

            gender_map = {
                "Male": 0,
                "Female": 1
            }

            smoking_map = {
                "Non-smoker": 0,
                "Smoker": 1
            }

        with col2:
            st.write("### Patient Symptoms")

            chest_pain = st.checkbox("Chest Pain")
            shortness_of_breath = st.checkbox("Shortness of Breath")
            fatigue = st.checkbox("Fatigue")
            cough = st.checkbox("Cough")
            excessive_thirst = st.checkbox("Excessive Thirst")
            ecg_abnormal = st.checkbox("Abnormal ECG")

        # Create evidence dictionary for Bayesian inference
        if st.button("Diagnose Patient"):
            evidence = {
                'Age': age_map[age_group],
                'Gender': gender_map[gender],
                'Smoking': smoking_map[smoking],
                'ChestPain': int(chest_pain),
                'ShortnessOfBreath': int(shortness_of_breath),
                'Fatigue': int(fatigue),
                'Cough': int(cough),
                'ExcessiveThirst': int(excessive_thirst),
                'ECGAbnormal': int(ecg_abnormal)
            }

            # Get diagnosis
            diagnosis = st.session_state.bayesian_network.predict_diagnosis(evidence)

            # Display diagnosis results
            st.subheader("Diagnosis Results")

            # Create a DataFrame for the results
            diagnosis_df = pd.DataFrame({
                'Disease': list(diagnosis.keys()),
                'Probability': list(diagnosis.values())
            })

            # Sort by probability
            diagnosis_df = diagnosis_df.sort_values('Probability', ascending=False)

            # Display as a bar chart
            fig = px.bar(
                diagnosis_df,
                x='Disease',
                y='Probability',
                color='Probability',
                color_continuous_scale='Reds',
                title='Disease Probability Based on Symptoms',
                labels={'Probability': 'Probability of Disease'}
            )

            fig.update_layout(
                xaxis_title='Disease',
                yaxis_title='Probability',
                yaxis_range=[0, 1]
            )

            st.plotly_chart(fig)

            # Get treatment recommendations
            disease_evidence = {
                'HeartDisease': int(diagnosis['HeartDisease'] > 0.5),
                'LungDisease': int(diagnosis['LungDisease'] > 0.5),
                'Diabetes': int(diagnosis['Diabetes'] > 0.5)
            }

            treatments = st.session_state.bayesian_network.recommend_treatment(disease_evidence)

            # Display treatment recommendations
            st.subheader("Treatment Recommendations")

            # Create a DataFrame for the treatments
            treatment_df = pd.DataFrame({
                'Treatment': list(treatments.keys()),
                'Recommendation Strength': list(treatments.values())
            })

            # Sort by recommendation strength
            treatment_df = treatment_df.sort_values('Recommendation Strength', ascending=False)

            # Display as a bar chart
            fig = px.bar(
                treatment_df,
                x='Treatment',
                y='Recommendation Strength',
                color='Recommendation Strength',
                color_continuous_scale='Blues',
                title='Treatment Recommendations Based on Diagnosis',
                labels={'Recommendation Strength': 'Recommendation Strength'}
            )

            fig.update_layout(
                xaxis_title='Treatment',
                yaxis_title='Recommendation Strength',
                yaxis_range=[0, 1]
            )

            st.plotly_chart(fig)

            # Display interpretation
            st.subheader("Interpretation")

            # Create interpretation based on highest probability disease
            top_disease = diagnosis_df.iloc[0]['Disease']
            top_prob = diagnosis_df.iloc[0]['Probability']

            if top_prob > 0.7:
                confidence = "high"
            elif top_prob > 0.4:
                confidence = "moderate"
            else:
                confidence = "low"

            interpretation = f"""
            Based on the provided symptoms and patient information, there is a **{confidence} probability**
            ({top_prob:.1%}) that the patient has **{top_disease}**.
            """

            st.markdown(interpretation)

            # Add treatment recommendation
            top_treatment = treatment_df.iloc[0]['Treatment']
            top_treatment_prob = treatment_df.iloc[0]['Recommendation Strength']

            if top_treatment_prob > 0.7:
                treatment_text = f"**{top_treatment}** is strongly recommended."
            elif top_treatment_prob > 0.4:
                treatment_text = f"Consider **{top_treatment}** as a treatment option."
            else:
                treatment_text = f"**{top_treatment}** might be considered, but further evaluation is needed."

            st.markdown(treatment_text)

    with bayesian_tab2:
        st.subheader("Bayesian Network Visualization")

        # Plot the Bayesian Network
        fig = st.session_state.bayesian_network.plot_network()
        st.pyplot(fig)

        # Explanation of the network
        st.markdown("""
        ### Network Structure Explanation

        The Bayesian Network above shows the probabilistic relationships between:

        - **Demographics** (blue nodes): Age, Gender, Smoking status
        - **Diseases** (red nodes): Heart Disease, Lung Disease, Diabetes
        - **Symptoms** (green nodes): Chest Pain, ECG Abnormalities, Cough, etc.
        - **Treatments** (yellow nodes): Emergency Care, Medications, etc.

        Arrows indicate causal relationships. For example:
        - Age influences the probability of Heart Disease and Diabetes
        - Smoking influences the probability of Heart Disease and Lung Disease
        - Heart Disease causes symptoms like Chest Pain and Fatigue

        This network allows us to perform probabilistic inference: given observed symptoms,
        we can calculate the probability of different diseases and recommend appropriate treatments.
        """)

    with bayesian_tab3:
        st.subheader("Treatment Recommendation System")

        st.markdown("""
        The Bayesian Network can recommend treatments based on diagnosed conditions.
        Select the conditions below to see treatment recommendations:
        """)

        # Create columns for conditions
        cond_col1, cond_col2, cond_col3 = st.columns(3)

        with cond_col1:
            heart_disease = st.checkbox("Heart Disease")

        with cond_col2:
            lung_disease = st.checkbox("Lung Disease")

        with cond_col3:
            diabetes = st.checkbox("Diabetes")

        if st.button("Get Treatment Recommendations"):
            # Create evidence dictionary
            disease_evidence = {
                'HeartDisease': int(heart_disease),
                'LungDisease': int(lung_disease),
                'Diabetes': int(diabetes)
            }

            # Get treatment recommendations
            treatments = st.session_state.bayesian_network.recommend_treatment(disease_evidence)

            # Display treatment recommendations
            st.subheader("Treatment Recommendations")

            # Create a DataFrame for the treatments
            treatment_df = pd.DataFrame({
                'Treatment': list(treatments.keys()),
                'Recommendation Strength': list(treatments.values())
            })

            # Sort by recommendation strength
            treatment_df = treatment_df.sort_values('Recommendation Strength', ascending=False)

            # Display as a bar chart
            fig = px.bar(
                treatment_df,
                x='Treatment',
                y='Recommendation Strength',
                color='Recommendation Strength',
                color_continuous_scale='Blues',
                title='Treatment Recommendations',
                labels={'Recommendation Strength': 'Recommendation Strength'}
            )

            fig.update_layout(
                xaxis_title='Treatment',
                yaxis_title='Recommendation Strength',
                yaxis_range=[0, 1]
            )

            st.plotly_chart(fig)

            # Display recommendations as text
            st.subheader("Detailed Recommendations")

            for index, row in treatment_df.iterrows():
                treatment = row['Treatment']
                strength = row['Recommendation Strength']

                if strength > 0.7:
                    recommendation = f"**Strongly Recommended**: {treatment} (Confidence: {strength:.1%})"
                elif strength > 0.4:
                    recommendation = f"**Recommended**: {treatment} (Confidence: {strength:.1%})"
                elif strength > 0.2:
                    recommendation = f"**Consider**: {treatment} (Confidence: {strength:.1%})"
                else:
                    recommendation = f"**Not Recommended**: {treatment} (Confidence: {strength:.1%})"

                st.markdown(recommendation)

    st.markdown("""
    Queuing theory is a mathematical study of waiting lines or queues. In healthcare, it helps model
    patient flow through departments, waiting times, and resource utilization.
    """)

    # Create tabs for different queuing theory options
    queue_tab1, queue_tab2, queue_tab3 = st.tabs(["Simulation", "Service Time Distributions", "Department Configuration"])

    with queue_tab1:
        st.subheader("Simulate Hospital Queues")

        col1, col2 = st.columns(2)

        with col1:
            sim_duration = st.slider("Simulation Duration (hours)", 24, 168, 72, 24)
            arrival_rate_queue = st.slider("Patient Arrival Rate (per hour)", 0.5, 10.0, 3.0, 0.5)

        if st.button("Run Simulation"):
            with st.spinner("Running simulation..."):
                # Run the simulation
                results = st.session_state.queuing_model.simulate(
                    duration=sim_duration,
                    arrival_rate=arrival_rate_queue
                )

            # Display results
            st.subheader("Simulation Results")

            # Create a DataFrame for the summary statistics
            summary_data = []
            for dept, stats in results.items():
                summary_data.append({
                    'Department': dept,
                    'Avg. Waiting Time (hours)': stats['avg_waiting_time'],
                    'Avg. Service Time (hours)': stats['avg_service_time'],
                    'Avg. Queue Length': stats['avg_queue_length'],
                    'Avg. Utilization (%)': stats['avg_utilization'] * 100
                })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df)

            # Create visualizations

            # Waiting time by department
            fig = px.bar(
                summary_df,
                x='Department',
                y='Avg. Waiting Time (hours)',
                title="Average Waiting Time by Department",
                color='Department'
            )
            st.plotly_chart(fig)

            # Utilization by department
            fig = px.bar(
                summary_df,
                x='Department',
                y='Avg. Utilization (%)',
                title="Resource Utilization by Department",
                color='Department'
            )
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig)

            # Queue length by department
            fig = px.bar(
                summary_df,
                x='Department',
                y='Avg. Queue Length',
                title="Average Queue Length by Department",
                color='Department'
            )
            st.plotly_chart(fig)

    with queue_tab2:
        st.subheader("Service Time Distributions")

        # Display service time distributions
        fig = st.session_state.queuing_model.plot_service_time_distributions()
        st.pyplot(fig)

        # Explain the distributions
        st.markdown("""
        ### Service Time Distribution Explanation

        Different hospital departments have different service time distributions:

        1. **ER (Emergency Room)**: Service times follow an **exponential distribution**, which is memoryless.
           This means the probability of a patient being discharged in the next hour is independent of how long
           they've already been there. This is appropriate for emergency care where some patients are treated
           quickly while others may require extended care.

        2. **Ward**: Service times follow a **normal distribution**, reflecting more predictable lengths of stay
           for general hospital care. Most patients stay for a duration close to the mean, with fewer patients
           having very short or very long stays.

        3. **ICU (Intensive Care Unit)**: Service times also follow a **normal distribution** but with a higher
           mean and standard deviation, reflecting the longer and more variable stays typical of intensive care.

        These distributions are used in the queuing model to simulate realistic patient flow through the hospital.
        """)

    with queue_tab3:
        st.subheader("Department Configuration")

        # Display current department configuration
        dept_config = st.session_state.queuing_model.departments

        for dept, config in dept_config.items():
            st.write(f"### {dept} Department")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"Number of Servers: {config['servers']}")
                st.write(f"Service Time Distribution: {config['service_time_dist'].capitalize()}")

                if config['service_time_dist'] == 'exponential':
                    st.write(f"Mean Service Time: {config['service_time_params']['scale']:.1f} hours")
                elif config['service_time_dist'] == 'normal':
                    st.write(f"Mean Service Time: {config['service_time_params']['loc']:.1f} hours")
                    st.write(f"Standard Deviation: {config['service_time_params']['scale']:.1f} hours")

            with col2:
                st.write(f"Queue Discipline: {config['queue_discipline'].upper()}")

                if config['queue_discipline'] == 'priority':
                    st.write("Priority-based queuing (higher severity patients are treated first)")
                else:
                    st.write("First-in, first-out queuing (patients are treated in order of arrival)")

            st.markdown("---")

        # Explain queuing theory concepts
        st.markdown("""
        ### Queuing Theory Concepts

        The hospital queuing model uses several key concepts from queuing theory:

        1. **Arrival Process**: Patients arrive according to a Poisson process with rate λ.

        2. **Service Process**: Each department has its own service time distribution.

        3. **Number of Servers**: Each department has multiple servers (doctors, beds, etc.).

        4. **Queue Discipline**: The order in which patients are served:
           - FIFO (First-In, First-Out): Patients are served in order of arrival
           - Priority: Patients with higher severity are served first

        5. **Performance Metrics**:
           - Waiting Time: Time spent in queue before service
           - Service Time: Time spent receiving care
           - Queue Length: Number of patients waiting
           - Utilization: Proportion of time servers are busy

        This model helps hospital administrators optimize resource allocation and improve patient flow.
        """)

# Diagnosis Prediction page
elif page == "Diagnosis Prediction":
    st.title("🔬 Diagnosis Prediction")

    # Create tabs for different diagnosis options
    diag_tab1, diag_tab2, diag_tab3 = st.tabs(["Predict Diagnosis", "Model Performance", "Feature Importance"])

    with diag_tab1:
        st.subheader("Predict Patient Diagnosis")

        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Patient Age", 18, 90, 50)
            los = st.slider("Length of Stay (days)", 1, 14, 5)
            department = st.selectbox("Department", ['ER', 'Ward', 'ICU'])
            condition = st.selectbox("Medical Condition", ['Cardiac', 'Neuro', 'Injury'])

        if st.button("Predict Diagnosis"):
            # Create patient data
            patient_data = {
                'Age': age,
                'Length_of_Stay': los,
                'Department': department,
                'Medical_Condition': condition
            }

            # Make prediction
            diagnosis, probability = st.session_state.diagnosis_model.predict(patient_data)

            # Display prediction
            st.subheader("Prediction Results")
            st.write(f"Predicted Diagnosis: **{diagnosis}**")
            st.write(f"Confidence: {probability:.2%}")

            # Create a gauge chart for the probability
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            st.plotly_chart(fig)

    with diag_tab2:
        st.subheader("Model Performance")

        if st.button("Evaluate Model"):
            with st.spinner("Training and evaluating model..."):
                model = SimpleDiagnosisModel()
                results = model.train("../data/hospital_patient_dataset.csv")

            # Display results
            st.write("Model trained successfully!")

            # Test with sample patients
            test_patients = [
                {'Age': 25, 'Length_of_Stay': 3, 'Department': 'ER', 'Medical_Condition': 'Cardiac'},
                {'Age': 45, 'Length_of_Stay': 7, 'Department': 'Ward', 'Medical_Condition': 'Neuro'},
                {'Age': 70, 'Length_of_Stay': 12, 'Department': 'ICU', 'Medical_Condition': 'Injury'}
            ]

            # Create a dataframe to display results
            test_results = []
            for patient in test_patients:
                diagnosis, probability = model.predict(patient)
                test_results.append({
                    'Age': patient['Age'],
                    'Length of Stay': patient['Length_of_Stay'],
                    'Department': patient['Department'],
                    'Medical Condition': patient['Medical_Condition'],
                    'Predicted Diagnosis': diagnosis,
                    'Probability': f"{probability:.2%}"
                })

            # Display test results
            st.subheader("Test Predictions")
            st.dataframe(pd.DataFrame(test_results))

    with diag_tab3:
        st.subheader("Model Factors")

        # Display information about the model's factors
        st.markdown("""
        ### Factors Influencing Diagnosis Prediction

        Our diagnosis prediction model considers several key factors:

        1. **Department**: Different departments have different diagnosis patterns
           - ER: More acute conditions
           - Ward: More stable conditions
           - ICU: More severe conditions

        2. **Medical Condition**: The presenting medical condition is a strong predictor
           - Cardiac: Heart-related issues
           - Neuro: Neurological issues
           - Injury: Trauma and injuries

        3. **Age**: Patient age affects likelihood of different diagnoses
           - Young patients (< 40 years)
           - Middle-aged patients (40-60 years)
           - Elderly patients (> 60 years)

        4. **Length of Stay**: Duration of hospitalization correlates with diagnosis severity
           - Short stays (< 5 days)
           - Medium stays (5-10 days)
           - Long stays (> 10 days)

        The model calculates probabilities based on these factors and their interactions.
        """)

# Data Analysis page
elif page == "Data Analysis":
    st.title("📊 Hospital Data Analysis")

    # Create tabs for different analysis options
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["Basic Statistics", "Patient Flow Analysis", "Length of Stay Analysis"])

    with analysis_tab1:
        st.subheader("Basic Statistics")

        # Get EDA results
        eda_results = exploratory_data_analysis(df)

        # Display basic statistics
        st.write("Dataset Shape:", df.shape)
        st.write("Basic Statistics:")
        st.dataframe(eda_results['basic_stats'])

        # Display department distribution
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Department Distribution")
            fig = px.pie(
                values=eda_results['department_counts'].values,
                names=eda_results['department_counts'].index,
                title="Patient Distribution by Department"
            )
            st.plotly_chart(fig)

        with col2:
            st.subheader("Medical Condition Distribution")
            fig = px.pie(
                values=eda_results['condition_counts'].values,
                names=eda_results['condition_counts'].index,
                title="Patient Distribution by Medical Condition"
            )
            st.plotly_chart(fig)

        # Display diagnosis distribution
        st.subheader("Diagnosis Distribution")
        fig = px.bar(
            x=eda_results['diagnosis_counts'].index,
            y=eda_results['diagnosis_counts'].values,
            title="Patient Distribution by Diagnosis",
            labels={'x': 'Diagnosis', 'y': 'Count'}
        )
        st.plotly_chart(fig)

    with analysis_tab2:
        st.subheader("Patient Flow Analysis")

        # Create Sankey diagram
        st.write("Patient Flow Sankey Diagram")
        fig = plot_patient_flow_sankey(df)
        st.plotly_chart(fig)

        # Department to diagnosis relationship
        st.subheader("Department to Diagnosis Relationship")
        dept_diag = pd.crosstab(df['Department'], df['Diagnosis'])

        # Convert to percentages
        dept_diag_pct = dept_diag.div(dept_diag.sum(axis=1), axis=0) * 100

        # Plot heatmap
        fig = px.imshow(
            dept_diag_pct,
            text_auto='.1f',
            labels=dict(x="Diagnosis", y="Department", color="Percentage"),
            title="Department to Diagnosis Relationship (%)",
            color_continuous_scale="Blues"
        )
        fig.update_layout(
            xaxis_title="Diagnosis",
            yaxis_title="Department"
        )
        st.plotly_chart(fig)

        # Display raw counts
        st.write("Raw Counts:")
        st.dataframe(dept_diag)

    with analysis_tab3:
        st.subheader("Length of Stay Analysis")

        # Length of stay by department
        st.write("Length of Stay by Department")
        fig = px.box(
            df,
            x="Department",
            y="Length_of_Stay",
            color="Department",
            title="Length of Stay by Department",
            labels={"Length_of_Stay": "Length of Stay (days)"}
        )
        st.plotly_chart(fig)

        # Length of stay by medical condition
        st.write("Length of Stay by Medical Condition")
        fig = px.box(
            df,
            x="Medical_Condition",
            y="Length_of_Stay",
            color="Medical_Condition",
            title="Length of Stay by Medical Condition",
            labels={"Length_of_Stay": "Length of Stay (days)"}
        )
        st.plotly_chart(fig)

        # Length of stay by diagnosis
        st.write("Length of Stay by Diagnosis")
        fig = px.box(
            df,
            x="Diagnosis",
            y="Length_of_Stay",
            color="Diagnosis",
            title="Length of Stay by Diagnosis",
            labels={"Length_of_Stay": "Length of Stay (days)"}
        )
        st.plotly_chart(fig)

        # Age vs Length of Stay
        st.write("Age vs Length of Stay")
        fig = px.scatter(
            df,
            x="Age",
            y="Length_of_Stay",
            color="Department",
            size="Length_of_Stay",
            hover_data=["Medical_Condition", "Diagnosis"],
            title="Age vs Length of Stay",
            labels={"Length_of_Stay": "Length of Stay (days)"}
        )
        st.plotly_chart(fig)

# 3D Visualizations page
elif page == "3D Visualizations":
    st.title("🔄 3D Interactive Visualizations")

    # Create info box explaining 3D visualizations
    viz_info = """
    <strong>3D Interactive Visualizations</strong> provide an immersive way to explore complex healthcare data and models.
    These visualizations allow you to see relationships and patterns that might not be apparent in traditional 2D charts.
    """

    st.markdown(ui_theme.create_info_box(viz_info, type='info'), unsafe_allow_html=True)

    # Create tabs for different 3D visualization options
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Patient Flow 3D", "Queuing Model 3D", "Hidden Markov Model 3D"])

    with viz_tab1:
        st.subheader("3D Patient Flow Visualization")

        # Get transition matrix from the Markov model
        transition_matrix = st.session_state.patient_flow_model.transition_matrix
        department_names = ["ER", "Ward", "ICU", "Discharged"]

        # Create 3D visualization
        fig = vis_3d.create_3d_patient_flow(transition_matrix, department_names)

        # Display the visualization
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ### 3D Patient Flow Explanation

        This 3D visualization shows patient transitions between hospital departments:

        - **Nodes** represent hospital departments
        - **Edges** represent patient transitions between departments
        - **Edge thickness** indicates transition probability
        - **Edge color** corresponds to the source department

        The 3D layout allows you to see the complete flow network from different angles.
        You can rotate, zoom, and pan the visualization to explore different perspectives.
        """)

    with viz_tab2:
        st.subheader("3D Hospital Queuing Visualization")

        # Run a simulation to get queuing data
        queuing_data = st.session_state.queuing_model.simulate(duration=72, arrival_rate=3.0)

        # Create 3D visualization
        fig = vis_3d.create_3d_queuing_visualization(queuing_data)

        # Display the visualization
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ### 3D Queuing Model Explanation

        This 3D visualization shows the queuing dynamics in different hospital departments:

        - **Nodes** represent hospital departments
        - **Cylinders** represent queue lengths
        - **Vertical bars** represent resource utilization

        The 3D layout allows you to compare queue lengths and utilization across departments.
        You can rotate, zoom, and pan the visualization to explore different perspectives.
        """)

    with viz_tab3:
        st.subheader("3D Hidden Markov Model Visualization")

        # Create 3D visualization of the HMM
        fig = vis_3d.create_3d_hmm_visualization(st.session_state.hidden_markov_model)

        # Display the visualization
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ### 3D Hidden Markov Model Explanation

        This 3D visualization shows the structure of the Hidden Markov Model:

        - **Red nodes** represent hidden states (patient health states)
        - **Blue nodes** represent observations (symptoms, vital signs)
        - **Red edges** represent transition probabilities between states
        - **Blue edges** represent emission probabilities from states to observations

        The 3D layout separates hidden states and observations into different planes,
        making it easier to understand the two-layer structure of the HMM.
        You can rotate, zoom, and pan the visualization to explore different perspectives.
        """)

    # Add a section for combined 3D visualization
    st.subheader("Integrated Hospital System Visualization")

    st.markdown("""
    The 3D visualizations above provide different perspectives on the hospital system.
    By combining these models, we can create a comprehensive understanding of patient flow,
    from arrival to diagnosis, treatment, and discharge.

    ### Key Insights from 3D Visualizations:

    1. **Patient Flow**: The 3D network visualization reveals the most common patient pathways through the hospital.

    2. **Resource Utilization**: The 3D queuing visualization shows which departments have the highest utilization and longest queues.

    3. **Health State Transitions**: The 3D HMM visualization shows how patients transition between different health states.

    4. **Bottlenecks**: By comparing the visualizations, we can identify bottlenecks in the hospital system.

    5. **Optimization Opportunities**: The 3D visualizations help identify opportunities for optimizing resource allocation and patient flow.
    """)





# Footer
st.markdown("---")
st.markdown("Intelligent Patient Flow and Diagnosis Modeling in Smart Hospitals | Stochastic Processes Project")
# Remove any stray div tags that might be showing up
st.markdown("<style>div.stMarkdown div {display: block !important;}</style>", unsafe_allow_html=True)
