
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from utils import create_transition_matrix

class HospitalStochasticModel:
    """
    A stochastic model for hospital patient flow using Markov chains and other stochastic processes.
    """

    def __init__(self, data_path=None):
        """
        Initialize the stochastic model.

        Args:
            data_path (str, optional): Path to the data file. If provided, transition probabilities
                                      will be calculated from the data.
        """
        # Default states and transition matrix
        self.states = ['ER', 'Ward', 'ICU', 'Discharged']
        self.transition_matrix = np.array([
            [0.1, 0.6, 0.2, 0.1],  # ER
            [0.0, 0.2, 0.3, 0.5],  # Ward
            [0.0, 0.1, 0.4, 0.5],  # ICU
            [0.0, 0.0, 0.0, 1.0]   # Discharged
        ])

        # Service time distributions (in hours) for each department
        self.service_time_distributions = {
            'ER': stats.expon(scale=4),      # Exponential with mean 4 hours
            'Ward': stats.norm(loc=72, scale=24),  # Normal with mean 3 days, std 1 day
            'ICU': stats.norm(loc=120, scale=48)   # Normal with mean 5 days, std 2 days
        }

        # Arrival rate (patients per hour) - Poisson process
        self.arrival_rate = 2.5  # 2.5 patients per hour on average

        # If data is provided, update the model parameters
        if data_path:
            self.update_from_data(data_path)

    def update_from_data(self, data_path):
        """
        Update model parameters from data.

        Args:
            data_path (str): Path to the data file
        """
        try:
            df = pd.read_csv(data_path)

            # Update states and transition matrix
            self.states, self.transition_matrix = create_transition_matrix(df)

            # Update service time distributions
            for dept in ['ER', 'Ward', 'ICU']:
                if dept in df['Department'].unique():
                    # Convert Length_of_Stay to hours (assuming it's in days in the dataset)
                    los_hours = df[df['Department'] == dept]['Length_of_Stay'] * 24

                    # Fit distribution
                    if len(los_hours) > 0:
                        # Try to fit exponential distribution for ER
                        if dept == 'ER':
                            # Exponential distribution has one parameter
                            self.service_time_distributions[dept] = stats.expon(scale=los_hours.mean())
                        else:
                            # Normal distribution for Ward and ICU
                            self.service_time_distributions[dept] = stats.norm(loc=los_hours.mean(), scale=los_hours.std())

            # Update arrival rate (if we had timestamp data, we could calculate this more accurately)
            # For now, we'll use a simple approximation
            self.arrival_rate = len(df) / 30  # Assuming the data spans 30 days

        except Exception as e:
            print(f"Error updating model from data: {e}")
            # Keep default parameters

    def simulate_patient_flow(self, start_state, steps, include_times=False):
        """
        Simulate patient flow through the hospital.

        Args:
            start_state (str): Starting department
            steps (int): Maximum number of steps to simulate
            include_times (bool): Whether to include service times in the output

        Returns:
            list or tuple: If include_times is False, returns a list of departments.
                          If include_times is True, returns a tuple of (departments, times).
        """
        if start_state not in self.states:
            raise ValueError(f"Start state must be one of {self.states}")

        current_state = self.states.index(start_state)
        path = [start_state]
        times = []

        for _ in range(steps):
            # Sample service time for current department
            if self.states[current_state] != 'Discharged':
                service_time = max(1, self.service_time_distributions[self.states[current_state]].rvs())
                times.append(service_time)

            # Determine next state
            next_state_idx = np.random.choice(len(self.states), p=self.transition_matrix[current_state])
            next_state = self.states[next_state_idx]
            path.append(next_state)

            # If discharged, end simulation
            if next_state == 'Discharged':
                break

            current_state = next_state_idx

        if include_times:
            return path, times
        else:
            return path

    def simulate_arrivals(self, duration_hours):
        """
        Simulate patient arrivals using a Poisson process.

        Args:
            duration_hours (float): Duration to simulate in hours

        Returns:
            list: Arrival times in hours
        """
        # Poisson process - exponential inter-arrival times
        inter_arrival_times = stats.expon(scale=1/self.arrival_rate).rvs(size=100)  # Generate more than needed
        arrival_times = np.cumsum(inter_arrival_times)

        # Keep only arrivals within the duration
        arrival_times = arrival_times[arrival_times <= duration_hours]

        return arrival_times.tolist()

    def simulate_hospital_day(self, duration_hours=24):
        """
        Simulate a day in the hospital.

        Args:
            duration_hours (float): Duration to simulate in hours

        Returns:
            dict: Simulation results
        """
        # Simulate arrivals
        arrival_times = self.simulate_arrivals(duration_hours)

        # Simulate patient flow for each arrival
        results = []

        for arrival_time in arrival_times:
            # 70% start in ER, 20% in Ward, 10% in ICU
            start_dept = np.random.choice(['ER', 'Ward', 'ICU'], p=[0.7, 0.2, 0.1])

            # Simulate this patient's journey
            path, times = self.simulate_patient_flow(start_dept, 10, include_times=True)

            # Calculate absolute times
            absolute_times = [arrival_time]
            for t in times:
                absolute_times.append(absolute_times[-1] + t)

            results.append({
                'arrival_time': arrival_time,
                'start_department': start_dept,
                'path': path,
                'service_times': times,
                'absolute_times': absolute_times,
                'total_los': sum(times)
            })

        return results

    def plot_transition_matrix(self):
        """
        Plot the transition matrix as a heatmap.

        Returns:
            matplotlib.figure.Figure: The figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(self.transition_matrix, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.states, yticklabels=self.states, ax=ax)
        ax.set_title('Patient Flow Transition Probabilities')
        ax.set_xlabel('To Department')
        ax.set_ylabel('From Department')
        return fig

# Function to maintain backward compatibility
def simulate_patient_flow(start_state, steps):
    """
    Simulate patient flow (backward compatibility function).

    Args:
        start_state (str): Starting department
        steps (int): Maximum number of steps to simulate

    Returns:
        list: Sequence of departments
    """
    model = HospitalStochasticModel()
    return model.simulate_patient_flow(start_state, steps)
