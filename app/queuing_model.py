import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from collections import deque

class HospitalQueueingModel:
    """
    A queuing theory model for hospital departments.

    This model simulates patient flow through hospital departments using queuing theory,
    with different service time distributions for each department.
    """

    def __init__(self):
        """
        Initialize the hospital queuing model.
        """
        # Department configurations
        self.departments = {
            'ER': {
                'servers': 10,  # Number of doctors/beds
                'service_time_dist': 'exponential',  # Type of distribution
                'service_time_params': {'scale': 4.0},  # Mean of 4 hours
                'queue_discipline': 'priority'  # Priority queue based on severity
            },
            'Ward': {
                'servers': 50,  # Number of beds
                'service_time_dist': 'normal',
                'service_time_params': {'loc': 72.0, 'scale': 24.0},  # Mean of 3 days, std of 1 day
                'queue_discipline': 'fifo'  # First-in, first-out
            },
            'ICU': {
                'servers': 15,  # Number of ICU beds
                'service_time_dist': 'normal',
                'service_time_params': {'loc': 120.0, 'scale': 48.0},  # Mean of 5 days, std of 2 days
                'queue_discipline': 'priority'  # Priority queue based on severity
            }
        }

        # Transition probabilities between departments
        self.transition_probs = {
            'ER': {'Ward': 0.6, 'ICU': 0.2, 'Discharged': 0.2},
            'Ward': {'ICU': 0.1, 'Discharged': 0.9},
            'ICU': {'Ward': 0.3, 'Discharged': 0.7}
        }

        # Priority distributions (1-5, with 5 being highest priority)
        self.priority_dists = {
            'ER': [0.1, 0.2, 0.4, 0.2, 0.1],  # Distribution of priorities for ER
            'ICU': [0.0, 0.1, 0.2, 0.3, 0.4]  # Distribution of priorities for ICU
        }

        # Initialize queues and servers
        self.reset_simulation()

    def reset_simulation(self):
        """
        Reset the simulation state.
        """
        # Initialize queues for each department
        self.queues = {dept: deque() for dept in self.departments}

        # Initialize servers (currently occupied beds/doctors)
        self.servers = {dept: [] for dept in self.departments}

        # Initialize statistics
        self.stats = {
            'waiting_times': {dept: [] for dept in self.departments},
            'service_times': {dept: [] for dept in self.departments},
            'queue_lengths': {dept: [] for dept in self.departments},
            'utilization': {dept: [] for dept in self.departments}
        }

        # Initialize clock
        self.clock = 0

    def generate_service_time(self, department):
        """
        Generate a service time for a patient in the given department.

        Args:
            department (str): Department name

        Returns:
            float: Service time in hours
        """
        dept_config = self.departments[department]
        dist_type = dept_config['service_time_dist']
        params = dept_config['service_time_params']

        if dist_type == 'exponential':
            return max(0.5, np.random.exponential(**params))
        elif dist_type == 'normal':
            return max(0.5, np.random.normal(**params))
        elif dist_type == 'lognormal':
            return max(0.5, np.random.lognormal(**params))
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    def generate_priority(self, department):
        """
        Generate a priority level for a patient in the given department.

        Args:
            department (str): Department name

        Returns:
            int: Priority level (1-5, with 5 being highest priority)
        """
        if department in self.priority_dists:
            return np.random.choice(range(1, 6), p=self.priority_dists[department])
        else:
            return 3  # Default medium priority

    def add_patient(self, department, priority=None):
        """
        Add a patient to the specified department.

        Args:
            department (str): Department name
            priority (int, optional): Priority level (1-5)

        Returns:
            dict: Patient information
        """
        if department not in self.departments:
            raise ValueError(f"Unknown department: {department}")

        # Generate priority if not provided
        if priority is None:
            priority = self.generate_priority(department)

        # Generate service time
        service_time = self.generate_service_time(department)

        # Create patient record
        patient = {
            'id': np.random.randint(10000, 99999),
            'department': department,
            'priority': priority,
            'service_time': service_time,
            'arrival_time': self.clock,
            'start_service_time': None,
            'departure_time': None
        }

        # Check if a server is available
        if len(self.servers[department]) < self.departments[department]['servers']:
            # Server available, start service immediately
            patient['start_service_time'] = self.clock
            patient['departure_time'] = self.clock + service_time
            self.servers[department].append(patient)
            self.stats['waiting_times'][department].append(0)
        else:
            # No server available, add to queue
            self.queues[department].append(patient)

            # Sort queue by priority if using priority discipline
            if self.departments[department]['queue_discipline'] == 'priority':
                # Convert to list, sort, and convert back to deque
                queue_list = list(self.queues[department])
                queue_list.sort(key=lambda x: (-x['priority'], x['arrival_time']))
                self.queues[department] = deque(queue_list)

        # Update statistics
        self.stats['queue_lengths'][department].append(len(self.queues[department]))
        self.stats['utilization'][department].append(
            len(self.servers[department]) / self.departments[department]['servers']
        )

        return patient

    def process_departure(self, department, patient_index):
        """
        Process a patient departure from a department.

        Args:
            department (str): Department name
            patient_index (int): Index of the patient in the servers list

        Returns:
            dict: Departed patient information
        """
        # Check if the department and patient index are valid
        if department not in self.servers:
            print(f"Warning: Department {department} not found in servers")
            return None

        if patient_index >= len(self.servers[department]):
            print(f"Warning: Patient index {patient_index} out of range for department {department}")
            return None

        # Remove patient from servers
        patient = self.servers[department].pop(patient_index)

        # Record service time
        service_time = patient['departure_time'] - patient['start_service_time']
        self.stats['service_times'][department].append(service_time)

        # Check if there are patients waiting in the queue
        if self.queues[department]:
            # Get next patient from queue
            next_patient = self.queues[department].popleft()

            # Start service for next patient
            next_patient['start_service_time'] = self.clock
            next_patient['departure_time'] = self.clock + next_patient['service_time']
            self.servers[department].append(next_patient)

            # Record waiting time
            waiting_time = self.clock - next_patient['arrival_time']
            self.stats['waiting_times'][department].append(waiting_time)

        # Determine next department or discharge
        next_step = np.random.choice(
            list(self.transition_probs[department].keys()),
            p=list(self.transition_probs[department].values())
        )

        # If not discharged, add to next department
        if next_step != 'Discharged':
            # Transfer priority or generate new one
            priority = patient.get('priority', None)
            self.add_patient(next_step, priority)

        # Update statistics
        self.stats['queue_lengths'][department].append(len(self.queues[department]))
        self.stats['utilization'][department].append(
            len(self.servers[department]) / self.departments[department]['servers']
        )

        return patient

    def simulate(self, duration, arrival_rate=3.0):
        """
        Run a simulation for the specified duration.

        Args:
            duration (float): Duration to simulate in hours
            arrival_rate (float): Average number of arrivals per hour

        Returns:
            dict: Simulation statistics
        """
        try:
            # Reset simulation
            self.reset_simulation()

            # Initialize event list with initial arrivals
            # Generate inter-arrival times using exponential distribution
            arrival_times = np.cumsum(np.random.exponential(1/arrival_rate, size=int(arrival_rate * duration * 2)))
            arrival_times = arrival_times[arrival_times <= duration]

            # Department distribution for arrivals
            dept_probs = {'ER': 0.7, 'Ward': 0.2, 'ICU': 0.1}
            departments = np.random.choice(
                list(dept_probs.keys()),
                size=len(arrival_times),
                p=list(dept_probs.values())
            )

            # Create events for arrivals
            events = [{'time': time, 'type': 'arrival', 'department': dept}
                     for time, dept in zip(arrival_times, departments)]

            # Run simulation
            while events:
                # Get next event
                events.sort(key=lambda x: x['time'])
                event = events.pop(0)

                # Update clock
                self.clock = event['time']

                if event['type'] == 'arrival':
                    # Process arrival
                    patient = self.add_patient(event['department'])

                    # If service started immediately, schedule departure
                    if patient['departure_time'] is not None:
                        events.append({
                            'time': patient['departure_time'],
                            'type': 'departure',
                            'department': patient['department'],
                            'patient_index': len(self.servers[patient['department']]) - 1
                        })

                elif event['type'] == 'departure':
                    # Process departure - safely handle potential errors
                    try:
                        departed_patient = self.process_departure(event['department'], event['patient_index'])

                        # If departure processing failed, skip this event
                        if departed_patient is None:
                            continue

                    except Exception as e:
                        print(f"Error processing departure: {e}")
                        continue

            # Calculate summary statistics
            summary = {}
            for dept in self.departments:
                dept_stats = {}

                # Average waiting time
                if self.stats['waiting_times'][dept]:
                    dept_stats['avg_waiting_time'] = np.mean(self.stats['waiting_times'][dept])
                else:
                    dept_stats['avg_waiting_time'] = 0

                # Average service time
                if self.stats['service_times'][dept]:
                    dept_stats['avg_service_time'] = np.mean(self.stats['service_times'][dept])
                else:
                    dept_stats['avg_service_time'] = 0

                # Average queue length
                if self.stats['queue_lengths'][dept]:
                    dept_stats['avg_queue_length'] = np.mean(self.stats['queue_lengths'][dept])
                else:
                    dept_stats['avg_queue_length'] = 0

                # Average utilization
                if self.stats['utilization'][dept]:
                    dept_stats['avg_utilization'] = np.mean(self.stats['utilization'][dept])
                else:
                    dept_stats['avg_utilization'] = 0

                summary[dept] = dept_stats

            return summary

        except Exception as e:
            print(f"Error in simulation: {e}")
            # Return empty statistics if simulation fails
            return {dept: {
                'avg_waiting_time': 0,
                'avg_service_time': 0,
                'avg_queue_length': 0,
                'avg_utilization': 0
            } for dept in self.departments}

    def plot_service_time_distributions(self):
        """
        Plot the service time distributions for each department.

        Returns:
            matplotlib.figure.Figure: The figure object
        """
        fig, axes = plt.subplots(1, len(self.departments), figsize=(15, 5))

        for i, (dept, config) in enumerate(self.departments.items()):
            ax = axes[i]

            # Generate sample service times
            dist_type = config['service_time_dist']
            params = config['service_time_params']

            if dist_type == 'exponential':
                x = np.linspace(0, stats.expon.ppf(0.99, **params), 1000)
                y = stats.expon.pdf(x, **params)
                samples = np.random.exponential(**params, size=1000)
            elif dist_type == 'normal':
                x = np.linspace(stats.norm.ppf(0.01, **params), stats.norm.ppf(0.99, **params), 1000)
                y = stats.norm.pdf(x, **params)
                samples = np.random.normal(**params, size=1000)
            elif dist_type == 'lognormal':
                x = np.linspace(stats.lognorm.ppf(0.01, **params), stats.lognorm.ppf(0.99, **params), 1000)
                y = stats.lognorm.pdf(x, **params)
                samples = np.random.lognormal(**params, size=1000)

            # Plot PDF
            ax.plot(x, y, 'r-', lw=2, label='PDF')

            # Plot histogram of samples
            ax.hist(samples, bins=30, density=True, alpha=0.6, color='blue')

            # Set labels and title
            ax.set_xlabel('Service Time (hours)')
            ax.set_ylabel('Density')
            ax.set_title(f'{dept} Service Time Distribution')

            # Add mean line
            mean = np.mean(samples)
            ax.axvline(mean, color='green', linestyle='--', label=f'Mean: {mean:.1f}h')

            ax.legend()

        plt.tight_layout()
        return fig

    @classmethod
    def create_default_model(cls):
        """
        Create a default hospital queuing model.

        Returns:
            HospitalQueueingModel: Configured model
        """
        return cls()
