import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

class PoissonArrivalModel:
    """
    A Poisson process model for patient arrivals to the hospital.
    
    This model simulates patient arrivals using a Poisson process, which is
    appropriate for modeling random arrivals that occur independently of each other.
    """
    
    def __init__(self, arrival_rate=2.5):
        """
        Initialize the Poisson arrival model.
        
        Args:
            arrival_rate (float): Average number of arrivals per hour
        """
        self.arrival_rate = arrival_rate
        
        # Department distribution (probability of arriving at each department)
        self.department_probs = {
            'ER': 0.7,
            'Ward': 0.2,
            'ICU': 0.1
        }
        
        # Time-of-day modifiers (hourly factors that modify the base arrival rate)
        self.hourly_modifiers = np.ones(24)
        # Morning peak (8-11 AM)
        self.hourly_modifiers[8:12] = 1.5
        # Afternoon lull (2-4 PM)
        self.hourly_modifiers[14:16] = 0.8
        # Evening peak (6-9 PM)
        self.hourly_modifiers[18:22] = 1.3
        # Night lull (11 PM - 6 AM)
        self.hourly_modifiers[23:] = 0.5
        self.hourly_modifiers[:6] = 0.5
        
        # Day-of-week modifiers
        self.daily_modifiers = {
            'Monday': 1.2,
            'Tuesday': 1.1,
            'Wednesday': 1.0,
            'Thursday': 1.0,
            'Friday': 1.1,
            'Saturday': 0.8,
            'Sunday': 0.7
        }
    
    def set_arrival_rate(self, rate):
        """
        Set the average arrival rate.
        
        Args:
            rate (float): Average number of arrivals per hour
        """
        if rate <= 0:
            raise ValueError("Arrival rate must be positive")
        self.arrival_rate = rate
    
    def set_department_probs(self, probs):
        """
        Set the department arrival probabilities.
        
        Args:
            probs (dict): Dictionary mapping department names to probabilities
        """
        if not isinstance(probs, dict):
            raise TypeError("Department probabilities must be a dictionary")
        
        if not all(0 <= p <= 1 for p in probs.values()):
            raise ValueError("All probabilities must be between 0 and 1")
        
        if abs(sum(probs.values()) - 1.0) > 1e-10:
            raise ValueError("Probabilities must sum to 1")
        
        self.department_probs = probs
    
    def simulate_arrivals(self, duration_hours, start_hour=0, day_of_week='Monday'):
        """
        Simulate patient arrivals over a specified duration.
        
        Args:
            duration_hours (float): Duration to simulate in hours
            start_hour (int): Hour of day to start simulation (0-23)
            day_of_week (str): Day of week to start simulation
            
        Returns:
            list: List of dictionaries containing arrival information
        """
        # Validate inputs
        if duration_hours <= 0:
            raise ValueError("Duration must be positive")
        
        if not 0 <= start_hour < 24:
            raise ValueError("Start hour must be between 0 and 23")
        
        if day_of_week not in self.daily_modifiers:
            raise ValueError(f"Day of week must be one of {list(self.daily_modifiers.keys())}")
        
        # Initialize results
        arrivals = []
        current_hour = start_hour
        current_day = day_of_week
        days_of_week = list(self.daily_modifiers.keys())
        day_index = days_of_week.index(current_day)
        
        # Simulate hour by hour
        for hour_offset in range(int(duration_hours) + 1):
            # Calculate current hour and day
            current_hour = (start_hour + hour_offset) % 24
            day_index = (days_of_week.index(day_of_week) + (start_hour + hour_offset) // 24) % 7
            current_day = days_of_week[day_index]
            
            # Get modifiers for current hour and day
            hour_modifier = self.hourly_modifiers[current_hour]
            day_modifier = self.daily_modifiers[current_day]
            
            # Calculate effective arrival rate
            effective_rate = self.arrival_rate * hour_modifier * day_modifier
            
            # Generate number of arrivals for this hour
            num_arrivals = np.random.poisson(effective_rate)
            
            # Generate arrival times within this hour
            if num_arrivals > 0:
                arrival_times = np.random.uniform(0, 1, num_arrivals)
                arrival_times.sort()
                
                # Generate departments
                departments = np.random.choice(
                    list(self.department_probs.keys()),
                    size=num_arrivals,
                    p=list(self.department_probs.values())
                )
                
                # Add arrivals to results
                for i in range(num_arrivals):
                    absolute_time = hour_offset + arrival_times[i]
                    arrivals.append({
                        'time': absolute_time,
                        'hour': current_hour,
                        'day': current_day,
                        'department': departments[i]
                    })
        
        return arrivals
    
    def plot_arrival_pattern(self, duration_hours=24, start_hour=0, day_of_week='Monday'):
        """
        Plot the expected arrival pattern over time.
        
        Args:
            duration_hours (float): Duration to plot in hours
            start_hour (int): Hour of day to start plot (0-23)
            day_of_week (str): Day of week to start plot
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Initialize data
        hours = []
        rates = []
        days = []
        
        # Calculate rates for each hour
        current_hour = start_hour
        current_day = day_of_week
        days_of_week = list(self.daily_modifiers.keys())
        day_index = days_of_week.index(current_day)
        
        for hour_offset in range(int(duration_hours) + 1):
            # Calculate current hour and day
            current_hour = (start_hour + hour_offset) % 24
            day_index = (days_of_week.index(day_of_week) + (start_hour + hour_offset) // 24) % 7
            current_day = days_of_week[day_index]
            
            # Get modifiers for current hour and day
            hour_modifier = self.hourly_modifiers[current_hour]
            day_modifier = self.daily_modifiers[current_day]
            
            # Calculate effective arrival rate
            effective_rate = self.arrival_rate * hour_modifier * day_modifier
            
            hours.append(hour_offset)
            rates.append(effective_rate)
            days.append(current_day)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot rates
        ax.plot(hours, rates, marker='o', linestyle='-')
        
        # Add day boundaries
        day_changes = [i for i in range(1, len(days)) if days[i] != days[i-1]]
        for day_change in day_changes:
            ax.axvline(x=day_change, color='gray', linestyle='--', alpha=0.7)
            ax.text(day_change, max(rates) * 1.05, days[day_change], 
                   ha='center', va='bottom', rotation=90, alpha=0.7)
        
        # Add first day label
        ax.text(0, max(rates) * 1.05, days[0], 
               ha='center', va='bottom', rotation=90, alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('Hours from Start')
        ax.set_ylabel('Expected Arrivals per Hour')
        ax.set_title('Expected Patient Arrival Pattern')
        
        # Set x-axis ticks
        ax.set_xticks(range(0, int(duration_hours) + 1, 6))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_department_distribution(self):
        """
        Plot the distribution of arrivals by department.
        
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot department distribution
        departments = list(self.department_probs.keys())
        probs = list(self.department_probs.values())
        
        ax.bar(departments, probs, color=sns.color_palette("Blues_d", len(departments)))
        
        # Set labels and title
        ax.set_xlabel('Department')
        ax.set_ylabel('Probability')
        ax.set_title('Patient Arrival Distribution by Department')
        
        # Add percentage labels
        for i, p in enumerate(probs):
            ax.text(i, p + 0.02, f"{p:.1%}", ha='center')
        
        # Set y-axis limit
        ax.set_ylim(0, max(probs) * 1.2)
        
        return fig
    
    @classmethod
    def create_default_model(cls):
        """
        Create a default Poisson arrival model with pre-configured parameters.
        
        Returns:
            PoissonArrivalModel: Configured model
        """
        model = cls(arrival_rate=3.0)
        
        # Set department probabilities
        model.set_department_probs({
            'ER': 0.65,
            'Ward': 0.25,
            'ICU': 0.10
        })
        
        return model
