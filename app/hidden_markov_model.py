import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from hmmlearn import hmm

class HiddenMarkovPatientModel:
    """
    A Hidden Markov Model for patient state modeling.

    This model uses hidden states to represent the underlying health condition of patients,
    which may not be directly observable but can be inferred from observable measurements.
    """

    def __init__(self, n_hidden_states=3, n_observations=4):
        """
        Initialize the Hidden Markov Model.

        Args:
            n_hidden_states (int): Number of hidden states (e.g., severity levels)
            n_observations (int): Number of possible observations (e.g., symptoms, vital signs)
        """
        self.n_hidden_states = n_hidden_states
        self.n_observations = n_observations
        # For MultinomialHMM, n_trials must be set (set to 1 for single observation per time step)
        self.model = hmm.MultinomialHMM(n_components=n_hidden_states, n_trials=1)

        # State names (for interpretation)
        self.state_names = [f"State_{i}" for i in range(n_hidden_states)]

        # Observation names (for interpretation)
        self.observation_names = [f"Obs_{i}" for i in range(n_observations)]

        # Default parameters (will be overridden by training)
        self.model.startprob_ = np.ones(n_hidden_states) / n_hidden_states
        self.model.transmat_ = np.ones((n_hidden_states, n_hidden_states)) / n_hidden_states
        self.model.emissionprob_ = np.ones((n_hidden_states, n_observations)) / n_observations

    def set_state_names(self, names):
        """
        Set meaningful names for the hidden states.

        Args:
            names (list): List of state names
        """
        if len(names) != self.n_hidden_states:
            raise ValueError(f"Expected {self.n_hidden_states} state names, got {len(names)}")
        self.state_names = names

    def set_observation_names(self, names):
        """
        Set meaningful names for the observations.

        Args:
            names (list): List of observation names
        """
        if len(names) != self.n_observations:
            raise ValueError(f"Expected {self.n_observations} observation names, got {len(names)}")
        self.observation_names = names

    def train(self, observations, lengths=None):
        """
        Train the HMM on observation sequences.

        Args:
            observations (array): Observation sequences
            lengths (list): Lengths of each sequence

        Returns:
            float: Log-likelihood of the model
        """
        # Reshape observations if needed
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)

        # Train the model
        log_likelihood = self.model.fit(observations, lengths)

        return log_likelihood

    def predict_states(self, observations):
        """
        Predict the most likely sequence of hidden states.

        Args:
            observations (array): Observation sequence (indices of observations)

        Returns:
            array: Most likely state sequence
        """
        # For MultinomialHMM, we need to convert observation indices to one-hot encoded vectors
        # where each vector has n_trials=1 for the observed category and 0 for others
        n_samples = len(observations)
        X = np.zeros((n_samples, self.n_observations))

        # Set 1 for the observed category in each sample
        for i in range(n_samples):
            obs_idx = observations[i, 0]  # Get the observation index
            X[i, obs_idx] = 1

        # Predict states
        states = self.model.predict(X)

        return states

    def predict_next_observation(self, current_state):
        """
        Predict the next observation given the current state.

        Args:
            current_state (int): Current hidden state

        Returns:
            int: Most likely next observation
        """
        # Get emission probabilities for the current state
        emission_probs = self.model.emissionprob_[current_state]

        # Get the most likely observation
        next_observation = np.argmax(emission_probs)

        return next_observation

    def simulate(self, n_steps, start_state=None):
        """
        Simulate a sequence of states and observations.

        Args:
            n_steps (int): Number of steps to simulate
            start_state (int): Starting state (if None, sample from initial distribution)

        Returns:
            tuple: (states, observations)
        """
        # Initialize
        states = np.zeros(n_steps, dtype=int)
        observations = np.zeros(n_steps, dtype=int)

        # Sample initial state if not provided
        if start_state is None:
            states[0] = np.random.choice(self.n_hidden_states, p=self.model.startprob_)
        else:
            states[0] = start_state

        # Sample initial observation
        observations[0] = np.random.choice(self.n_observations, p=self.model.emissionprob_[states[0]])

        # Generate sequence
        for t in range(1, n_steps):
            # Sample next state
            states[t] = np.random.choice(self.n_hidden_states, p=self.model.transmat_[states[t-1]])

            # Sample observation
            observations[t] = np.random.choice(self.n_observations, p=self.model.emissionprob_[states[t]])

        return states, observations

    def plot_transition_matrix(self):
        """
        Plot the transition matrix as a heatmap.

        Returns:
            matplotlib.figure.Figure: The figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(self.model.transmat_, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.state_names, yticklabels=self.state_names, ax=ax)
        ax.set_title('Hidden State Transition Probabilities')
        ax.set_xlabel('To State')
        ax.set_ylabel('From State')
        return fig

    def plot_emission_matrix(self):
        """
        Plot the emission matrix as a heatmap.

        Returns:
            matplotlib.figure.Figure: The figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(self.model.emissionprob_, annot=True, fmt='.2f', cmap='Greens',
                   xticklabels=self.observation_names, yticklabels=self.state_names, ax=ax)
        ax.set_title('Emission Probabilities')
        ax.set_xlabel('Observation')
        ax.set_ylabel('Hidden State')
        return fig

    @classmethod
    def create_patient_hmm(cls):
        """
        Create a pre-configured HMM for patient health states.

        Returns:
            HiddenMarkovPatientModel: Configured model
        """
        # Create model with 3 hidden states and 4 observations
        model = cls(n_hidden_states=3, n_observations=4)

        # Set meaningful names
        model.set_state_names(['Stable', 'Deteriorating', 'Critical'])
        model.set_observation_names(['Normal Vitals', 'Mild Symptoms', 'Severe Symptoms', 'Critical Symptoms'])

        # Set initial state probabilities (most patients start stable)
        model.model.startprob_ = np.array([0.7, 0.2, 0.1])

        # Set transition matrix
        # Rows: from state, Columns: to state
        model.model.transmat_ = np.array([
            [0.7, 0.2, 0.1],  # Stable -> mostly stays stable
            [0.3, 0.5, 0.2],  # Deteriorating -> can improve or worsen
            [0.1, 0.3, 0.6]   # Critical -> mostly stays critical
        ])

        # Set emission matrix
        # Rows: hidden state, Columns: observation
        model.model.emissionprob_ = np.array([
            [0.6, 0.3, 0.1, 0.0],  # Stable -> mostly normal vitals
            [0.1, 0.5, 0.3, 0.1],  # Deteriorating -> mostly mild/severe symptoms
            [0.0, 0.1, 0.3, 0.6]   # Critical -> mostly severe/critical symptoms
        ])

        return model


# Example usage
if __name__ == "__main__":
    # Create a patient HMM
    patient_hmm = HiddenMarkovPatientModel.create_patient_hmm()

    # Simulate a patient trajectory
    states, observations = patient_hmm.simulate(n_steps=10, start_state=0)

    # Print the trajectory
    print("Simulated Patient Trajectory:")
    for t in range(len(states)):
        state_name = patient_hmm.state_names[states[t]]
        obs_name = patient_hmm.observation_names[observations[t]]
        print(f"Day {t+1}: Hidden State = {state_name}, Observation = {obs_name}")

    # Plot transition and emission matrices
    patient_hmm.plot_transition_matrix()
    patient_hmm.plot_emission_matrix()
    plt.show()
