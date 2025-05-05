import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import networkx as nx
import math

class HospitalVisualization3D:
    """
    A class for creating 3D visualizations of hospital data and models.
    """
    
    def __init__(self):
        """Initialize the visualization class with default settings."""
        # Color schemes
        self.color_scheme = {
            'ER': '#FF5733',  # Red-orange
            'Ward': '#33A1FF',  # Blue
            'ICU': '#FF33A1',  # Pink
            'Discharged': '#33FF57',  # Green
            'background': '#111111',  # Dark background
            'grid': '#333333',  # Dark grid
            'text': '#FFFFFF'  # White text
        }
        
        # Department coordinates in 3D space
        self.department_coords = {
            'ER': {'x': 0, 'y': 0, 'z': 0},
            'Ward': {'x': 10, 'y': 0, 'z': 0},
            'ICU': {'x': 5, 'y': 8.66, 'z': 0},  # Positioned to form a triangle
            'Discharged': {'x': 5, 'y': 4, 'z': 7}  # Above the center
        }
        
        # Default layout settings
        self.layout_settings = {
            'width': 800,
            'height': 600,
            'margin': {'l': 0, 'r': 0, 'b': 0, 't': 30},
            'paper_bgcolor': self.color_scheme['background'],
            'plot_bgcolor': self.color_scheme['background'],
            'scene': {
                'xaxis': {
                    'showbackground': True,
                    'backgroundcolor': self.color_scheme['background'],
                    'gridcolor': self.color_scheme['grid'],
                    'showticklabels': False
                },
                'yaxis': {
                    'showbackground': True,
                    'backgroundcolor': self.color_scheme['background'],
                    'gridcolor': self.color_scheme['grid'],
                    'showticklabels': False
                },
                'zaxis': {
                    'showbackground': True,
                    'backgroundcolor': self.color_scheme['background'],
                    'gridcolor': self.color_scheme['grid'],
                    'showticklabels': False
                }
            },
            'font': {'color': self.color_scheme['text']}
        }
    
    def create_3d_patient_flow(self, transition_matrix, department_names=None):
        """
        Create a 3D visualization of patient flow between departments.
        
        Args:
            transition_matrix (np.ndarray): Matrix of transition probabilities
            department_names (list): List of department names
            
        Returns:
            plotly.graph_objects.Figure: 3D visualization
        """
        if department_names is None:
            department_names = list(self.department_coords.keys())
        
        # Create a graph
        G = nx.DiGraph()
        
        # Add nodes (departments)
        for dept in department_names:
            G.add_node(dept, pos=(
                self.department_coords[dept]['x'],
                self.department_coords[dept]['y'],
                self.department_coords[dept]['z']
            ))
        
        # Add edges (transitions)
        for i, source in enumerate(department_names):
            for j, target in enumerate(department_names):
                if i != j and transition_matrix[i, j] > 0:
                    G.add_edge(source, target, weight=transition_matrix[i, j])
        
        # Get node positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Create figure
        fig = go.Figure()
        
        # Add nodes (departments)
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_z = [pos[node][2] for node in G.nodes()]
        
        # Add department nodes
        fig.add_trace(go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode='markers+text',
            marker={
                'size': 15,
                'color': [self.color_scheme[dept] for dept in G.nodes()],
                'opacity': 0.8
            },
            text=list(G.nodes()),
            textposition="top center",
            hoverinfo='text',
            name='Departments'
        ))
        
        # Add edges (transitions)
        edge_x = []
        edge_y = []
        edge_z = []
        edge_colors = []
        edge_widths = []
        
        for edge in G.edges(data=True):
            source, target, data = edge
            x0, y0, z0 = pos[source]
            x1, y1, z1 = pos[target]
            
            # Create a curved path between nodes
            pts = self._create_curve(x0, y0, z0, x1, y1, z1)
            
            edge_x.extend(pts[0])
            edge_y.extend(pts[1])
            edge_z.extend(pts[2])
            
            # Add None to create a break in the line
            edge_x.append(None)
            edge_y.append(None)
            edge_z.append(None)
            
            # Edge color based on source department
            edge_colors.append(self.color_scheme[source])
            
            # Edge width based on transition probability
            edge_widths.append(data['weight'] * 10)
        
        # Add transition edges
        fig.add_trace(go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line={
                'color': edge_colors,
                'width': 5
            },
            opacity=0.7,
            hoverinfo='none',
            name='Transitions'
        ))
        
        # Add animated particles to show flow
        self._add_animated_particles(fig, G, pos)
        
        # Update layout
        fig.update_layout(**self.layout_settings)
        fig.update_layout(
            title={
                'text': '3D Patient Flow Visualization',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            scene_camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )
        
        return fig
    
    def create_3d_queuing_visualization(self, queuing_data):
        """
        Create a 3D visualization of hospital queues.
        
        Args:
            queuing_data (dict): Dictionary with queuing statistics by department
            
        Returns:
            plotly.graph_objects.Figure: 3D visualization
        """
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{'type': 'scatter3d'}]]
        )
        
        # Department coordinates
        x_coords = [self.department_coords[dept]['x'] for dept in queuing_data.keys()]
        y_coords = [self.department_coords[dept]['y'] for dept in queuing_data.keys()]
        z_coords = [self.department_coords[dept]['z'] for dept in queuing_data.keys()]
        
        # Department names
        dept_names = list(queuing_data.keys())
        
        # Queue lengths
        queue_lengths = [data['avg_queue_length'] for data in queuing_data.values()]
        
        # Waiting times
        waiting_times = [data['avg_waiting_time'] for data in queuing_data.values()]
        
        # Utilization
        utilization = [data['avg_utilization'] for data in queuing_data.values()]
        
        # Add department nodes
        fig.add_trace(
            go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='markers+text',
                marker={
                    'size': 15,
                    'color': [self.color_scheme[dept] for dept in dept_names],
                    'opacity': 0.8
                },
                text=dept_names,
                textposition="top center",
                hoverinfo='text',
                name='Departments'
            )
        )
        
        # Add queue length cylinders
        for i, dept in enumerate(dept_names):
            x, y, z = x_coords[i], y_coords[i], z_coords[i]
            queue_length = queue_lengths[i]
            
            # Create cylinder for queue length
            if queue_length > 0:
                cylinder_pts = self._create_cylinder(x, y, z, queue_length)
                
                fig.add_trace(
                    go.Mesh3d(
                        x=cylinder_pts[0],
                        y=cylinder_pts[1],
                        z=cylinder_pts[2],
                        color=self.color_scheme[dept],
                        opacity=0.7,
                        name=f'{dept} Queue'
                    )
                )
        
        # Add utilization bars
        for i, dept in enumerate(dept_names):
            x, y, z = x_coords[i], y_coords[i], z_coords[i]
            util = utilization[i]
            
            # Create vertical bar for utilization
            if util > 0:
                bar_pts = self._create_vertical_bar(x + 2, y, z, util * 5)
                
                fig.add_trace(
                    go.Mesh3d(
                        x=bar_pts[0],
                        y=bar_pts[1],
                        z=bar_pts[2],
                        color=self.color_scheme[dept],
                        opacity=0.5,
                        name=f'{dept} Utilization'
                    )
                )
        
        # Update layout
        fig.update_layout(**self.layout_settings)
        fig.update_layout(
            title={
                'text': '3D Hospital Queuing Visualization',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            scene_camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )
        
        return fig
    
    def create_3d_hmm_visualization(self, hmm_model):
        """
        Create a 3D visualization of a Hidden Markov Model.
        
        Args:
            hmm_model: The Hidden Markov Model object
            
        Returns:
            plotly.graph_objects.Figure: 3D visualization
        """
        # Get model parameters
        n_states = hmm_model.n_hidden_states
        n_observations = hmm_model.n_observations
        
        # Create figure
        fig = go.Figure()
        
        # Calculate coordinates for states in a circle
        state_coords = []
        radius = 5
        for i in range(n_states):
            angle = 2 * math.pi * i / n_states
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = 0
            state_coords.append((x, y, z))
        
        # Calculate coordinates for observations in a circle above the states
        obs_coords = []
        obs_radius = 7
        for i in range(n_observations):
            angle = 2 * math.pi * i / n_observations
            x = obs_radius * math.cos(angle)
            y = obs_radius * math.sin(angle)
            z = 5
            obs_coords.append((x, y, z))
        
        # Add state nodes
        state_x = [coord[0] for coord in state_coords]
        state_y = [coord[1] for coord in state_coords]
        state_z = [coord[2] for coord in state_coords]
        
        fig.add_trace(go.Scatter3d(
            x=state_x,
            y=state_y,
            z=state_z,
            mode='markers+text',
            marker={
                'size': 10,
                'color': 'red',
                'opacity': 0.8
            },
            text=[f"State {i}" for i in range(n_states)],
            textposition="top center",
            name='Hidden States'
        ))
        
        # Add observation nodes
        obs_x = [coord[0] for coord in obs_coords]
        obs_y = [coord[1] for coord in obs_coords]
        obs_z = [coord[2] for coord in obs_coords]
        
        fig.add_trace(go.Scatter3d(
            x=obs_x,
            y=obs_y,
            z=obs_z,
            mode='markers+text',
            marker={
                'size': 10,
                'color': 'blue',
                'opacity': 0.8
            },
            text=[f"Obs {i}" for i in range(n_observations)],
            textposition="top center",
            name='Observations'
        ))
        
        # Add transition edges between states
        edge_x = []
        edge_y = []
        edge_z = []
        
        for i in range(n_states):
            for j in range(n_states):
                if hmm_model.model.transmat_[i, j] > 0.01:  # Only show significant transitions
                    x0, y0, z0 = state_coords[i]
                    x1, y1, z1 = state_coords[j]
                    
                    # Create curved path
                    pts = self._create_curve(x0, y0, z0, x1, y1, z1)
                    
                    edge_x.extend(pts[0])
                    edge_y.extend(pts[1])
                    edge_z.extend(pts[2])
                    
                    # Add None to create a break
                    edge_x.append(None)
                    edge_y.append(None)
                    edge_z.append(None)
        
        fig.add_trace(go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line={'color': 'rgba(255, 0, 0, 0.5)', 'width': 2},
            name='State Transitions'
        ))
        
        # Add emission edges from states to observations
        emission_x = []
        emission_y = []
        emission_z = []
        
        for i in range(n_states):
            for j in range(n_observations):
                if hmm_model.model.emissionprob_[i, j] > 0.1:  # Only show significant emissions
                    x0, y0, z0 = state_coords[i]
                    x1, y1, z1 = obs_coords[j]
                    
                    # Create straight line
                    emission_x.extend([x0, x1, None])
                    emission_y.extend([y0, y1, None])
                    emission_z.extend([z0, z1, None])
        
        fig.add_trace(go.Scatter3d(
            x=emission_x,
            y=emission_y,
            z=emission_z,
            mode='lines',
            line={'color': 'rgba(0, 0, 255, 0.3)', 'width': 1},
            name='Emission Probabilities'
        ))
        
        # Update layout
        fig.update_layout(**self.layout_settings)
        fig.update_layout(
            title={
                'text': '3D Hidden Markov Model Visualization',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            scene_camera=dict(
                eye=dict(x=0, y=0, z=2)
            )
        )
        
        return fig
    
    def _create_curve(self, x0, y0, z0, x1, y1, z1, pts=20, height_factor=0.5):
        """Create a curved path between two points in 3D space."""
        # Midpoint with height offset
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        mid_z = max(z0, z1) + height_factor * math.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
        
        # Create points along a quadratic Bezier curve
        t = np.linspace(0, 1, pts)
        x = (1-t)**2 * x0 + 2*(1-t)*t * mid_x + t**2 * x1
        y = (1-t)**2 * y0 + 2*(1-t)*t * mid_y + t**2 * y1
        z = (1-t)**2 * z0 + 2*(1-t)*t * mid_z + t**2 * z1
        
        return x, y, z
    
    def _create_cylinder(self, x, y, z, height, radius=0.5, pts=20):
        """Create a cylinder at the specified position with given height."""
        # Create points around the circle
        theta = np.linspace(0, 2*np.pi, pts)
        
        # Bottom circle
        x_bottom = x + radius * np.cos(theta)
        y_bottom = y + radius * np.sin(theta)
        z_bottom = np.full_like(theta, z)
        
        # Top circle
        x_top = x + radius * np.cos(theta)
        y_top = y + radius * np.sin(theta)
        z_top = np.full_like(theta, z + height)
        
        # Combine points
        x_pts = np.concatenate([x_bottom, x_top])
        y_pts = np.concatenate([y_bottom, y_top])
        z_pts = np.concatenate([z_bottom, z_top])
        
        return x_pts, y_pts, z_pts
    
    def _create_vertical_bar(self, x, y, z, height, width=0.5, depth=0.5):
        """Create a vertical bar at the specified position with given height."""
        # Create the 8 corners of the bar
        x_pts = [x-width/2, x+width/2, x+width/2, x-width/2, x-width/2, x+width/2, x+width/2, x-width/2]
        y_pts = [y-depth/2, y-depth/2, y+depth/2, y+depth/2, y-depth/2, y-depth/2, y+depth/2, y+depth/2]
        z_pts = [z, z, z, z, z+height, z+height, z+height, z+height]
        
        return x_pts, y_pts, z_pts
    
    def _add_animated_particles(self, fig, G, pos, num_particles=50):
        """Add animated particles to show flow along edges."""
        # This is a placeholder for animation
        # In a real implementation, this would use Plotly's animation capabilities
        # For now, we'll just add static particles along the paths
        
        particle_x = []
        particle_y = []
        particle_z = []
        particle_colors = []
        
        for edge in G.edges(data=True):
            source, target, data = edge
            x0, y0, z0 = pos[source]
            x1, y1, z1 = pos[target]
            
            # Create points along the path
            pts = self._create_curve(x0, y0, z0, x1, y1, z1)
            
            # Sample a few points for particles
            indices = np.linspace(0, len(pts[0])-1, 5).astype(int)
            
            particle_x.extend([pts[0][i] for i in indices])
            particle_y.extend([pts[1][i] for i in indices])
            particle_z.extend([pts[2][i] for i in indices])
            
            # Particle color based on source
            particle_colors.extend([self.color_scheme[source]] * len(indices))
        
        # Add particles
        fig.add_trace(go.Scatter3d(
            x=particle_x,
            y=particle_y,
            z=particle_z,
            mode='markers',
            marker={
                'size': 3,
                'color': particle_colors,
                'opacity': 0.8
            },
            name='Flow'
        ))
        
        return fig
