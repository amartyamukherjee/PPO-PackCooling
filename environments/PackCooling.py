import numpy as np
import gym
from gym import spaces
from matplotlib import pyplot as plt

class PackCooling(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, Nx=101, dt=0.01):
        super(PackCooling, self).__init__()

        self.x = np.linspace(0.0,1.0,Nx)
        self.dx = 1 / (Nx-1)
        # Must satisfy dx >= sigma*dt = dt for stability of interpolation
        self.dt = self.dx if dt > self.dx else dt

        # Define the parameters of the model
        self.D = 0.02 # Thermal diffusion coefficient within the battery pack
        self.R = 0.01 # Thermal resistance between the battery pack and the cooling fluid
        self.h = np.exp # Internal heat generation in the battery pack due to charging/discharge

        # Initial condition to set for U. Set to cosine series to respect boundary conditions.
        self.num_fourier_coeffs = 5

        self.initial_condition_cosine = np.zeros((self.num_fourier_coeffs,Nx))
        for i in range(self.num_fourier_coeffs):
            self.initial_condition_cosine[i,:] = np.cos(i*self.x)

        # Continuous actions between 0 and 1 to control the transport speed of the cooling fluid
        self.action_space = spaces.Box(0.0,1.0)

        # Observations aranged as [U, W, x_pos]
        self.observation_space = spaces.Box(low=0, high=255, shape=
                        (3, Nx, 1), dtype=np.uint8)
        
        self.temperature_ax.fig.gca()

    def step(self, action):
        # Execute one time step within the environment
        ...
    def reset(self):
        # Reset the state of the environment to an initial state

        # Set u to a random cosine series
        self.u = np.dot(np.random.uniform(0,1,(self.num_fourier_coeffs)),self.initial_condition_cosine)

        # Set w to a list of zeros
        self.w = 0 * self.x

        self.x_pos = self.x
        
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        self.temperature_ax
