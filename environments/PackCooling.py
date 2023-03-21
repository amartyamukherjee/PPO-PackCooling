import numpy as np
import gym
from gym import spaces

from environments.PackCoolingGraph import PackCoolingGraph

LOOKBACK_WINDOW_SIZE = 0

class PackCooling(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['live', 'file', 'none']}
    visualization = None

    def __init__(self, linear=False, Nx=101, dt=0.01, N_iter=10):
        super(PackCooling, self).__init__()

        self.x = np.linspace(0.0,1.0,Nx)
        self.dx = 1 / (Nx-1)
        # Must satisfy dx >= sigma*dt = dt for stability of interpolation
        self.dt = self.dx if dt > self.dx else dt

        # Define the parameters of the model
        self.D = 0.2 # Thermal diffusion coefficient within the battery pack
        self.R = 10.0 # Thermal resistance between the battery pack and the cooling fluid
        self.N_iter = N_iter # Number of times to run the numerical solution in step()

        if linear:
            # h(u) = u
            self.F = np.eye(Nx)
            self.h = self.h_linear
        else:
            # h(u) = exp(0.1*u)
            self.h = self.h_nonlinear

        # Initial condition to set for U. Set to cosine series to respect boundary conditions.
        self.num_fourier_coeffs = 5

        self.initial_condition_cosine = np.zeros((self.num_fourier_coeffs,Nx))
        for i in range(self.num_fourier_coeffs):
            self.initial_condition_cosine[i,:] = np.cos(i*self.x)

        # Continuous actions between 0 and 1 to control the transport speed of the cooling fluid
        self.action_space = spaces.Box(0.0, 1.0, dtype=np.float32)

        # Observations aranged as [U, W]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=
                        (2*Nx, 1), dtype=np.float32)
        
    def h_nonlinear(self, u):
        return np.exp(0.1*u)
    
    def h_linear(self,u):
        return self.F @ u

    def step(self, action):
        # Execute one time step within the environment
        ...
    def reset(self):
        # Reset the state of the environment to an initial state

        # Set u to a random cosine series
        self.u = np.dot(np.random.uniform(0,1,(self.num_fourier_coeffs)),self.initial_condition_cosine)

        # Set w to a list of zeros
        self.w = np.zeros_like(self.x)

        self.u_render = np.zeros((LOOKBACK_WINDOW_SIZE,self.x.shape[0]))
        self.w_render = np.zeros((LOOKBACK_WINDOW_SIZE,self.x.shape[0]))
        self.sigma_render = np.zeros((LOOKBACK_WINDOW_SIZE))

        self.u_render[0,:] = self.u
        self.w_render[0,:] = self.w

        self.t = 0
        
    def render(self, mode='live', **kwargs):
        # Render the environment to the screen
        if mode == 'live':
            if self.visualization == None:
                self.visualization = PackCoolingGraph(
                    kwargs.get('title', None))

            if self.t >= LOOKBACK_WINDOW_SIZE:
                self.visualization.render(
                    self.u_render, self.w_render, self.sigma_render)
