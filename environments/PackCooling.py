import numpy as np
import gym
from gym import spaces
from scipy import sparse

from environments.PackCoolingGraph import PackCoolingGraph

LOOKBACK_WINDOW_SIZE = 2048

class PackCooling(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['live', 'file', 'none']}
    visualization = None

    def __init__(self, linear=False, Nx=101, dt=0.01, N_iter=1, sigma_0=-5.0):
        super(PackCooling, self).__init__()

        self.x = np.linspace(0.0,1.0,Nx)
        self.dx = 1 / (Nx-1)
        self.Nx = Nx
        # Must satisfy dx >= sigma*dt = dt for stability of interpolation
        self.dt = self.dx if dt > self.dx else dt

        # Define the parameters of the model
        self.D = 0.01 # Thermal diffusion coefficient within the battery pack
        self.R = 2.0 # Thermal resistance between the battery pack and the cooling fluid
        self.N_iter = N_iter # Number of times to run the numerical solution in step()
        self.sigma_0 = sigma_0

        self.linear = linear

        if linear:
            # h(u) = u
            self.F = 0.1*np.eye(Nx)
            self.h = self.h_linear
            self.h_prime = self.h_linear_prime
        else:
            # h(u) = exp(0.1*u)
            self.F = None
            self.h = self.h_nonlinear
            self.h_prime = self.h_nonlinear_prime

        # Initial condition to set for U. Set to cosine series to respect boundary conditions.
        self.num_fourier_coeffs = 10

        self.initial_condition_cosine = np.zeros((self.num_fourier_coeffs,Nx))
        for i in range(self.num_fourier_coeffs):
            self.initial_condition_cosine[i,:] = np.cos(2*np.pi*i*self.x)
        
        self.CN_matrix_init()

        # Continuous actions between 0 and 1 to control the transport speed of the cooling fluid
        self.action_space = spaces.Box(0.0, 1.0, dtype=np.float32)

        # Observations aranged as [U, W]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=
                        (2*Nx, 1), dtype=np.float32)
        
    def h_nonlinear(self, state):
        h_u = np.zeros_like(state)
        h_u[:self.Nx] = np.exp(0.1*state[:self.Nx])
        return h_u
    
    def h_nonlinear_prime(self, state):
        h_prime_u = np.zeros((state.shape[0],state.shape[0]))
        h_prime_u[:self.Nx,:self.Nx] = 0.1*np.diag(np.exp(0.1*state[:self.Nx]))
        return h_prime_u
    
    def h_linear(self,state):
        h_u = np.zeros_like(state)
        h_u[:self.Nx] = self.F @ state
        return h_u
    
    def h_linear_prime(self,state):
        if state.ndim == 1:
            h_prime_u = np.zeros((state.shape[0],state.shape[0]))
        else:
            h_prime_u = np.zeros((state.shape[0],state.shape[1]))
        h_prime_u[:self.Nx,:self.Nx] = self.F
        return h_prime_u
    
    def CN_matrix_init(self):
        u_identity_term = (1/self.dt)*sparse.eye(self.Nx).toarray()

        diffusion_term = sparse.diags(
            [1,-2,1],
            [-1,0,1],
            shape=(self.Nx,self.Nx)
        ).toarray()
        diffusion_term[0,1] += 1
        diffusion_term[-1,-2] += 1
        diffusion_term = (self.D/(2*self.dx**2))*diffusion_term

        u_reaction_term = (1/(2*self.R))*sparse.diags(
            [-1,1],
            [0,self.Nx],
            shape=(self.Nx,2*self.Nx)
        ).toarray()
        
        w_identity_term = (1/self.dt)*sparse.eye(self.Nx).toarray()

        w_reaction_term_lhs = (1/(2*self.R))*sparse.diags(
            [1,-1],
            [0,self.Nx],
            shape=(self.Nx,2*self.Nx)
        ).toarray()

        A_lhs = np.zeros((2*self.Nx,2*self.Nx))
        A_rhs = np.zeros((2*self.Nx,2*self.Nx+1))

        A_lhs[:self.Nx,:self.Nx] += u_identity_term
        A_rhs[:self.Nx,:self.Nx] += u_identity_term
        A_lhs[:self.Nx,:self.Nx] -= diffusion_term
        A_rhs[:self.Nx,:self.Nx] += diffusion_term
        A_lhs[:self.Nx,:] -= u_reaction_term
        A_rhs[:self.Nx,:-1] += u_reaction_term

        A_lhs[self.Nx:,self.Nx:] += w_identity_term
        A_lhs[self.Nx:,:] -= w_reaction_term_lhs
        self.A_lhs = A_lhs
        self.A_rhs = A_rhs

    def CN_matrix(self,sigma):
        l = (sigma*self.dt)/self.dx

        w_reaction_term_rhs = sparse.diags(
            [l,1-l,-l,-1+l],
            [-1,0,self.Nx-1,self.Nx],
            shape=(self.Nx,2*self.Nx+1)
        ).toarray()
        w_reaction_term_rhs[0,1] += l
        w_reaction_term_rhs[0,self.Nx-1] = 0
        w_reaction_term_rhs[0,-1] = l
        w_reaction_term_rhs = w_reaction_term_rhs/(2*self.R)

        w_identity_term_rhs = sparse.diags(
            [l,1-l],
            [-1,0],
            shape=(self.Nx,self.Nx+1)
        ).toarray()
        w_identity_term_rhs[0,-1] = l
        w_identity_term_rhs = w_identity_term_rhs/self.dt

        A_lhs = self.A_lhs.copy()
        A_rhs = self.A_rhs.copy()

        A_rhs[self.Nx:,self.Nx:] += w_identity_term_rhs
        A_rhs[self.Nx:,:] += w_reaction_term_rhs

        return A_lhs, A_rhs
    
    def newtonRaphsonMethod(self,state,A_lhs,A_rhs,N):
        rhs = ((A_rhs @ state.reshape([-1,1])).reshape(-1) + 0.5*self.h(state[:-1])).copy()
        state = state[:-1]
        for _ in range(N):
            state_grad = np.linalg.solve((A_lhs - 0.5*self.h_prime(state)),
                                         (A_lhs @ state.reshape([-1,1])).reshape(-1) - 0.5*self.h(state) - rhs)
            if np.isnan(state_grad).any():
                state -= np.linalg.lstsq((A_lhs - 0.5*self.h_prime(state)),
                                        (A_lhs @ state.reshape([-1,1])).reshape(-1) - 0.5*self.h(state) - rhs,
                                        rcond=1)[0]
            else:
                state -= state_grad
        return state

    def step(self, action):
        # Execute N_iter time steps within the environment
        action = (action + 1) / 2
        state = np.concatenate((self.u,self.w))
        for _ in range(self.N_iter):
            state = np.concatenate((state,[self.sigma_0]))

            A_lhs, A_rhs = self.CN_matrix(action)

            if self.linear:
                new_state = np.linalg.lstsq(A_lhs - 0.5*self.h_prime(A_lhs),
                                            (A_rhs + 0.5*self.h_prime(A_rhs)) @ state.reshape([-1,1])).reshape(-1)
                if np.isnan(new_state).any():
                    state = np.linalg.lstsq(A_lhs - 0.5*self.h_prime(A_lhs),
                                            (A_rhs + 0.5*self.h_prime(A_rhs)) @ state.reshape([-1,1]),
                                            rcond=1)[0].reshape(-1)
                else:
                    state = new_state
            else:
                state = self.newtonRaphsonMethod(state,A_lhs,A_rhs,10)
            
        self.u = state[:self.Nx]
        self.w = state[self.Nx:]
        reward = -np.dot(self.u,self.u)

        self.sigma_render[self.timestep] = action

        self.timestep += 1

        self.u_render[self.timestep,:] = self.u
        self.w_render[self.timestep,:] = self.w

        done = (self.timestep == LOOKBACK_WINDOW_SIZE or reward < -1e7)

        return state, reward, done, False, None

    def reset(self):
        # Reset the state of the environment to an initial state

        # Set u to a random cosine series
        self.u = np.dot(np.random.uniform(-2,2,(self.num_fourier_coeffs)),self.initial_condition_cosine)

        # Set w to a list of zeros
        self.w = self.sigma_0 * np.ones(self.Nx)

        self.u_render = np.zeros((LOOKBACK_WINDOW_SIZE+1,self.x.shape[0]))
        self.w_render = np.zeros((LOOKBACK_WINDOW_SIZE+1,self.x.shape[0]))
        self.sigma_render = np.zeros((LOOKBACK_WINDOW_SIZE))

        self.u_render[0,:] = self.u
        self.w_render[0,:] = self.w

        self.timestep = 0

        return np.concatenate((self.u,self.w)), None
    
    def seed(self,seed):
        np.random.seed(seed)
        
    def render(self, mode='live', **kwargs):
        # Render the environment to the screen
        if mode == 'live':
            if self.visualization == None:
                self.visualization = PackCoolingGraph(
                    kwargs.get('title', None))

            self.visualization.render(
                self.u_render[:self.timestep,:],
                self.w_render[:self.timestep,:],
                self.sigma_render[:self.timestep],
                self.dt*self.N_iter)
