import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np


class QuadHover(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, delay=3, comp_delay_prob=0.0, noise=0.1, noise_p=0.1, thrust_tc=0.02, settle=1.0, dt=0.02, seed=0):
        # Constants
        self.G = 9.81
        self.MAX_H = 15.0  # treat as inclusive bounds
        self.MIN_H = 0.05
        self.MAX_T = 30.0  # time limit

        # Keywords
        # TODO: implement computational delay
        self.dt = dt
        self.settle = settle  # initial settling period without any control
        self.delay = delay  # delay in steps
        self.comp_delay_prob = comp_delay_prob
        self.noise_std = noise  # white noise
        self.noise_p_std = noise_p  # noise proportional to divergence
        self.thrust_tc = thrust_tc  # thrust time constant

        # Figure for rendering

        # Reset to get initial observation
        self.reset()

        # Initialize spaces
        # Thrust value as action, (div, div_dot) as observation
        self.action_space = spaces.Box(low=-0.8 * self.G, high=0.5 * self.G, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # Seed
        self.seed(seed)

    def step(self, action, wind=0.0):
        # TODO: put wind in a pre-initialized process instead of external?
        # Take action and update state
        # Forward Euler method
        self.state += (self._get_state_dot(action) + [0.0, wind, 0.0]) * self.dt
        self.t += self.dt

        # Check whether done
        done = self._check_done()

        # Get reward
        reward = self._get_reward()

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # Compute ground truth divergence
        # TODO: add factor of 2 after comparison
        div = -self.state[1] / (self.state[0] if abs(self.state[0]) > 1e-5 else 1e-5)
        div_dot = (div - self.div_ph) / self.dt

        # Overwrite placeholder
        self.div_ph = div

        # Add noise (regular and proportional)
        # Use old noisy estimate for noisy div_dot
        div += np.random.normal(0.0, self.noise_std) + div * np.random.normal(0.0, self.noise_p_std)
        div_dot = (div - self.obs[-1][0]) / self.dt

        # Append to end of deque; if == max length then first is popped
        self.obs.append([div, div_dot])

        return np.array(self.obs[0])


    def _clamp(self, value, minimum, maximum):
        return max(min(value, maximum), minimum)


    def _check_done(self):
        out_of_bounds = self._check_out_of_bounds()
        out_of_time = self._check_out_of_time()
        return out_of_bounds or out_of_time


    def _check_out_of_bounds(self):
        return self.state[0] < self.MIN_H or self.state[0] > self.MAX_H:


    def _check_out_of_time(self):
        return self.t >= self.MAX_T:


    def _get_reward(self):
        if self._check_done():
            return 1.0 / (self.t * self.state[1] * self.state[1] + 0.01) - self._clamp(self.state[0], self.MIN_H, self.MAX_H)
        else:
            return -1.0


    def _get_state_dot(self, action):
        # Action is delta thrust relative to hover thrust in m/s^2
        # So: state_dot for the first two states (height, velocity)
        # is just the last two states (velocity, thrust in m/s^2)!
        # First do nothing for some time, to allow settling of controller
        # and filling of deque
        if self.t < self.settle:
            action = 0.0
        else:
            action = self._clamp(action, self.action_space.low[0], self.action_space.high[0])

        # Thrust_dot = (new desired thrust - previous thrust) / (dt + tau_T)
        return np.array([self.state[1], self.state[2], (action - self.state[2]) / (self.dt + self.thrust_tc)])


    def reset(self, h0=5.0):
        # Check validity of initial height
        assert h0 >= self.MIN_H and h0 <= self.MAX_H

        # State is (height, velocity, effective thrust)
        self.state = np.array([h0, 0.0, 0.0])

        # We need a placeholder for ground truth divergence to compute div_dot
        self.div_ph = 0.0
        # Observations include noise, deque to allow for delay
        # TODO: shouldn't size of deque be delay + 1?
        # TODO: should we fill this deque beforehand? So not taking any action?
        # Zeros are just for the initial calculation of div_dot
        self.obs = deque([0.0, 0.0], maxlen=self.delay)

        self.t = 0.0

        return self._get_obs()


    def render(self, mode="human"):
        pass


    def close(self):
        pass


    def seed(self, seed):
        np.random.seed(seed)
