from collections import deque

import gym
from gym import spaces

import numpy as np


class QuadHover(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        delay=3,
        noise=0.1,
        noise_p=0.1,
        g=9.81,
        g_bounds=(-0.8, 0.5),
        thrust_tc=0.02,
        settle=1.0,
        wind=0.1,
        h0=5.0,
        dt=0.02,
        ds_act=1,
        jitter=0.0,
        max_t=30.0,
        seed=0,
    ):
        # Constants
        self.MAX_H = 15.0  # treat as inclusive bounds
        self.MIN_H = 0.05

        # Keywords
        self.G = g
        self.dt = dt
        self.ds_act = ds_act  # action selection every ds_act steps (so 1 means each step)
        self.jitter_prob = jitter  # probability of computational jitter
        self.max_t = max_t
        self.settle = settle  # initial settling period without any control
        self.delay = delay  # delay in steps
        self.noise_std = noise  # white noise
        self.noise_p_std = noise_p  # noise proportional to divergence
        self.wind_std = wind
        self.thrust_tc = thrust_tc  # thrust time constant

        # Seed
        self.seed(seed)

        # Reset to get initial observation
        self.reset(h0)

        # Initialize spaces
        # Thrust value as action, (div, div_dot) as observation
        self.action_space = spaces.Box(low=g_bounds[0], high=g_bounds[1], shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

        # Check values
        self.checks()

    def checks(self):
        # Check values
        assert self.delay >= 0 and isinstance(self.delay, int)
        assert self.noise_std >= 0.0 and self.noise_p_std >= 0.0
        assert self.action_space.low[0] >= -1.0
        assert self.thrust_tc > 0.0
        assert self.settle >= 0.0
        assert self.wind_std >= 0.0
        assert self.jitter_prob >= 0.0 and self.jitter_prob <= 1.0
        assert self.dt > 0.0
        assert self.ds_act > 0 and isinstance(self.ds_act, int)
        assert self.max_t > 0.0
        assert (self.delay + 1) * self.dt < self.settle

    def step(self, action, jitter_prob=None):
        # Set computational jitter
        if jitter_prob is None:
            jitter_prob = self.jitter_prob

        # Update wind
        self._get_wind()

        # Take action
        self._get_action(action)

        # Update state with forward Euler
        self.state += (
            self._get_state_dot()
            + [0.0, self.wind + self.disturbance[0], self.disturbance[1]]
        ) * self.dt
        self.t += self.dt
        self.steps += 1

        # Check whether done
        done = self._check_done()

        # Clamp state to prevent negative altitudes
        if done:
            self.state[0] = self._clamp(self.state[0], self.MIN_H, self.MAX_H)

        # Get reward
        reward = self._get_reward()

        # Computational jitter: do another step (without possibility of another delay)
        if np.random.random() < jitter_prob:
            self._get_obs()
            return self.step(action, jitter_prob=0.0)

        return self._get_obs(), reward, done, {}

    def set_disturbance(self, v_disturbance, a_disturbance):
        self.disturbance = [v_disturbance, a_disturbance]

    def unset_disturbance(self):
        self.disturbance = [0.0, 0.0]

    def _get_action(self, action):
        # Take new action
        if not self.steps % self.ds_act:
            self.action = action
        # Else keep previous action

    def _get_wind(self):
        if self.wind_std > 0.0:
            self.wind += (
                (np.random.normal(0.0, self.wind_std) - self.wind)
                * self.dt
                / (self.dt + self.wind_std)
            )

    def _get_obs(self):
        # Compute ground truth divergence
        div = -2.0 * self.state[1] / max(1e-5, self.state[0])
        div_dot = (div - self.div_ph[0]) / self.dt

        # Overwrite placeholder
        self.div_ph[:] = [div, div_dot]

        # Add noise (regular and proportional)
        # Use old noisy estimate for noisy div_dot
        div += np.random.normal(0.0, self.noise_std) + abs(div) * np.random.normal(
            0.0, self.noise_p_std
        )
        div_dot = (div - self.obs[-1][0]) / self.dt

        # Append to end of deque; if == max length then first is popped
        self.obs.append([div, div_dot])

        return np.array(self.obs[0], dtype=np.float32)

    def _clamp(self, value, minimum, maximum):
        return max(min(value, maximum), minimum)

    def _check_done(self):
        return self._check_out_of_bounds() or self._check_out_of_time()

    def _check_out_of_bounds(self):
        return self.state[0] < self.MIN_H or self.state[0] > self.MAX_H

    def _check_out_of_time(self):
        return self.t >= self.max_t

    def _get_reward(self):
        # Use raw states because placeholder hasn't been updated yet
        return 1.0 - np.abs(-2.0 * self.state[1] / max(1e-5, self.state[0]))

    def _get_state_dot(self):
        # Action is delta G relative to hover G in Gs
        # So: state_dot for the first two states (height, velocity)
        # is just the last two states (velocity, action * G in m/s^2)!
        # First do nothing for some time, to allow settling of controller
        # and filling of deque
        if self.t < self.settle:
            action = 0.0
        else:
            action = self._clamp(
                self.action, self.action_space.low[0], self.action_space.high[0]
            )

        # Thrust_dot = (new desired thrust - previous thrust) / (dt + tau_T)
        return np.array(
            [
                self.state[1],
                self.state[2],
                (action * self.G - self.state[2]) / (self.dt + self.thrust_tc),
            ],
            dtype=np.float32,
        )

    def reset(self, h0=5.0):
        # Check validity of initial height
        # assert h0 >= self.MIN_H and h0 <= self.MAX_H
        assert h0 >= self.MIN_H
        self.MAX_H = h0 + 5.0

        # State is (height, velocity, effective thrust)
        self.state = np.array([h0, 0.0, 0.0], dtype=np.float32)

        # We need a placeholder for ground truth divergence to compute div_dot
        self.div_ph = np.array([0.0, 0.0], dtype=np.float32)
        # Observations include noise, deque to allow for delay
        # Zeros are just for the initial calculation of div_dot
        self.obs = deque([[0.0, 0.0]], maxlen=self.delay + 1)

        # Other: variables that are always initialized at 0
        self.t = 0.0
        self.steps = 0
        self.wind = 0.0
        self.action = 0.0
        self.disturbance = [0.0, 0.0]

        return self._get_obs()

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.seeds = seed
        np.random.seed(seed)
