from typing import Tuple

import gym
import numpy as np
from dm_control import manipulation, suite


class DeepMindControl:
    def __init__(self, name: str, size: Tuple[int, int] = (64, 64), camera=0):
        if 'vision' in name:
            self._env = manipulation.load(environment_name=name)
        else:
            domain, task = name.split('_', 1)
            self._env = suite.load(domain, task)
        self._size = size
        self._camera = camera

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            spaces[key] = gym.spaces.Box(
                -np.inf, np.inf, value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(
            0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        time_step = self._env.step(action)
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        reward = time_step.reward or 0
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)
