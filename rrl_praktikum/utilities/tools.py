import pathlib

import tensorflow as tf
import numpy as np

from rrl_praktikum.utilities import episodes


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def args_type(default):
    if isinstance(default, bool):
        return lambda x: bool(['False', 'True'].index(x))
    if isinstance(default, int):
        return lambda x: float(x) if ('e' in x or '.' in x) else int(x)
    if isinstance(default, pathlib.Path):
        return lambda x: pathlib.Path(x).expanduser()
    return type(default)


def static_scan(fn, inputs, start, reverse=False):
    last = start
    outputs = [[] for _ in tf.nest.flatten(start)]
    indices = range(len(tf.nest.flatten(inputs)[0]))
    if reverse:
        indices = reversed(indices)
    for index in indices:
        inp = tf.nest.map_structure(lambda x: x[index], inputs)
        last = fn(last, inp)
        [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
    if reverse:
        outputs = [list(reversed(x)) for x in outputs]
    outputs = [tf.stack(x, 0) for x in outputs]
    return tf.nest.pack_sequence_as(start, outputs)


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * tf.ones_like(reward)
    dims = list(range(reward.shape.ndims))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
    if axis != 0:
        reward = tf.transpose(reward, dims)
        value = tf.transpose(value, dims)
        pcont = tf.transpose(pcont, dims)
    if bootstrap is None:
        bootstrap = tf.zeros_like(value[-1])
    next_values = tf.concat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    returns = static_scan(
        lambda agg, cur: cur[0] + cur[1] * lambda_ * agg,
        (inputs, pcont), bootstrap, reverse=True)
    if axis != 0:
        returns = tf.transpose(returns, dims)
    return returns


def simulate(agent, envs, steps=0, episodes=0, state=None):
    # Initialize or unpack simulation state.
    if state is None:
        step, episode = 0, 0
        done = np.ones(len(envs), np.bool)
        length = np.zeros(len(envs), np.int32)
        obs = [None] * len(envs)
        agent_state = None
    else:
        step, episode, done, length, obs, agent_state = state
    while (steps and step < steps) or (episodes and episode < episodes):
        # Reset envs if necessary.
        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            promises = [envs[i].reset(blocking=False) for i in indices]
            for index, promise in zip(indices, promises):
                obs[index] = promise()
        # Step agents.
        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
        action, agent_state = agent(obs, done, agent_state)
        action = np.array(action)
        assert len(action) == len(envs)
        # Step envs.
        promises = [e.step(a, blocking=False) for e, a in zip(envs, action)]
        obs, _, done = zip(*[p()[:3] for p in promises])
        obs = list(obs)
        done = np.stack(done)
        episode += int(done.sum())
        length += 1
        step += (done * length).sum()
        length *= (1 - done)
    # Return new state to allow resuming the simulation.
    return step - steps, episode - episodes, done, length, obs, agent_state


def count_steps(datadir, config):
    return episodes.count_episodes(datadir)[1] * config.action_repeat


class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if self._last is None:
            self._last = step
            return True
        if step >= self._last + self._every:
            self._last += self._every
            return True
        return False


class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False
