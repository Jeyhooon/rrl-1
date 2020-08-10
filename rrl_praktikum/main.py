import argparse
import functools
import json
import pathlib

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

from rrl_praktikum.envs.panda_base_env import PandaBaseEnv
from rrl_praktikum.envs.wrappers.async_wrapper import Async
from rrl_praktikum.envs.deepmind_control import DeepMindControl
from rrl_praktikum.envs.wrappers.action_repeat import ActionRepeat
from rrl_praktikum.envs.wrappers.collect import Collect
from rrl_praktikum.envs.wrappers.normalize_actions import NormalizeActions
from rrl_praktikum.envs.wrappers.reward_obs import RewardObs
from rrl_praktikum.envs.wrappers.time_limit import TimeLimit
from rrl_praktikum.models.dreamer import Dreamer
from rrl_praktikum.utilities import tools, summaries, episodes
from rrl_praktikum.envs.kick_env import KickEnv
from rrl_praktikum.envs.reach_env import ReachEnv

_KNOWN_TASKS = [ReachEnv, KickEnv]


def define_config():
    config = tools.AttrDict()
    # General.
    config.logdir = pathlib.Path('.')
    config.seed = 0
    config.steps = 5e6
    config.eval_every = 1e4
    config.log_every = 1e3
    config.log_scalars = True
    config.log_images = True
    config.gpu_growth = True
    config.precision = 16
    # Environment.
    config.task = 'dmc_reach_duplo_vision'
    config.envs = 1
    config.parallel = 'none'
    config.action_repeat = 1
    config.time_limit = 999
    config.prefill = 5000
    config.eval_noise = 0.0
    config.clip_rewards = 'none'
    # Model.
    config.deter_size = 200
    config.stoch_size = 30
    config.num_units = 400
    config.dense_act = 'elu'
    config.cnn_act = 'relu'
    config.cnn_depth = 32
    config.pcont = False
    config.free_nats = 3.0
    config.kl_scale = 1.0
    config.pcont_scale = 10.0
    config.weight_decay = 0.0
    config.weight_decay_pattern = r'.*'
    # Training.
    config.batch_size = 50
    config.batch_length = 50
    config.train_every = 1000
    config.train_steps = 100
    config.pretrain = 100
    config.model_lr = 6e-4
    config.value_lr = 8e-5
    config.actor_lr = 8e-5
    config.grad_clip = 100.0
    config.dataset_balance = False
    # Behavior.
    config.discount = 0.99
    config.disclam = 0.95
    config.horizon = 15
    config.action_dist = 'tanh_normal'
    config.action_init_std = 5.0
    config.expl = 'additive_gaussian'
    config.expl_amount = 0.3
    config.expl_decay = 0.0
    config.expl_min = 0.0
    return config


def summarize_episode(episode, config, datadir, writer, prefix):
    episodes_, steps = episodes.count_episodes(datadir)
    length = (len(episode['reward']) - 1) * config.action_repeat
    ret = episode['reward'].sum()
    print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
    metrics = [
        (f'{prefix}/return', float(episode['reward'].sum())),
        (f'{prefix}/length', len(episode['reward']) - 1),
        (f'episodes', episodes_)]
    step = tools.count_steps(datadir, config)
    with (config.logdir / 'metrics.jsonl').open('a') as f:
        f.write(json.dumps(dict([('step', step)] + metrics)) + '\n')
    with writer.as_default():  # Env might run in a different thread.
        tf.summary.experimental.set_step(step)
        [tf.summary.scalar('sim/' + k, v) for k, v in metrics]
        if prefix == 'test':
            summaries.video_summary(f'sim/{prefix}/video', episode['image'][None])


def make_env(config, writer, prefix, datadir, store):
    suite, task = config.task.split('_', 1)
    if suite == 'dmc':
        env = DeepMindControl(task)
    elif suite == 'panda':
        env = _resolve_panda_env(task_name=task)
    else:
        raise NotImplementedError(suite)
    env = ActionRepeat(env, config.action_repeat)
    env = NormalizeActions(env)
    env = TimeLimit(env, config.time_limit / config.action_repeat)
    callbacks = []
    if store:
        callbacks.append(lambda ep: episodes.save_episodes(datadir, [ep]))
    callbacks.append(lambda ep: summarize_episode(ep, config, datadir, writer, prefix))
    env = Collect(env, callbacks, config.precision)
    env = RewardObs(env)
    return env


def _resolve_panda_env(task_name):
    env_name = f'{task_name.capitalize()}Env'
    for task in _KNOWN_TASKS:
        if task.__name__ == env_name:
            return task()
    raise NotImplementedError(f'Task {task_name} is not implemented for Panda.')


def main(config):
    if config.gpu_growth:
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        prec.set_policy(prec.Policy('mixed_float16'))
    config.steps = int(config.steps)
    config.logdir.mkdir(parents=True, exist_ok=True)
    print('Logdir', config.logdir)

    # Create environments.
    datadir = config.logdir / 'episodes'
    writer = tf.summary.create_file_writer(
        str(config.logdir), max_queue=1000, flush_millis=20000)
    writer.set_as_default()
    train_envs = [Async(lambda: make_env(config, writer, 'train', datadir, store=True), config.parallel)
                  for _ in range(config.envs)]
    test_envs = [Async(lambda: make_env(config, writer, 'test', datadir, store=False), config.parallel)
                 for _ in range(config.envs)]
    actspace = train_envs[0].action_space

    # Prefill dataset with random episodes.
    step = tools.count_steps(datadir, config)
    prefill = max(0, config.prefill - step)
    print(f'Prefill dataset with {prefill} steps.')
    random_agent = lambda o, d, _: ([actspace.sample() for _ in d], None)
    tools.simulate(random_agent, train_envs, prefill / config.action_repeat)
    writer.flush()
    # Train and regularly evaluate the agent.
    step = tools.count_steps(datadir, config)
    print(f'Simulating agent for {config.steps-step} steps.')
    agent = Dreamer(config, datadir, actspace, writer)
    if (config.logdir / 'variables.pkl').exists():
        print('Load checkpoint.')
        agent.load(config.logdir / 'variables.pkl')
    state = None
    while step < config.steps:
        print('Start evaluation.')
        tools.simulate(
            functools.partial(agent, training=False), test_envs, episodes=1)
        writer.flush()
        print('Start collection.')
        steps = config.eval_every // config.action_repeat
        state = tools.simulate(agent, train_envs, steps, state=state)
        step = tools.count_steps(datadir, config)
        agent.save(config.logdir / 'variables.pkl')
    for env in train_envs + test_envs:
        env.close()


if __name__ == '__main__':
    try:
        import colored_traceback
        colored_traceback.add_hook()
    except ImportError:
        pass
    parser = argparse.ArgumentParser()
    for key, value in define_config().items():
        parser.add_argument(f'--{key}', type=tools.args_type(value), default=value)
    main(parser.parse_args())
