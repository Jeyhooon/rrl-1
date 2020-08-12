from matplotlib import pyplot as plt

from rrl_praktikum.envs.kick_env import KickEnv
from rrl_praktikum.envs.push_env import PushEnv
from rrl_praktikum.envs.reach_env import ReachEnv

if __name__ == '__main__':
    env = PushEnv()

    obs, rewards, done, _ = env.step(env.action_space.sample())

    plt.imshow(obs['image'])
    plt.show()
