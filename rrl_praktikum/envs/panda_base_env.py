import gym
import numpy as np
from simulation.src.gym_envs.mujoco.mujoco_env import MujocoEnv
from simulation.src.gym_envs.mujoco.panda_mujoco import PandaVelocityNoGripper
from simulation.src.robot_setup.Mujoco_Panda_Sim_Interface import Scene
from simulation.src.robot_setup.Mujoco_Scene_Object import MujocoCamera


class PandaBaseEnv(MujocoEnv):
    def __init__(self, max_steps=6000, control_timesteps=1, dt=0.001):
        super().__init__(max_steps=max_steps)
        camera = MujocoCamera(cam_name='rgb_front', cam_pos=[2.0, 0.0, 1.0], cam_euler=[0, 1.2, 1.57],
                              cam_mode='fixed', fovy=25)
        objects = self._scene_objects()
        self.scene = Scene(control='velocity',
                           camera_list=[camera],
                           dt=dt,
                           object_list=objects,
                           render=False)

        self.episode = 0
        self.agent = PandaVelocityNoGripper(scene=self.scene,
                                            render=False,
                                            control_timesteps=control_timesteps)

        self.reset()

    @property
    def action_space(self):
        return self.agent.get_action_space()

    @property
    def observation_space(self):
        spaces = {'image': gym.spaces.Box(
            0, 255, (64, 64) + (3,), dtype=np.uint8)}
        return gym.spaces.Dict(spaces)

    def _get_obs(self):
        return self.render()

    def render(self, mode='rgb_array'):
        return self.scene.get_rgb_image_from_cam(cam_name='rgb_front', width=64, height=64)[::-1, :, :]

    def step(self, action):
        self.agent.apply_action(action)
        obs = {'image': self._get_obs()}
        reward = self._reward()
        done = self._termination()
        self.env_step_counter += 1
        return obs, reward, done, {}

    def reset(self):
        raise NotImplementedError

    def _termination(self):
        raise NotImplementedError

    def _reward(self):
        raise NotImplementedError

    def _scene_objects(self):
        raise NotImplementedError
