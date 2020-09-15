import numpy as np
from simulation.src.robot_setup.Mujoco_Scene_Object import MujocoPrimitiveObject, MujocoObject

from rrl_praktikum.envs.panda_base_env import PandaBaseEnv

RED = [1, 0, 0, 1]
BLUE = [0, 0, 1, 1]
BLACK = [0, 0, 0, 1]
WHITE = [1, 1, 1, 1]

Z_OFFSET = 0.2


class KickEnv(PandaBaseEnv):
    """
    Kick the reachable box as close to the unreachable box as possible.
    """
    def __init__(self, reward_type='distance_only', **kwargs):
        self.reward_type = reward_type
        self.initial_distance = 0.5220153
        super().__init__(**kwargs)

    def _scene_objects(self):
        z_offset = 0.2
        friction = [0.001, 0.001, 0.0001]
        tray = MujocoObject(object_name='tray',
                            pos=[0.5, 0, z_offset],
                            quat=[0, 0, 0, 0],
                            friction=friction)
        red_box = MujocoPrimitiveObject(obj_name='red_box',
                                        obj_pos=[0.55, 0, z_offset + 0.2],
                                        geom_rgba=[1, 0, 0, 1],
                                        geom_friction=friction)
        blue_box = MujocoPrimitiveObject(obj_name='blue_box',
                                         obj_pos=[0.7, 0.5, z_offset + 0.2],
                                         geom_rgba=[0, 0, 1, 1])
        return [tray, red_box, blue_box]

    def reset(self):
        self.scene.sim.reset()

        # set initial position and velocity
        qpos = self.scene.init_qpos
        qvel = self.scene.init_qvel

        # randomize x and y pos of boxes
        x_pos_red = np.random.uniform(0.45, 0.65, 1)
        y_pos_red = np.random.uniform(-0.2, 0.2, 1)
        x_pos_blue = np.random.uniform(0.6, 0.7, 1)
        y_pos_blue = np.random.uniform(0.4, 0.5, 1)  # * np.where(np.random.binomial(1, 0.5), 1, -1)

        self.initial_distance = box_distance(x_pos_red, y_pos_red, x_pos_blue, y_pos_blue)
        qpos[9] = x_pos_red
        qpos[10] = y_pos_red
        qpos[16] = x_pos_blue
        qpos[17] = y_pos_blue
        self.agent.set_state(qpos, qvel)
        self.agent.panda.receiveState()

        return {'image': self._get_obs()}

    def _reward(self):
        red_box_pos = self.scene.sim.data.qpos[9:12]
        blue_box_pos = self.scene.sim.data.qpos[16:19]
        current_distance = box_distance(red_box_pos[0], red_box_pos[1], blue_box_pos[0], blue_box_pos[1])
        if self.reward_type == 'distance_only':
            return 100 * (self.initial_distance - current_distance) / self.initial_distance
        else:
            raise NotImplementedError


def box_distance(x_pos_red, y_pos_red, x_pos_blue, y_pos_blue):
    return np.linalg.norm(np.array([x_pos_red, y_pos_red]) - np.array([x_pos_blue, y_pos_blue]))
