import numpy as np
from simulation.src.robot_setup.Mujoco_Scene_Object import MujocoObject, MujocoPrimitiveObject

from rrl_praktikum.envs.panda_base_env import PandaBaseEnv


class PushEnv(PandaBaseEnv):
    """
    Push two boxes as close together as possible.
    """
    def __init__(self, mode='easy', reward_type='distance_only', min_distance=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.reward_type = reward_type
        self.min_distance = min_distance
        self.initial_distance = 0.2

    def reset(self):
        self.scene.sim.reset()

        # set initial position and velocity
        qpos = self.scene.init_qpos
        qvel = self.scene.init_qvel

        # randomize x and y pos of boxes
        # red box is always in easy reach
        x_pos_red = np.random.uniform(0.45, 0.65, 1)
        y_pos_red = np.random.uniform(-0.2, 0.2, 1)
        x_pos_blue = x_pos_red
        y_pos_blue = y_pos_red
        while box_distance(x_pos_red, y_pos_red, x_pos_blue, y_pos_blue) < self.min_distance:
            if self.mode == 'easy':
                # blue box also in easy reach
                x_pos_blue = np.random.uniform(0.45, 0.6, 1)
                y_pos_blue = np.random.uniform(-0.2, 0.2, 1)
            elif self.mode == 'hard':
                # blue box may be harder to reach
                x_pos_blue = np.random.uniform(0.35, 0.6, 1)
                y_pos_blue = np.random.uniform(-0.5, 0.5, 1)
            else:
                raise NotImplementedError

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
            return 100 * self.initial_distance / current_distance
        else:
            raise NotImplementedError

    def _scene_objects(self):
        z_offset = 0.2
        tray = MujocoObject(object_name='tray',
                            pos=[0.5, 0, z_offset],
                            quat=[0, 0, 0, 0])
        red_box = MujocoPrimitiveObject(obj_name='red_box',
                                        obj_pos=[0.55, 0.1, z_offset + 0.2],
                                        geom_rgba=[1, 0, 0, 1])
        blue_box = MujocoPrimitiveObject(obj_name='blue_box',
                                         obj_pos=[0.55, -0.1, z_offset + 0.2],
                                         geom_rgba=[0, 0, 1, 1])
        return [tray, red_box, blue_box]


def box_distance(x_pos_red, y_pos_red, x_pos_blue, y_pos_blue):
    return np.linalg.norm(np.array([x_pos_red, y_pos_red]) - np.array([x_pos_blue, y_pos_blue]))
