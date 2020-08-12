import numpy as np
from simulation.src.robot_setup.Mujoco_Scene_Object import MujocoObject, MujocoPrimitiveObject

from rrl_praktikum.envs.panda_base_env import PandaBaseEnv


class ReachEnv(PandaBaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self):
        self.scene.sim.reset()

        # set initial position and velocity
        qpos = self.scene.init_qpos
        qvel = self.scene.init_qvel

        # randomize x and y pos of box
        x_pos = np.random.uniform(0.45, 0.55, 1)
        y_pos = np.random.uniform(-0.05, 0.05, 1)
        qpos[9] = x_pos
        qpos[10] = y_pos

        self.agent.set_state(qpos, qvel)
        self.agent.panda.receiveState()

        return {'image': self._get_obs()}

    def _termination(self):
        end_eff_pose = self.agent.tcp_pos
        box_pos = self.scene.sim.data.qpos[9:12]

        # calculate the distance from end effector to object
        d = goal_distance(np.array(box_pos), np.array(end_eff_pose))

        if d <= 0.03:
            self.terminated = True

        if self.terminated or self.env_step_counter > self.max_steps:
            self.terminated = 0
            self.env_step_counter = 0
            self.episode += 1
            self._observation = self._get_obs()
            return True
        return False

    def _reward(self):
        box_pos = self.scene.sim.data.qpos[9:12]
        end_eff_coords = self.agent.tcp_pos
        distance = goal_distance(np.array(end_eff_coords), np.array(box_pos))
        goal_reached = np.logical_and(0.0 <= distance, distance <= 0.1)
        d = np.where(distance < 0.0, 0.0 - distance, distance - 0.1) / 0.1
        scale = np.sqrt(-2 * np.log(0.1))
        value = np.where(goal_reached, 1.0, np.exp(-0.5 * (d*scale)**2))
        return float(value) if np.isscalar(distance) else value

    def _scene_objects(self):
        z_offset = 0.2
        tray = MujocoObject(object_name='tray',
                            pos=[0.5, 0, z_offset],
                            quat=[0, 0, 0, 0])
        obj1 = MujocoPrimitiveObject(obj_name='box',
                                     obj_pos=[0.5, 0, z_offset + 0.2],
                                     geom_rgba=[1, 0, 0, 1])
        return [tray, obj1]


def goal_distance(obj_a, obj_b):
    return np.linalg.norm(obj_a - obj_b, axis=-1)
