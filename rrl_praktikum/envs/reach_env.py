import numpy as np
from simulation.src.robot_setup.Mujoco_Scene_Object import MujocoObject, MujocoPrimitiveObject

from rrl_praktikum.envs.panda_base_env import PandaBaseEnv


class ReachEnv(PandaBaseEnv):
    def reset(self):
        self.scene.sim.reset()

        # set initial position and velocity
        qpos = self.scene.init_qpos
        qvel = self.scene.init_qvel

        # randomize x and y pos of box
        x_pos = np.random.uniform(0.25, 0.55, 1)
        y_pos = np.random.uniform(-0.4, 0.4, 1)
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
        reward = -distance

        if distance <= 0.03:
            print('success')
            reward = np.float32(1000.0) + (100 - distance * 80)
        return reward

    def _scene_objects(self):
        z_offset = 0.2
        table = MujocoPrimitiveObject(obj_pos=[0.85, 0.0, z_offset],
                                      obj_name="table",
                                      geom_size=[0.5, 0.5, 0.2],
                                      mass=2000,
                                      geom_material="table_mat")
        ball = MujocoPrimitiveObject(obj_pos=[0.5, 0.0, 0.35],
                                     obj_name='ball',
                                     mass=0.01,
                                     geom_type='sphere',
                                     geom_rgba=[1, 0, 0, 1],
                                     geom_size=[0.015, 0.015, 0.015])
        return [table, ball]


def goal_distance(obj_a, obj_b):
    return np.linalg.norm(obj_a - obj_b, axis=-1)
