import numpy as np
from simulation.src.robot_setup.Mujoco_Scene_Object import MujocoPrimitiveObject

from rrl_praktikum.envs.panda_base_env import PandaBaseEnv

RED = [1, 0, 0, 1]
BLUE = [0, 0, 1, 1]
BLACK = [0, 0, 0, 1]
WHITE = [1, 1, 1, 1]

Z_OFFSET = 0.2


class KickEnv(PandaBaseEnv):
    def _scene_objects(self):
        table = MujocoPrimitiveObject(obj_pos=[0.85, 0.0, 0.1],
                                      obj_name="table",
                                      geom_size=[0.5, 0.5, 0.2],
                                      mass=2000,
                                      geom_material="table_mat")

        goalie = MujocoPrimitiveObject(obj_pos=[1.3, 0.0, 0.35],
                                       obj_name='goalie',
                                       mass=1,
                                       geom_rgba=BLUE,
                                       geom_size=[0.015, 0.015, 0.015])

        ball = MujocoPrimitiveObject(obj_pos=[0.5, 0.0, 0.35],
                                     obj_name='ball',
                                     mass=0.01,
                                     geom_type='sphere',
                                     geom_rgba=RED,
                                     geom_size=[0.015, 0.015, 0.015])

        left_goal_post = MujocoPrimitiveObject(obj_pos=[1.3, 0.1, 0.35],
                                               obj_name='left_goal_post',
                                               geom_rgba=BLUE,
                                               geom_size=[0.005, 0.005, 0.02])
        right_goal_post = MujocoPrimitiveObject(obj_pos=[1.3, -0.1, 0.35],
                                                obj_name='right_goal_post',
                                                geom_rgba=BLUE,
                                                geom_size=[0.005, 0.005, 0.02])
        left_boundary = MujocoPrimitiveObject(obj_pos=[0.9, 0.2, 0.35],
                                              obj_name='left_boundary',
                                              geom_rgba=BLACK,
                                              geom_size=[0.4, 0.005, 0.02])
        left_back_boundary = MujocoPrimitiveObject(obj_pos=[1.3, 0.15, 0.35],
                                                   obj_name='left_back_boundary',
                                                   geom_rgba=BLACK,
                                                   geom_size=[0.005, 0.05, 0.02])
        right_boundary = MujocoPrimitiveObject(obj_pos=[0.9, -0.2, 0.35],
                                               obj_name='right_boundary',
                                               geom_rgba=BLACK,
                                               geom_size=[0.4, 0.005, 0.02])
        right_back_boundary = MujocoPrimitiveObject(obj_pos=[1.3, -0.15, 0.35],
                                                    obj_name='right_back_boundary',
                                                    geom_rgba=BLACK,
                                                    geom_size=[0.005, 0.05, 0.02])

        return [table, goalie, ball, left_goal_post, right_goal_post, left_boundary, right_boundary,
                left_back_boundary, right_back_boundary]

    def reset(self):
        self.scene.sim.reset()

        # set initial position and velocity
        qpos = self.scene.init_qpos
        qvel = self.scene.init_qvel

        # randomize x and y pos of ball and goalie
        # x_pos = np.random.uniform(0.35, 0.5, 1)
        # y_pos = np.random.uniform(-0.1, 0.1, 1)
        # while abs(x_pos - 0.4) < 0.04 or abs(y_pos - 0.0) < 0.04:
        #     x_pos = np.random.uniform(0.35, 0.5, 1)
        #     y_pos = np.random.uniform(-0.1, 0.1, 1)
        # qpos[30] = x_pos
        # qpos[31] = y_pos
        # y_pos = np.random.uniform(-0.07, 0.07, 1)
        # qpos[24] = y_pos

        self.agent.set_state(qpos, qvel)
        self.agent.panda.receiveState()

        return self._get_obs()

    def _termination(self):
        return False

    def _reward(self):
        return 0
