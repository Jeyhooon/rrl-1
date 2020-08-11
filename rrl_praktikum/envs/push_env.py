import numpy as np
from simulation.src.robot_setup.Mujoco_Scene_Object import MujocoObject, MujocoPrimitiveObject

from rrl_praktikum.envs.panda_base_env import PandaBaseEnv


class PushEnv(PandaBaseEnv):
    def reset(self):
        self.scene.sim.reset()

        # set initial position and velocity
        qpos = self.scene.init_qpos
        qvel = self.scene.init_qvel

        # randomize x and y pos of box
        # x_pos = np.random.uniform(0.25, 0.55, 1)
        # y_pos = np.random.uniform(-0.4, 0.4, 1)
        # qpos[9] = x_pos
        # qpos[10] = y_pos
        print(qpos)
        self.agent.set_state(qpos, qvel)
        self.agent.panda.receiveState()

        return {'image': self._get_obs()}

    def _termination(self):
        return False

    def _reward(self):
        return 0

    def _scene_objects(self):
        z_offset = 0.2
        tray = MujocoObject(object_name='tray',
                            pos=[0.5, 0, z_offset],
                            quat=[0, 0, 0, 0])
        box = MujocoPrimitiveObject(obj_name='box',
                                    obj_pos=[0.5, 0, z_offset + 0.2],
                                    geom_rgba=[1, 0, 0, 1])
        return [tray, box]
