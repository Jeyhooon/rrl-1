import gym
from simulation.src.gym_envs.mujoco.mujoco_env import MujocoEnv
from simulation.src.robot_setup.Mujoco_Camera import Camera
from simulation.src.robot_setup.Mujoco_Scene_Object import MujocoPrimitiveObject, MujocoCamera, MujocoObject
import numpy as np
from simulation.src.robot_setup.Mujoco_Panda_Sim_Interface import Scene
from simulation.src.gym_envs.mujoco.ik_controller import IkController
from simulation.src.gym_envs.mujoco.panda_mujoco import PandaInverseKinematics, PandaTorque, PandaJointControl
from simulation.src.gym_envs.mujoco.utils import goal_distance

from rrl_praktikum.utilities.tools import AttrDict

RED = [1, 0, 0, 1]
BLUE = [0, 0, 1, 1]
BLACK = [0, 0, 0, 1]
WHITE = [1, 1, 1, 1]

Z_OFFSET = 0.2


class KickingEnv(MujocoEnv):
    """
    Kicking task: The agent is asked to kick a ball into a goal. He gets more reward the closer the ball gets to the
    goal and a huge reward when he scores.
    """
    def __init__(self,
                 max_steps=2000,
                 control='velocity',
                 kv=None,
                 kv_scale=1,
                 kp=None,
                 kp_scale=None,
                 controller=None,
                 coordinates='absolute',
                 step_limitation='percentage',
                 percentage=0.05,
                 vector_norm=0.01,
                 dt=0.001,
                 trajectory_length=100,
                 control_timesteps=1,
                 include_objects=True,
                 randomize_ball=True,
                 randomize_goalie=True,
                 render=True):
        """
        Args:
            max_steps:
                Maximum number of steps per episode
            control:
                Choose between:
                <ik> (inverse kinematics),
                <position> (joint position control)
                <torque> (torque based control) or
                <velocity> (joint velocity control)
            kv:
                Velocity feedback gain for each joint (list with num joints entries)
            kv_scale:
                Scales each Velocity feedback gain by a scalar (single value)
            kp:
                Position feedback gain for each joint (list with num joints entries)
            kp_scale:
                Scales each Position feedback gain by a scalar (single value)
            controller:
                Used for controlling the robot using inverse kinematics.
            coordinates:
                Choose between:
                <absolute> 3D cartesian coordinates where the robot should move
                <relative> 3D directional vector in which the robot should move
            step_limitation:
                Choose between:
                <percentage> Limits the movement of the robot according to a given percentage
                <norm> Sets the upper bound of the movement of the robot according to a given vector norm
            percentage:
                Percentage of the given distance that the robot actually moves. Has no effect when <norm> as
                step_limitation is chosen.
            vector_norm:
                Vector norm for the distance the robot can move with a single command. Has no effect when
                <percentage> is chosen.
            dt:
                1 / number of timesteps needed for computing one second of wall-clock time
            trajectory_length:
                Length of trajectories used for moving the robot.
            render:
                Determines if the scene should be visualized.
        """
        super().__init__(max_steps=max_steps)
        self.include_objects = include_objects
        self.randomize_ball = randomize_ball
        self.randomize_goalie = randomize_goalie

        objects = self._scene_objects_new()
        camera = MujocoCamera(cam_name='rgb_front', cam_pos=[2.0, 0.0, 1.0], cam_euler=[0, 1.2, 1.57],
                              cam_mode='fixed', fovy=25)

        self.scene = Scene(control=control,
                           camera_list=[camera],
                           dt=dt,
                           object_list=objects,
                           render=render,
                           kv=kv,
                           kv_scale=kv_scale,
                           kp=kp,
                           kp_scale=kp_scale)

        self.controller = None
        self.episode = 0

        if control == 'ik':  # inverse kinematics control
            if controller is None:
                self.controller = IkController()
            else:
                self.controller = controller
            self.agent = PandaInverseKinematics(scene=self.scene,
                                                controller=self.controller,
                                                coordinates=coordinates,
                                                step_limitation=step_limitation,
                                                percentage=percentage,
                                                vector_norm=vector_norm,
                                                dt=dt,
                                                trajectory_length=trajectory_length,
                                                render=render)
        elif control == 'velocity':  # joint velocity control
            self.agent = PandaJointControl(scene=self.scene,
                                           control=control,
                                           render=render,
                                           control_timesteps=control_timesteps)

        elif control == 'position':  # forward kinematics control
            self.agent = PandaJointControl(scene=self.scene,
                                           control=control,
                                           render=render,
                                           control_timesteps=control_timesteps)

        elif control == 'torque':  # torque control
            self.agent = PandaTorque(scene=self.scene,
                                     render=render)
        else:
            raise ValueError("Error, invalid control value. Choose between <ik> (inverse kinematics), "
                             "<position> (forward kinematics), <torque> (torque based control) or <velocity> (joint "
                             "velocity control).")

        self.action_space = self.agent.get_action_space()

        # upper and lower bound on the observations
        self.observation_bound = 100
        self.observation_high = np.array([self.observation_bound] * self.get_observation_dimension())
        self.observation_space = gym.spaces.Box(low=-self.observation_high, high=self.observation_high)

        self.reset()

    def _scene_objects_new(self):
        tray = MujocoObject(object_name='tray',
                            pos=[0.5, 0, Z_OFFSET],
                            quat=[0, 0, 0, 0])
        ball = MujocoPrimitiveObject(obj_pos=[0.5, 0.0, Z_OFFSET + 0.2],
                                     obj_name='ball',
                                     mass=0.01,
                                     geom_type='sphere',
                                     geom_rgba=RED,
                                     geom_size=[0.015, 0.015, 0.015])
        return [tray, ball]

    def _scene_objects(self):
        table = MujocoPrimitiveObject(obj_pos=[0.85, 0.0, 0.2],
                                      obj_name="table",
                                      geom_size=[0.5, 0.5, 0.2],
                                      mass=2000,
                                      geom_material="table_mat")

        player = MujocoPrimitiveObject(obj_pos=[0.4, 0.0, 0.35],
                                       obj_name='player',
                                       mass=0.01,
                                       geom_rgba=BLUE,
                                       geom_size=[0.015, 0.015, 0.015])
        goalie = MujocoPrimitiveObject(obj_pos=[1.3, 0.0, 0.35],
                                       obj_name='goalie',
                                       mass=1,
                                       geom_rgba=RED,
                                       geom_size=[0.015, 0.015, 0.015])
        ball = MujocoPrimitiveObject(obj_pos=[0.5, 0.0, 0.35],
                                     obj_name='ball',
                                     mass=0.01,
                                     geom_type='sphere',
                                     geom_rgba=WHITE,
                                     geom_size=[0.015, 0.015, 0.015])

        left_goal_post = MujocoPrimitiveObject(obj_pos=[1.3, 0.1, 0.35],
                                               obj_name=f'left_goal_post',
                                               geom_rgba=RED,
                                               geom_size=[0.005, 0.005, 0.02])
        right_goal_post = MujocoPrimitiveObject(obj_pos=[1.3, -0.1, 0.35],
                                                obj_name=f'right_goal_post',
                                                geom_rgba=RED,
                                                geom_size=[0.005, 0.005, 0.02])
        left_boundary = MujocoPrimitiveObject(obj_pos=[0.8, 0.2, 0.35],
                                              obj_name='left_boundary',
                                              geom_rgba=BLACK,
                                              geom_size=[0.5, 0.005, 0.02])
        left_back_boundary = MujocoPrimitiveObject(obj_pos=[1.3, 0.15, 0.35],
                                                   obj_name='left_back_boundary',
                                                   geom_rgba=BLACK,
                                                   geom_size=[0.005, 0.05, 0.02])
        right_boundary = MujocoPrimitiveObject(obj_pos=[0.8, -0.2, 0.35],
                                               obj_name='right_boundary',
                                               geom_rgba=BLACK,
                                               geom_size=[0.5, 0.005, 0.02])
        right_back_boundary = MujocoPrimitiveObject(obj_pos=[1.3, -0.15, 0.35],
                                                    obj_name='right_back_boundary',
                                                    geom_rgba=BLACK,
                                                    geom_size=[0.005, 0.05, 0.02])

        if self.include_objects:
            obj_list = [table, player, goalie, ball, left_goal_post, right_goal_post, left_boundary,
                        right_boundary, left_back_boundary, right_back_boundary]
        else:
            obj_list = []

        return obj_list

    def reset(self):
        self.scene.sim.reset()

        # set initial position and velocity
        qpos = self.scene.init_qpos
        qvel = self.scene.init_qvel

        # # randomize x and y pos of ball and goalie
        # if self.randomize_ball:
        #     x_pos = np.random.uniform(0.35, 0.5, 1)
        #     y_pos = np.random.uniform(-0.1, 0.1, 1)
        #     while abs(x_pos - 0.4) < 0.04 or abs(y_pos - 0.0) < 0.04:
        #         x_pos = np.random.uniform(0.35, 0.5, 1)
        #         y_pos = np.random.uniform(-0.1, 0.1, 1)
        #     qpos[30] = x_pos
        #     qpos[31] = y_pos
        # if self.randomize_goalie:
        #     y_pos = np.random.uniform(-0.07, 0.07, 1)
        #     qpos[24] = y_pos

        self.agent.set_state(qpos, qvel)
        self.agent.panda.receiveState()

        if self.controller:
            self.controller.reset()

        return self._get_obs()

    def step(self, action):
        self.agent.apply_action(action)
        self._observation = self._get_obs()
        done = self._termination()
        reward = self._reward()
        self.env_step_counter += 1
        return self._observation, reward, done, {}

    def _get_obs(self):
        # self.agent.panda.receiveState()
        #
        # # current_coord_velocity = self.panda.current_c_vel
        # # current_coord_orientation = self.panda.current_c_pos
        # # current_coord_orientation_velocity = self.panda.current_c_quat_vel
        # current_joint_position = self.agent.panda.current_j_pos
        # current_joint_velocity = self.agent.panda.current_j_vel
        # current_finger_position = self.agent.panda.current_fing_pos
        # current_finger_velocity = self.agent.panda.current_fing_vel
        # # current_load = np.array(self.agent.panda.current_load)
        # current_coord_position = self.agent.panda.current_c_pos
        #
        # player_pos = self.scene.sim.data.qpos[16:19]
        # goalie_pos = self.scene.sim.data.qpos[23:26]
        # ball_pos = self.scene.sim.data.qpos[30:33]
        # # ball_vel = self.scene.sim.data.qvel[30]
        #
        # obs = np.concatenate([current_joint_position,
        #                       current_joint_velocity,
        #                       current_finger_position,
        #                       current_finger_velocity,
        #                       current_coord_position,
        #                       player_pos,
        #                       goalie_pos,
        #                       ball_pos])

        obs = self.scene.get_rgb_image_from_cam(cam_name='rgb_front', width=64, height=64)
        return obs
        # return np.array([0])

    def get_observation_dimension(self):
        return self._get_obs().size

    def _parse_obs(self):
        # obs = self._observation
        # player_pos = obs[21:24]
        # goalie_pos = obs[24:27]
        # ball_pos = obs[27:]
        # return player_pos, goalie_pos, ball_pos

        return self._observation

    def _termination(self):
        # _, _, ball_pos = self._parse_obs()
        #
        # # TODO: add check if ball is out of reach and not moving
        # # goal scored
        # if ball_pos[0] > 1.3:
        #     self.terminated = True
        #
        # if self.terminated or self.env_step_counter > self.max_steps:
        #     self.terminated = 0
        #     self.env_step_counter = 0
        #     self.episode += 1
        #     self._observation = self._get_obs()
        #     return True
        return False

    def _reward(self):
        # player_pos, goalie_pos, ball_pos = self._parse_obs()
        # end_eff_coords = self.agent.tcp_pos
        # distance_player_tcp = goal_distance(np.array(end_eff_coords), np.array(player_pos))
        # if ball_pos[1] < -0.995:
        #     closest_goal_point = [1.3, -0.995, 0.35]
        # elif ball_pos[1] > 0.995:
        #     closest_goal_point = [1.3, 0.995, 0.35]
        # else:
        #     closest_goal_point = [1.3, ball_pos[1], 0.35]
        # distance_ball_goal = goal_distance(np.array(ball_pos), np.array(closest_goal_point))
        # distance_ball_table = abs(ball_pos[2] - 0.41)
        #
        # reward = -distance_player_tcp - 5 * distance_ball_goal - 10 * distance_ball_table
        #
        # if ball_pos[0] > 1.3:
        #     print('success')
        #     reward = np.float32(1000.0)
        # return reward
        return 0
