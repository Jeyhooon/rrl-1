from matplotlib import pyplot as plt
from simulation.src.robot_setup.Mujoco_Scene_Object import MujocoPrimitiveObject
from simulation.src.robot_setup import Robots
from simulation.src.robot_setup.Mujoco_Panda_Sim_Interface import Scene

from rrl_praktikum.envs.kick_env import KickEnv
from rrl_praktikum.envs.kicking_env import KickingEnv
from rrl_praktikum.envs.push_env import PushEnv
from rrl_praktikum.envs.reach_env import ReachEnv

RED = [1, 0, 0, 1]
BLUE = [0, 0, 1, 1]
BLACK = [0, 0, 0, 1]
WHITE = [1, 1, 1, 1]

EPSILON = 0.005


def run_simple_demo():
    table = MujocoPrimitiveObject(obj_pos=[0.9, 0.0, 0.2],
                                  obj_name="table",
                                  geom_size=[0.6, 0.5, 0.2],
                                  mass=2000,
                                  geom_material="table_mat")

    cue_blue = MujocoPrimitiveObject(obj_pos=[0.4, 0.0, 0.35],
                                     obj_name='cue_blue',
                                     mass=0.01,
                                     geom_rgba=BLUE,
                                     geom_size=[0.015, 0.015, 0.015])

    cue_red = MujocoPrimitiveObject(obj_pos=[1.3, 0.0, 0.35],
                                    obj_name='cue_red',
                                    mass=1,
                                    geom_rgba=RED,
                                    geom_size=[0.015, 0.015, 0.015])

    ball1 = MujocoPrimitiveObject(obj_pos=[0.5, 0.0, 0.35],
                                  obj_name='ball1',
                                  mass=0.01,
                                  geom_type='sphere',
                                  geom_rgba=WHITE,
                                  geom_size=[0.015, 0.015, 0.015])

    ball2 = MujocoPrimitiveObject(obj_pos=[0.45, 0.07, 0.35],
                                  obj_name='ball2',
                                  mass=0.01,
                                  geom_type='sphere',
                                  geom_rgba=WHITE,
                                  geom_size=[0.015, 0.015, 0.015])

    goal_red = _goal_posts(1.3, RED, 'red')
    boundaries = _boundaries()

    object_list = [table, cue_red, ball1, ball2] + goal_red + boundaries
    scene = Scene(object_list=object_list)
    robot = Robots.MuJoCoRobot(scene, gravity_comp=True, num_DoF=7)

    duration = 2
    robot.ctrl_duration = duration
    robot.startLogging()

    home_position = robot.current_c_pos.copy()
    home_orientation = robot.current_c_quat.copy()

    # grab cue
    # robot.gotoCartPositionAndQuat([0.4, 0.0, 0.5], [0, 1, 0, 0], duration=duration)
    # robot.set_gripper_width = 0.04
    # robot.gotoCartPositionAndQuat([0.4, 0.0, 0.41], [0, 1, 0, 0], duration=duration)
    # robot.set_gripper_width = 0.0

    # shoot ball1
    robot.gotoCartPositionAndQuat([0.4, -0.02, 0.42], [0, 1, 0, 0], duration=duration)
    robot.gotoCartPositionAndQuat([0.5, 0.03, 0.41], [0, 1, 0, 0], duration=0.18)

    # shoot ball2
    robot.gotoCartPositionAndQuat([0.35, 0.0, 0.42], [0, 1, 0, 0], duration=duration)
    robot.gotoCartPositionAndQuat([0.35, 0.09, 0.42], [0, 1, 0, 0], duration=duration)
    robot.gotoCartPositionAndQuat([0.45, 0.04, 0.41], [0, 1, 0, 0], duration=0.19)

    # go home
    robot.gotoCartPositionAndQuat(home_position, home_orientation, duration=duration)

    robot.stopLogging()


def _goal_posts(center, rgba, team_name):
    assert len(rgba) == 4

    left_goal_post = MujocoPrimitiveObject(obj_pos=[center, 0.1, 0.35],
                                           obj_name=f'left_goal_post_{team_name}',
                                           geom_rgba=rgba,
                                           geom_size=[0.005, 0.005, 0.02])

    right_goal_post = MujocoPrimitiveObject(obj_pos=[center, -0.1, 0.35],
                                            obj_name=f'right_goal_post_{team_name}',
                                            geom_rgba=rgba,
                                            geom_size=[0.005, 0.005, 0.02])

    return [left_goal_post, right_goal_post]


def _boundaries():
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
    return [left_boundary, right_boundary, left_back_boundary, right_back_boundary]


# def _print_step_result(obs, rewards, done, step):
#     print(f"Step: {step}")
#     print(f"tcp_pos: {obs[18:21]}")
#     print(f"player_pos: {obs[21:24]}, goalie_pos:{obs[24:27]}, ball_pos: {obs[27:]}")
#     print(f'Reward: {rewards}')
#     print(f"done: {done}\n\n")


if __name__ == '__main__':
    run_simple_demo()
