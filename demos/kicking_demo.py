from simulation.src.robot_setup.Mujoco_Scene_Object import MujocoPrimitiveObject
from simulation.src.robot_setup import Robots
from simulation.src.robot_setup.Mujoco_Panda_Sim_Interface import Scene


def run_demo():
    table = MujocoPrimitiveObject(obj_pos=[0.8, 0.0, 0.2],
                                  obj_name="table",
                                  geom_size=[0.5, 0.35, 0.2],
                                  mass=2000,
                                  geom_material="table_mat")

    cue = MujocoPrimitiveObject(obj_pos=[0.55, 0.0, 0.2],
                                obj_name='cue',
                                mass=0.01,
                                geom_rgba=[0, 0, 0, 1])

    ball = MujocoPrimitiveObject(obj_pos=[0.6, 0.0, 0.2],
                                 obj_name='ball',
                                 mass=0.01,
                                 geom_rgba=[1, 1, 1, 1])

    left_front_goal_post = MujocoPrimitiveObject(obj_pos=[1, 0.1, 0.2],
                                                 obj_name='left_front_goal_post',
                                                 geom_rgba=[1, 0, 0, 1],
                                                 geom_size=[0.01, 0.01, 0.01])

    right_front_goal_post = MujocoPrimitiveObject(obj_pos=[1, -0.1, 0.2],
                                                  obj_name='right_front_goal_post',
                                                  geom_rgba=[1, 0, 0, 1],
                                                  geom_size=[0.01, 0.01, 0.01])

    left_back_goal_post = MujocoPrimitiveObject(obj_pos=[1.07, 0.1, 0.2],
                                                obj_name='left_back_goal_post',
                                                geom_rgba=[1, 0, 0, 1],
                                                geom_size=[0.01, 0.01, 0.01])

    right_back_goal_post = MujocoPrimitiveObject(obj_pos=[1.07, -0.1, 0.2],
                                                 obj_name='right_back_goal_post',
                                                 geom_rgba=[1, 0, 0, 1],
                                                 geom_size=[0.01, 0.01, 0.01])

    object_list = [table, cue, ball, left_front_goal_post, right_front_goal_post,
                   left_back_goal_post, right_back_goal_post]
    scene = Scene(object_list)
    robot = Robots.MuJoCoRobot(scene, gravity_comp=True, num_DoF=7)

    duration = 2
    robot.ctrl_duration = duration
    robot.startLogging()

    home_position = robot.current_c_pos.copy()
    home_orientation = robot.current_c_quat.copy()

    # grab cue
    robot.gotoCartPositionAndQuat([0.55, 0.0, 0.5], [0, 1, 0, 0], duration=duration)
    robot.set_gripper_width = 0.04
    robot.gotoCartPositionAndQuat([0.55, 0.0, 0.42], [0, 1, 0, 0], duration=duration)
    robot.set_gripper_width = 0.0

    # shoot
    robot.gotoCartPositionAndQuat([0.5, 0.0, 0.43], [0, 1, 0, 0], duration=duration)
    robot.gotoCartPositionAndQuat([0.6, 0.0, 0.43], [0, 1, 0, 0], duration=0.165)
    robot.gotoCartPositionAndQuat(home_position, home_orientation, duration=duration)

    robot.stopLogging()


if __name__ == '__main__':
    run_demo()
