from control import Control
from scenario import Scenario
from utils.pybullet_tools.utils import connect, wait_for_user
import pybullet as p
import math

if __name__ == '__main__':
    connect(use_gui=True)

    scenario = Scenario()

    scenario.add_robot()
    scenario.add_hook(((0.4,0,0),(0,0,0,1)))
    scenario.add_cube(((0.6,0,0),(0,0,0,1)))

    p.setRealTimeSimulation(1)
    wait_for_user()

    control = Control(robot=scenario.panda)
    

    control.finger_open()
    wait_for_user()

    orn = p.getQuaternionFromEuler([math.pi,0.,0.])
    control.go_cartesian_space(eePose=((0.5, 0.0, 0.04), orn))
    wait_for_user()

    control.finger_close()
    wait_for_user()

    control.go_cartesian_space(eePose=((0.5, 0.0, 0.06), orn))
    wait_for_user() 

    control.go_cartesian_space(eePose=((0.5, 0.0, 0.07), orn))
    wait_for_user() 

    control.go_cartesian_space(eePose=((0.4, 0.0, 0.07), orn))
    wait_for_user() 


