
import os, sys, time
import pybullet as pb
import pandas as pd
import numpy as np
import math

from opt_solver import Solver
from execute import Executor

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
sys.path.insert(0, file_path + '/../../pddlstream/')
# from pddlstream.language.constants import Equal, And

from common.scn import Scenario 

from examples.pybullet.utils.pybullet_tools.utils import connect

RUN_TIMES = 1
def main():

    connect(use_gui=True) 

    # ---------------- 
    # Create the scene 
    # ----------------
    scn = Scenario()
    scn.reset()
    scn.add_robot()
    cup_1 = scn.add_cup(((0.5, 0, 0.035), (0,0,0,1)))
    scn.add_bar()
    scn.add_surface(((-0.4, 0, 0.01), (0,0,0,1)))
    pb.setRealTimeSimulation(1)
    time.sleep(1)
    pb.setRealTimeSimulation(0)

    # ---------------- 
    # Reset 
    # ----------------
    reset(scn)
    input('\n Scene is ready. Press Enter to plan. ')


    # -----------------
    # Solve the problem\]
    # -----------------
    # Dataframes
    # df_pt = pd.DataFrame(columns=['planning_time'])
    # df_su = pd.DataFrame(columns=['plan_success', 'execute_success'])
    # plan_success = 0
    # execute_success = 0

    #################################################
    # Solve the problem
    solver = Solver(scn.panda, movable=[scn.cup, scn.bar], tools=[scn.bar], surfaces=[scn.surface], marks=[])
    goal = ('on', scn.cup, scn.surface)
    executor = Executor(scn)
    # print("\033[92m {}\033[00m" .format(f'\n {i} Episode'))

    plan, cost, evaluations = solver.solve(goal) 

    if plan is None:
        print("\033[32m {}\033[00m" .format(' Plan failed'))
        return None


    input('\n Plan is ready. Press Enter to execute. ')
    executor.execute(solver.problem, plan)

   


#################################################

def reset(scn):
    # Cup
    r = np.random.uniform(low=0.8, high=1.)
    r = 0.5
    theta = np.random.uniform(low=-math.pi/4, high=math.pi/4)
    x = r*math.cos(theta)
    y = r*math.sin(theta)
    z = 0.1
    orn = [0,0,0,1]
    # pb.resetBasePositionAndOrientation(scn.cup, [x, y, z], orn)
    pb.resetBasePositionAndOrientation(scn.cup, [0.6, -0.1, 0.4], orn)

    # Bar
    pb.resetBasePositionAndOrientation(scn.bar, (1.55,0,0.1), (0,0,0,1))

    # pb.setRealTimeSimulation(1)
    # time.sleep(1)
    # pb.setRealTimeSimulation(0)

    


if __name__ == '__main__':
    main()