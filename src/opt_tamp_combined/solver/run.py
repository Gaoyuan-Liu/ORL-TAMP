
import os, sys, time
import pybullet as pb
import pandas as pd
import pickle

from opt_solver import Solver
from execute import Executor

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
sys.path.insert(0, file_path + '/../../')
sys.path.insert(0, file_path + '/../../pddlstream/')

from pddlstream.language.constants import Equal, And

from common.scn import Scenario

from examples.pybullet.utils.pybullet_tools.utils import connect
import pybullet as p

RUN_TIMES = 1
def main():

    connect(use_gui=True)

    #################################################
    # Scenario 
    scn = Scenario()
    scn.reset()
    robot = scn.add_robot(pose=((0,0,0), (0,0,0,1)))
    table_low = scn.add_low_table()
    table_high = scn.add_high_table()
    dish = scn.add_dish(pose=((0.45, 0.15, 0.31),(0,0,0,1)))
    bar = scn.add_bar(((0.6, -0.1, 0.33), (0,0,0,1)))
    cube = scn.add_cup(((0.8, 0.1, 0.33), (0,0,0,1)))
    pb.changeDynamics(scn.dish, -1, lateralFriction=2)


    # Print
    print(' \n====== Scenario ====== ')
    print(' Robot: ', robot)
    print(' Low Table: ', table_low)
    print(' High Table: ', table_high)
    print(' Dish: ', dish)
    print(' Bar: ', bar)
    print(' Cup: ', cube)
    print(' ====================== \n')
    pb.setRealTimeSimulation(1)
    time.sleep(1)
    pb.setRealTimeSimulation(0)


    input('\n Scene is ready. Press Enter to plan. ')

    #################################################
    # Solve the problem
    solver = Solver(robot, movable=[dish, cube], tools=[bar], surfaces=[table_low, table_high, dish],
                    top_grasps=[cube], edge_grasps=[dish])
    executor = Executor(scn)

    #################################################
    # Define the goal
    goal = And(('on', scn.dish, table_high), ('on', cube, scn.dish))
    
    #################################################
    # Solve the problem
 
    plan, cost, evaluations = solver.solve(goal)

   
    
    if plan is None:
        print('Planning failed')
        return
    else:
        # with open('./plan_.pkl', 'wb') as file:
        #     pickle.dump(plan, file)

        input('\n Plan is ready. Press Enter to execute. ')
        executor.execute(solver.problem, plan)

def reset(scn):
    pass




if __name__ == '__main__':
    main()