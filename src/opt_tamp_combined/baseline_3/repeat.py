
import os, sys, time
import pybullet as pb
import pandas as pd

from opt_solver import Solver
from post_process import Executor

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
sys.path.insert(0, file_path + '/../../utils/')
sys.path.insert(0, file_path + '/../../pddlstream/')

from pddlstream.language.constants import Equal, And

from scn import Scenario

from examples.pybullet.utils.pybullet_tools.utils import connect
import pybullet as p
import numpy as np

from policies_push import EdgePushPolicy
from policies_hook import HookingPolicy

RUN_TIMES = 10
def main():

    
    sim_id = connect(use_gui=True)

    #################################################
    # Scenario 
    scn = Scenario()
    scn.reset()
    scn.add_robot(pose=((0,0,0.2), (0,0,0,1)))
    table1 = scn.add_low_long_table()
    table2 = scn.add_high_table(pose=((0.6,-0.5,0), (0,0,0,1)))
    scn.add_dish(pose=((0.5, 0, 0.31),(0,0,0,1)))
    scn.add_bar(((0.6, -0.1, 0.33), (0,0,0,1)))
    # scn.add_cup(((0.8, 0.1, 0.33), (0,0,0,1)))
    scn.add_cube(((0.8, 0.1, 0.33), (0,0,0,1)))
    p.changeDynamics(scn.dish, -1, lateralFriction=1)

    # Print
    print(' \n====== Scenario ====== ')
    print(' Robot: ', scn.panda)
    print(' Low Table: ', table1)
    print(' High Table: ', table2)
    print(' Dish: ', scn.dish)
    print(' Bar: ', scn.hook)
    print(' Cup: ', scn.cube)
    print(' ====================== \n')
    pb.setRealTimeSimulation(1)
    time.sleep(1)
    pb.setRealTimeSimulation(0)


    input('Press enter to continue: ')


    #################################################
    # Solve the problem
    solver = Solver(scn.panda, movable=[scn.dish, scn.cube], tools=[scn.hook], surfaces=[table1, table2, scn.dish],
                    top_grasps=[scn.cube], edge_grasps=[scn.dish])
    executor = Executor(scn)

    #################################################
    # RL helper
    edgepush = EdgePushPolicy(scn)
    hook = HookingPolicy(scn)


    #################################################
    # Define the goal
    goal = And(('on', scn.dish, table2), ('on', scn.cube, scn.dish))
    # goal = And(('on', scn.dish, table2), ('on', scn.cube, table2))
    # goal = ('on', scn.cube, scn.dish)
    # goal = ('on', scn.dish, table2)


    #################################################
    # Solve the problem
    plan_success = 0
    execute_success = 0
    df_pt = pd.DataFrame(columns=['planning_time'])
    df_su = pd.DataFrame(columns=['plan_success', 'execute_success'])
    
    for i in range(RUN_TIMES):

        print(f' \nEpisode {i} \n')
        pb.resetBasePositionAndOrientation(scn.dish, (0.5, 0, 0.31), (0,0,0,1))
        pb.resetBasePositionAndOrientation(scn.cube, (0.8, 0, 0.33), (0,0,0,1))
        
        start = time.time()

        while True:

            success = True

            plan, cost, evaluations = solver.solve(goal)
            if plan is None:
                # Edgepush
                workspace = np.array([[0.25, 0.75], [-0.25, 0.25]])
                x_goal = np.random.uniform(low=workspace[0][0], high=workspace[0][1])
                # y_goal = np.random.choice([-0.23, 0.23])
                y_goal = 0.23
                z_goal = 0.31
                # p.resetBasePositionAndOrientation(goal_mark, (x_goal, y_goal, z_goal), (0,0,0,1))
                lg = np.array([x_goal, y_goal, z_goal])
                edgepush.specify(scn.dish, lg)
                edgepush.apply(None)

                # Hooking
                hook.specify(scn.cube, None)
                hook.apply(None)

            else:
                planning_time = end - start
                break

            end = time.time()
            planning_time = end - start

            if planning_time > 300:
                success = False
                break

        if success:
            plan_success += 1
            execute_success += 1

        #################################################

        df_pt.loc[i] = [planning_time]    
        df_su.loc[i] = [plan_success, execute_success]

        df_pt.to_csv('./results/planning_time.csv', index=False)
        df_su.to_csv('./results/success.csv', index=False)




if __name__ == '__main__':
    main()