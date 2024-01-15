
import os, sys, time
import pybullet as pb
import pandas as pd
import numpy as np
import math

from opt_solver import Solver
# from post_process import Executor

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
sys.path.insert(0, file_path + '/../../pddlstream/')
from pddlstream.language.constants import Equal, And

from scenario import Scenario

from examples.pybullet.utils.pybullet_tools.utils import connect
from policies import EdgePushPolicy


RUN_TIMES = 10
def main():

    
    sim_id = connect(use_gui=True) 
    scn = Scenario()
    scn.reset()
    scn.add_robot(pose=((0,0,0.2), (0,0,0,1)))
    table1 = scn.add_low_table(pose=((0.5,0,0), (0,0,0,1)))
    table2 = scn.add_high_table(pose=((0.5,-0.5,0), (0,0,0,1)))
    scn.add_dish(pose=((0.5, 0, 0.31),(0,0,0,1)))
    

    pb.setRealTimeSimulation(1)
    time.sleep(1)
    pb.setRealTimeSimulation(0)




    input('Press enter to continue: ')
    #################################################
    # Dataframes
    df_pt = pd.DataFrame(columns=['planning_time'])
    df_su = pd.DataFrame(columns=['plan_success', 'execute_success'])
    plan_success = 0
    execute_success = 0 


    #################################################
    # Planning
    solver = Solver(scn.panda, movable=[scn.dish], tools=[], surfaces=[table1, table2], marks=[])
    # executor = Executor(scn)
    goal = ('on', scn.dish, table2)
    edgepush = EdgePushPolicy(scn)
    
    
    #################################################
    for i in range(RUN_TIMES):
        print("\033[92m {}\033[00m" .format(f'\n {i} Episode'))

       
        pb.resetBasePositionAndOrientation(scn.dish, (0.5, 0, 0.31), (0,0,0,1))

        start = time.time()
        while True:
            success = True
            
            plan, cost, evaluations = solver.solve(goal)
            
            if plan is None:
                # RL
                # Here, we need to specify the sub-goal
                workspace = np.array([[0.25, 0.75], [-0.25, 0.25]])
                x_goal = np.random.uniform(low=workspace[0][0], high=workspace[0][1])
                y_goal = np.random.choice([-0.23, 0.23])
                # y_goal = 0.23
                z_goal = 0.31
                # p.resetBasePositionAndOrientation(goal_mark, (x_goal, y_goal, z_goal), (0,0,0,1))
                lg = np.array([x_goal, y_goal, z_goal])
                edgepush.specify(scn.dish, lg)
                edgepush.apply(None)
            else:
                break

            end = time.time()
            planning_time = end - start

            if planning_time > 100:
                success = False
                break
        
        if success:
            plan_success += 1
            execute_success += 1

        df_pt.loc[i] = [planning_time]    
        df_su.loc[i] = [plan_success, execute_success]

        df_pt.to_csv('./results/planning_time.csv', index=False)
        df_su.to_csv('./results/success.csv', index=False)





    #################################################
    # Execute the plan
    # if plan is None:
    #     exit()
    # print('Plan:', plan)    
    
    # executor.execute(solver.problem, plan, scn)


    # input('Press enter to continue: ')
def reset(scn):
    pb.resetBasePositionAndOrientation(scn.dish, (0.5, 0, 0.31), (0,0,0,1))

    pb.setRealTimeSimulation(1)
    time.sleep(1)
    pb.setRealTimeSimulation(0)    


if __name__ == '__main__':
    main()