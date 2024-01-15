
import os, sys, time
import pybullet as pb
import pandas as pd
import numpy as np
import math

from solver import Solver
from post_process import Executor

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
sys.path.insert(0, file_path + '/../../pddlstream/')
from pddlstream.language.constants import Equal, And

from scenario import Scenario

from examples.pybullet.utils.pybullet_tools.utils import connect

RUN_TIMES = 50
def main():

    
    sim_id = connect(use_gui=True) 
    scn = Scenario()
    scn.reset()
    scn.add_robot() # 1
    # cube1 = scenario.add_cube(((0.9, -0.3, 0.025), (0,0,0,1)))
    # cube2 = scenario.add_cube(((0.9, 0.3, 0.025), (0,0,0,1)))
    cup = scn.add_cup(((0.9, 0.3, 0.035), (0,0,0,1))) # 2
    # scenario.add_dish(((0.9, -0.3, 0.025), (0,0,0,1)))
    # scenario.add_table_low(((0.5, -0.5, 0.31), (0,0,0,1)))

    scn.add_hook()

    scn.add_goal_surface(((-0.4, 0.0, 0.01), (0,0,0,1)))

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
    # Solve the problem
    solver = Solver(scn.panda, movable=[scn.cup, scn.hook], tools=[scn.hook], surfaces=[scn.goal_surface], marks=[])
    goal = ('on', scn.cup, scn.goal_surface)
    executor = Executor()
    # solver = Solver(scenario.panda, movable=[cube1, cube2, scenario.hook], tools=[scenario.hook], surfaces=[scenario.goal_surface, scenario.hook_mark], marks=[scenario.hook_mark])
    # goal = And(('on', cube1, scenario.goal_surface), ('on', cube2, scenario.goal_surface)) #, ('on', cube2, scenario.goal_surface)
    
    
    for i in range(RUN_TIMES):
        print("\033[92m {}\033[00m" .format(f'\n {i} Episode'))
        reset(scn)
        start = time.time()
        plan, cost, evaluations = solver.solve(goal)
        end = time.time()
        planning_time = end - start
        df_pt.loc[i] = [planning_time]
        print('Plan:', plan)
        if plan is None:
            continue
        else:
            plan_success += 1


    #################################################
    # Execute the plan
        try: 
            executor.execute(solver.problem, plan, scn)
            execute_success += 1
        except:
            print('Execution failed')

    
        

    #################################################

        # Record
        
        df_su.loc[i] = [plan_success, execute_success]
        df_pt.to_csv('./results/planning_time.csv', index=False)
        df_su.to_csv('./results/success.csv', index=False)


#################################################
def reset(scn):
    # Cup
    r = np.random.uniform(low=0.8, high=1.)
    theta = np.random.uniform(low=-math.pi/4, high=math.pi/4)
    x = r*math.cos(theta)
    y = r*math.sin(theta)
    z = 0.025 
    orn = [0,0,0,1]
    pb.resetBasePositionAndOrientation(scn.cup, [x, y, z], orn)
    # pb.resetBasePositionAndOrientation(scn.cup, (1.0, -0.1, 0.035), (0,0,0,1))

    # Hook
    pb.resetBasePositionAndOrientation(scn.hook, (0.55,0,0.1), (0,0,0,1))

    pb.setRealTimeSimulation(1)
    time.sleep(1)
    pb.setRealTimeSimulation(0)

if __name__ == '__main__':
    main()