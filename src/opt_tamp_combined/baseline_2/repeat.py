
import os, sys, time
import pybullet as pb
import pandas as pd

from solver import Solver
from post_process import Executor

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
sys.path.insert(0, file_path + '/../../utils/')
sys.path.insert(0, file_path + '/../../pddlstream/')

from pddlstream.language.constants import Equal, And

from scn import Scenario

from examples.pybullet.utils.pybullet_tools.utils import connect

RUN_TIMES = 50
def main():

    
    sim_id = connect(use_gui=True) 
    scn = Scenario()
    scn.reset()
    scn.add_robot(pose=((0,0,0.2), (0,0,0,1)))
    table1 = scn.add_low_long_table()
    table2 = scn.add_high_table(pose=((0.6,-0.5,0), (0,0,0,1)))
    scn.add_dish(pose=((0.5, 0, 0.31),(0,0,0,1)))
    scn.add_bar(((0.6, -0.1, 0.33), (0,0,0,1)))
    # scn.add_cup(((0.8, 0.1, 0.33), (0,0,0,1)))
    scn.add_cube(((0.8, 0.1, 0.33), (0,0,0,1)))

    pb.changeDynamics(scn.dish, -1, lateralFriction=1)


    pb.setRealTimeSimulation(1)
    time.sleep(1)
    pb.setRealTimeSimulation(0)




    input('Press enter to continue: ')
    #################################################
    plan_success = 0
    execute_success = 0

    #################################################
    # Solve the problem
    
    solver = Solver(scn.panda, movable=[scn.dish, scn.cube], tools=[scn.hook], surfaces=[table1, table2, scn.dish],
                    top_grasps=[scn.cube], edge_grasps=[scn.dish])
    executor = Executor()
    
    df_su = pd.DataFrame(columns=['plan_success', 'execute_success'])
    df_pt = pd.DataFrame(columns=['planning_time'])

    goal = And(('on', scn.dish, table2), ('on', scn.cube, scn.dish))

    for i in range(RUN_TIMES):

        print(f' \nEpisode {i} \n')
        pb.resetBasePositionAndOrientation(scn.dish, (0.5, 0, 0.31), (0,0,0,1))
        start = time.time()
        plan, cost, evaluations = solver.solve(goal)
        end = time.time()
        planning_time = end - start
        df_pt.loc[i] = [planning_time]
        #################################################
        # Execute the plan
        if plan is None:
            print('Planning failed')
            # continue
        else:
            plan_success += 1
            

        #################################################
        # try:
            executor.execute(solver.problem, plan, scn)
            execute_success += 1
        # except:
            # print('Execution failed')
            
        
    
        df_su.loc[0] = [plan_success, execute_success]
        df_su.to_csv('./results/success.csv', index=False)
        df_pt.to_csv('./results/planning_time.csv', index=False)

        # input('Press enter to continue: ')



if __name__ == '__main__':
    main()