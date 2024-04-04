
import os, sys, time
import pybullet as pb

from opt_solver import Solver
from post_process import Executor

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
sys.path.insert(0, file_path + '/../../pddlstream/')

from scenario import Scenario

from examples.pybullet.utils.pybullet_tools.utils import connect
import pybullet as p

RUN_TIMES = 50
def main():

    
    sim_id = connect(use_gui=True) 
    scenario = Scenario()
    scenario.reset()
    scenario.add_robot(pose=((0,0,0.2), (0,0,0,1)))
    table1 = scenario.add_low_table(pose=((0.5,0,0), (0,0,0,1)))
    table2 = scenario.add_high_table(pose=((0.6,-0.5,0), (0,0,0,1)))

    print(f'table1 = {table1}')
    print(f'table2 = {table2}')
    
    scenario.add_dish(pose=((0.5, 0, 0.31),(0,0,0,1)))
    p.changeDynamics(scenario.dish, -1, lateralFriction=1)


    pb.setRealTimeSimulation(1)
    time.sleep(1)
    pb.setRealTimeSimulation(0)


    input('Press enter to continue: ')

    #################################################
    #################################################
    plan_success = 0
    execute_success = 0

    #################################################
    # Solve the problem
    solver = Solver(scenario.panda, movable=[scenario.dish], tools=[], surfaces=[table1, table2], marks=[])
    executor = Executor()


    #################################################
    # Solve the problem

    solver = Solver(scenario.panda, movable=[scenario.dish], tools=[], surfaces=[table1, table2], marks=[])
    # goal = ('on', scenario.cube, scenario.goal_surface)
    goal = ('on', scenario.dish, table2)
    plan, cost, evaluations = solver.solve(goal)

    #################################################
    # Execute the plan
    if plan is None:
        exit()

    # print(plan)
    
    executor.execute(solver.problem, plan, scenario)


    input('Press enter to continue: ')


if __name__ == '__main__':
    main()