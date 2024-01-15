
import os, sys, time
import pybullet as pb

from opt_solver import Solver
from post_process import Executor

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
sys.path.insert(0, file_path + '/../../pddlstream/')
from pddlstream.language.constants import Equal, And

from scenario import Scenario

from examples.pybullet.utils.pybullet_tools.utils import connect

def main():

    
    sim_id = connect(use_gui=True) 
    scenario = Scenario()
    scenario.reset()
    scenario.add_robot()
    # cube1 = scenario.add_cube(((0.9, -0.3, 0.025), (0,0,0,1)))
    # cube2 = scenario.add_cube(((0.9, 0.3, 0.025), (0,0,0,1)))
    cup = scenario.add_cup(((0.9, -0.3, 0.035), (0,0,0,1)))
    # scenario.add_dish(((0.9, -0.3, 0.025), (0,0,0,1)))
    # scenario.add_table_low(((0.5, -0.5, 0.31), (0,0,0,1)))

    scenario.add_hook()
    scenario.add_hook_mark()
    scenario.add_goal_surface(((0.0, 0.4, 0.01), (0,0,0,1)))

    pb.setRealTimeSimulation(1)
    time.sleep(1)
    pb.setRealTimeSimulation(0)




    input('Press enter to continue: ')

    #################################################
    # Planning
    solver = Solver(scenario.panda, movable=[scenario.cup, scenario.hook], tools=[scenario.hook], surfaces=[scenario.goal_surface, scenario.hook_mark], marks=[scenario.hook_mark])
    executor = Executor(scenario.panda, scenario)
    # solver = Solver(scenario.panda, movable=[cube1, cube2, scenario.hook], tools=[scenario.hook], surfaces=[scenario.goal_surface, scenario.hook_mark], marks=[scenario.hook_mark])
    # goal = And(('on', cube1, scenario.goal_surface), ('on', cube2, scenario.goal_surface)) #, ('on', cube2, scenario.goal_surface)
    goal = ('on', scenario.cup, scenario.goal_surface)
    # goal = ('on', scenario.hook, scenario.hook_mark)
    from policies_sb3 import HookingPolicy


    while True:
        plan, cost, evaluations = solver.solve(goal)

        if plan is None:
            # RL
            hook = HookingPolicy(scenario)
            hook.specify(scenario.cup, None)
            hook.apply(None)
        else:
            break



    #################################################
    # Execute the plan
    if plan is None:
        exit()
    print('Plan:', plan)    
    
    executor.execute(solver.problem, plan, scenario)


    input('Press enter to continue: ')


if __name__ == '__main__':
    main()