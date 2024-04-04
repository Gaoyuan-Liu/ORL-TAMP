
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

from common.scn import Scenario

from examples.pybullet.utils.pybullet_tools.utils import connect, create_obj

RUN_TIMES = 1



def main():

    # pb.connect(pb.GUI)
    connect()

    # ---------------- 
    # Create the scene 
    # ----------------
    scn = Scenario()
    scn.reset()
    scn.add_robot(pose=((0,0,0.0), (0,0,0,1)))
    table_1 = scn.add_low_table()
    table_2 = scn.add_high_table(pose=((0,-10,0), (0,0,0,1)))
    scn.add_plate()    

    # ---------------- 
    # Reset 
    # ----------------
    pb.resetBasePositionAndOrientation(scn.plate, (0.6, 0.15, 0.19), (0,0,0,1))
    pb.setRealTimeSimulation(1)
    time.sleep(1)
    pb.setRealTimeSimulation(0)
    pb.changeDynamics(scn.plate, -1, lateralFriction=5)
    pb.changeDynamics(scn.plate, -1, angularDamping=0.01)
    input('\n Scene is ready. Press Enter to plan. ')

    # -----------------
    # Solve the problem
    # -----------------
    # plan_success = 0
    # execute_success = 0
    solver = Solver(scn.robot, movable=[scn.plate], tools=[], surfaces=[table_1, table_2], marks=[])
    executor = Executor(scn)


    #################################################


    goal = ('on', scn.plate, table_2)

    # plan, cost, evaluations = solver.solve(goal)
    solver.problem_formation()
    plan = pickle.load(open("./plan.pkl", "rb", -1))

    
    if plan is None:
        print('Planning failed')
        return
    else:
        # with open('./plan.pkl', 'wb') as file:
        #     pickle.dump(plan, file)
        pass
       
    #################################################


    input('\n Plan is ready. Press Enter to execute. ')
    time.sleep(2)
    executor.execute(solver.problem, plan)

        





if __name__ == '__main__':
    main()