import numpy as np
from tracikpy import TracIKSolver

ee_pose = np.array([[ 0.0525767 , -0.64690764, -0.7607537 , 0.        ],
                    [-0.90099786, -0.35923817,  0.24320937, 0.2       ],
                    [-0.43062577,  0.67265031, -0.60174996, 0.4       ],
                    [ 0.        ,  0.        ,  0.        , 1.        ]])

ik_solver = TracIKSolver(
    "./../../common/models/franka_panda/panda_modified.urdf",
    "panda_link0",
    "panda_hand",
)
qout = ik_solver.ik(ee_pose, qinit=np.zeros(ik_solver.number_of_joints))
print(qout)