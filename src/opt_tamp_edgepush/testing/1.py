import pybullet as pb
import pybullet_data as pd
import math
import time
import numpy as np

id_1 = pb.connect(pb.GUI)
pb.disconnect(id_1)

id_2 = pb.connect(pb.DIRECT)

print(f'\nid_1 = {id_1}, id_2 = {id_2}')
