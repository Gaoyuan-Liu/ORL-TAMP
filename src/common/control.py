import pybullet as p

pandaNumDofs = 7
PANDA_ARM_JOINTS_ID = range(pandaNumDofs)
PANDA_FINGERS__ID = [9, 10]
pandaEndEffectorIndex = 11


class Control:
    def __init__(self, bullet_client=p) -> None:
        self.bullet_client = bullet_client
  
        # IK constraints
        self.ll = [-7]*pandaNumDofs
        # Upper limits for null space (todo: set them to proper range)
        self.ul = [7]*pandaNumDofs
        # Joint ranges for null space (todo: set them to proper range)
        self.jr = [7]*pandaNumDofs
        # Restposes for null space
        self.jointResetPositions=[0.0, 0.0, 0.0, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
        self.rp = self.jointResetPositions


    def go_joint_space(self, robot, jointPoses):
        for i in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(robot, i, self.bullet_client.POSITION_CONTROL, jointPoses[i],force=5 * 240.)


    def go_cartesian_space(self, robot, eePose):
        # IK
        jointPoses = self.bullet_client.calculateInverseKinematics(robot, pandaEndEffectorIndex, eePose[0], eePose[1], self.ll, self.ul,
        self.jr, self.rp, maxNumIterations=20)
        # Go
        self.go_joint_space(jointPoses)


    def finger_close(self, robot):
        finger_target = 0.00
        for i in [9,10]:
            self.bullet_client.setJointMotorControl2(robot, i, self.bullet_client.POSITION_CONTROL,finger_target ,force= 10)


    def finger_open(self, robot):
        finger_target = 0.04
        for i in [9,10]:
            self.bullet_client.setJointMotorControl2(robot, i, self.bullet_client.POSITION_CONTROL,finger_target ,force= 10)

    # def go_cartesian_linear(self, pose):



        
        