from utils.Control_Affine_System import *
import numpy as np
import math
import time
from math import pi as pi

class Turtlebot_Params():
    sampT = 0.02
    robot_radius = 0.09
    v = 0.05
    V_max = 1
    W_max = 1

class Turtlebot(Control_System):
    def __init__(self, ctrl_type = 'Turtlebot'):
        super().__init__('Turtlebot')
        self.params = Turtlebot_Params()
        self.state_dim = 3
        self.u_dim = 1
        self.cbf_clf_params = CBF_CLF_PARAMS()
        self.cbf_clf_params.clf_rate = 10
        self.cbf_clf_params.cbf_rate = 1
        self.cbf_clf_params.weight_input = 1
        self.cbf_clf_params.weight_slack = 1e+3
        self.cbf_clf_params.u_bound = self.params.W_max

        self.ctrl_input = Control_Input(ctrl_type)
        self.ctrl_input.dim = 1
        self.ctrl_input.V_max = self.params.V_max
        self.ctrl_input.W_max = self.params.W_max

        controller = Controller()
        self.u_ref = controller.u_ref[ctrl_type]
        self.set_ctrl_input = controller.set_ctrl_input[ctrl_type]

    def initialize(self):
        self.params.P_ini = np.array([0.0,  0.0, 0.0]).reshape(-1, 1)
        self.ctrl_input.P_goal  = np.array([-1.2,  1.2, pi/2]).reshape(-1, 1)
        self.x = self.params.P_ini[0]
        self.y = self.params.P_ini[1]
        self.theta = self.params.P_ini[2]
        self.state = np.array((self.x, self.y, self.theta))
        self.ctrl_input.goal_error = self.ctrl_input.P_goal[0:2] -  self.params.P_ini[0:2]
        t  = 0

        self.ob = Obstacle()
        self.ob.center_x = -0.38
        self.ob.center_y = 0.2625
        self.ob.safe_dis = 0.105

        self.ob.center_x2 = -0.725
        self.ob.center_y2 =  0.76
        self.ob.safe_dis2 = 0.185

        self.ob.center_x3 = -0.22
        self.ob.center_y3 = 0.935
        self.ob.safe_dis3 = 0.105

        return np.array([self.x]), np.array([self.y]), np.array([self.theta]), t
    
    def step(self, u):
        self.theta = self.theta + u[0] * self.params.sampT
        self.y = self.y + self.params.v * math.sin(self.theta) * self.params.sampT
        self.x = self.x + self.params.v * math.cos(self.theta) * self.params.sampT
        self.state = np.array((self.x, self.y, self.theta))
        return np.array([self.x, self.y, self.theta])
    
    def check_flag(self):
        flag = False
        print(f"Currnet distance to Goal {np.linalg.norm(self.ctrl_input.goal_error)} (m)")
        # flag = True if (np.linalg.norm(self.ctrl_input.goal_error) > 0.02) else False
        flag = True if (np.linalg.norm(self.ctrl_input.goal_error) > 0.005) else False
        # flag = True
        return flag

    def define_System(self):
        p_x, p_y, theta, v = sp.symbols('x[0] x[1] x[2] self.params.v')
        x = sp.Matrix([p_x, p_y, theta])

        f = sp.Matrix([[v*sp.cos(theta)], [v*sp.sin(theta)], [0]])
        g = sp.Matrix([[0], [0], [1]])
        return x, f, g
    
    def define_CBF(self, state):
        p_x = state[0]
        p_y = state[1]
        theta = state[2]
        v = sp.symbols('self.params.v')
        ob_x, ob_y = sp.symbols('self.ob.center_x self.ob.center_y')
        ob_x2, ob_y2 = sp.symbols('self.ob.center_x2 self.ob.center_y2')
        ob_x3, ob_y3 = sp.symbols('self.ob.center_x3 self.ob.center_y3')
        safe_dis, r_robot = sp.symbols('self.ob.safe_dis self.params.robot_radius')
        safe_dis2, safe_dis3 = sp.symbols('self.ob.safe_dis2 self.ob.safe_dis3')

        cbf = sp.Matrix([[(p_x - ob_x)**2 + (p_y - ob_y)**2 - (safe_dis + r_robot)**2 
                          + (2*v*sp.cos(theta)*(p_x - ob_x) + 2*v*sp.sin(theta)*(p_y - ob_y))],
                          [(p_x - ob_x2)**2 + (p_y - ob_y2)**2 - (safe_dis2 + r_robot)**2 
                          + (2*v*sp.cos(theta)*(p_x - ob_x2) + 2*v*sp.sin(theta)*(p_y - ob_y2))],
                          [(p_x - ob_x3)**2 + (p_y - ob_y3)**2 - (safe_dis3 + r_robot)**2 
                          + (2*v*sp.cos(theta)*(p_x - ob_x3) + 2*v*sp.sin(theta)*(p_y - ob_y3))]
                          ])
        dcbf = cbf.jacobian(state)     
        return cbf, dcbf
        
    def define_CLF(self, state):
        p_x = state[0]
        p_y = state[1]
        theta = state[2]
        v = sp.symbols('self.params.v')
        goal_x, goal_y = sp.symbols('self.ctrl_input.P_goal[0] self.ctrl_input.P_goal[1]')
        
        # clf = sp.Matrix([(-sp.sin(theta)*(p_x - goal_x) + sp.cos(theta)*(p_y - goal_y))**2])
        clf = sp.Matrix([ (p_x + v*sp.cos(theta)*0.02 - goal_x)**2 
                        + (p_y + v*sp.sin(theta)*0.02 - goal_y)**2])
        dclf = clf.jacobian(state)
        return clf, dclf
