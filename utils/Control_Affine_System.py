import numpy as np
from numpy import sin, cos
import sympy as sp
import cvxopt as cvx
import math

import time

class QP_PARAMS():
    A = None
    b = None
    H = None
    q = None

class CBF_CLF_PARAMS():
    B  = None  #  CBF
    dB = None  # dCBF
    V  = None  #  CLF
    dV = None  # dCLF

    clf_rate = None  # lambda
    cbf_rate = None  # gamma

    weight_input = None  # H
    weight_slack = None  # p
    u_bound      = None  # control input bound

class Obstacle():
    center_x = None
    center_y = None
    center_z = None

    radius_x = None
    radius_y = None
    radius_z = None

    safe_dis = None

class Control_Input():
    def __init__(self, ctrl_type):
        self.ctrl_type = ctrl_type
        self.q = None
        self.dq = None
        self.ddq = None
        self.q_cmd = None
        self.dq_cmd = None
        self.ddq_cmd = None

        self.q_goal    = None
        self.P_goal    = None
        self.goal_error = None

        self.jacobian = None
        self.V_max = None
        self.A_max = None
        self.W_max = None

        self.dim = None

class Controller():
    def __init__(self):
        self.u_ref = {
            'None'      : self.No_Reference,
            'PD-like'   : self.PD_Like,
            'Velocity'  : self.Velocity,
            'SMC'       : self.SMC,
            'Turtlebot' : self.Turtlebot
        }

        self.set_ctrl_input = {
            'None'      : self.No_Reference_ctrl_input,
            'PD-like'   : self.PD_Like,
            'Velocity'  : self.Velocity_ctrl_input,
            'SMC'       : self.SMC,
            'Turtlebot' : self.Turtlebot_ctrl_input
        }

        # PD_like params
        self.Kp = None
        self.Kv = None

    def No_Reference_ctrl_input(self, obj):
        obj.ctrl_input.goal_error = obj.ctrl_input.P_goal - obj.forward_kinematics(obj.q)

    def No_Reference(self, ctrl_input):
        return np.zeros((ctrl_input.dim, 1))
    
    def Turtlebot_ctrl_input(self, obj):
        obj.ctrl_input.goal_error = obj.ctrl_input.P_goal[0:2] - np.array([obj.x, obj.y])
        obj.ctrl_input.angle_to_goal = math.atan((obj.ctrl_input.goal_error[1]) / (obj.ctrl_input.goal_error[0]))-obj.theta
    
    def Turtlebot(self, ctrl_input):
        w_ref = 0.1 if (ctrl_input.angle_to_goal)>0 else -0.1
        return np.array([[w_ref]])

    def PD_Like(self, ctrl_input):
        q  = ctrl_input.q
        dq = ctrl_input.dq
        q_ref = ctrl_input.q_cmd
        u_ref = np.dot(self.Kv, np.dot(self.Kp, (q_ref - q)) - dq)
        return u_ref
    
    def Velocity_ctrl_input(self, obj):
        obj.ctrl_input.goal_error = obj.ctrl_input.P_goal - obj.forward_kinematics(obj.q)
        obj.ctrl_input.jacobian  = obj.jacobian(obj.q)
        obj.ctrl_input.q   = obj.q
        obj.ctrl_input.dq  = obj.dq
        obj.ctrl_input.ddq = obj.ddq

    def Velocity(self, ctrl_input):
        V_ref  = np.clip(30 * (ctrl_input.goal_error), -ctrl_input.V_max, ctrl_input.V_max)
        dq_ref = np.dot(np.linalg.inv(ctrl_input.jacobian), V_ref)
        u_ref  = np.dot(self.Kv, dq_ref - ctrl_input.dq)
        return u_ref

    def SMC(self, ctrl_input):
        pass

class Control_System():
    def __init__(self, system_name):
        self.system_name = system_name
        self.create_func()
    
    def create_func(self):
        x, f, g = self.define_System()
        cbf, dcbf = self.define_CBF(x)
        clf, dclf = self.define_CLF(x)

        if f is not None and g is not None:
            self.write_func_to_txt('f',  f, '(self, x)', isArray=True)
            self.write_func_to_txt('g',  g, '(self, x)', isArray=True)
        self.load_function_from_txt('f')
        self.load_function_from_txt('g')

        if cbf is not None:
            self.write_func_to_txt('cbf',  cbf, '(self, x)')
            self.write_func_to_txt('dcbf', dcbf, '(self, x)', isArray=True)
        self.load_function_from_txt('cbf')
        self.load_function_from_txt('dcbf')

        if clf is not None:
            self.write_func_to_txt('clf',  clf, '(self, x)')
            self.write_func_to_txt('dclf', dclf, '(self, x)', isArray=True)
        self.load_function_from_txt('clf')
        self.load_function_from_txt('dclf')



    def load_function_from_txt(self, func_name):
        path = './utils/' + self.system_name + '/func_txt_file/'
        file_path = path + func_name + '.txt'
        with open(file_path, 'r') as file:
            function_code = file.read()
            try:
                exec(function_code, globals(), locals())
                setattr(self, func_name, locals()[func_name])
            except SyntaxError:
                print('Invalid function definition in the ' + func_name + '.txt file.')

    def write_func_to_txt(self, func_name, content, input_str, isArray = False):
        path = './utils/' + self.system_name + '/func_txt_file/'
        file_path = path + func_name + '.txt'

        return_value = str(content)
        start_index = return_value.find("[[")
        end_index   = return_value.rfind("]]")
        return_value = return_value[start_index+1 : end_index+1]
        # return_value = return_value[start_index : end_index+2]

        with open(file_path, 'w') as file:
            file.write('def ' + func_name + input_str + ':\n')
            if isArray:
                file.write('\treturn np.array([' + return_value + '])')
            else:
                file.write('\treturn ' + return_value)

    def create_constraints(self, u_ref):
        qp_params = QP_PARAMS()
        x = self.state.squeeze()
        f = self.f(self, x)
        g = self.g(self, x)

        # print("f : ", f)
        # print("g : ", g)

        qp_params.A = np.empty((0, self.u_dim))
        qp_params.b = np.empty((0, 1))

        constraint_num = 0

        if self.use_CLF:
            V    = np.array(self.clf(self, x)).reshape(-1, 1)
            dclf = self.dclf(self, x).reshape(V.shape[0], self.state_dim)
            LfV  = np.dot(dclf, f)
            LgV  = np.dot(dclf, g)
            qp_params.A = np.concatenate((qp_params.A, LgV), axis=0)
            qp_params.b = np.concatenate((qp_params.b, -LfV - self.cbf_clf_params.clf_rate * V), axis=0)
            constraint_num += V.shape[0]

        if self.use_CBF:
            B    = np.array(self.cbf(self, x)).reshape(-1, 1)
            dcbf = self.dcbf(self, x).reshape(B.shape[0], self.state_dim)
            LfB  = np.dot(dcbf, f)
            LgB  = np.dot(dcbf, g)
            qp_params.A = np.concatenate((qp_params.A, -LgB), axis=0)
            qp_params.b = np.concatenate((qp_params.b, LfB + self.cbf_clf_params.cbf_rate * B), axis=0)
            constraint_num += B.shape[0]

        if self.use_input_bound:
            qp_params.A = np.concatenate((qp_params.A, np.eye(self.u_dim), -np.eye(self.u_dim)), axis=0)
            qp_params.b = np.concatenate((qp_params.b, self.cbf_clf_params.u_bound*np.ones((2*self.u_dim, 1))), axis=0)
            constraint_num += 2

        # print("A_cvx : ", qp_params.A)
        # print("b_cvx : ", qp_params.b)
        # print("===================================================")

        if self.use_slack:
            temp_slack_constrint = np.concatenate(([[-1]], np.zeros((constraint_num-1, 1))), axis=0)
            qp_params.A = np.concatenate((qp_params.A, temp_slack_constrint), axis=1)
            qp_params.H =  2.0 * np.eye(self.u_dim+1)
            qp_params.H[-1, -1] = self.cbf_clf_params.weight_slack
            qp_params.q = -0.0 * np.dot(np.eye(self.u_dim+1), np.concatenate((u_ref, [[0]]), axis=0))
        else:
            qp_params.H =  2.0 * np.eye(self.u_dim)
            qp_params.q = -0.0 * np.dot(np.eye(self.u_dim), u_ref)
        
        # print("H_cvx : ", qp_params.H)
        # print("q_cvx : ", qp_params.q)        

        return qp_params

    def ctrl_qp(self, qp_params):
        A_cvx = cvx.matrix(qp_params.A)
        b_cvx = cvx.matrix(qp_params.b)
        H_cvx = cvx.matrix(qp_params.H)
        q_cvx = cvx.matrix(qp_params.q)

        cvx.solvers.options['show_progress'] = False
        solution = cvx.solvers.qp(P=H_cvx, q=q_cvx, G=A_cvx, h=b_cvx)
        u_optimal = np.array(solution['x'])

        return u_optimal
    
    def define_System(self):
        x = None
        f = None
        g = None
        return x, f, g
    
    def define_CBF(self, x):
        cbf = None
        dcbf = None
        return cbf, dcbf 
        
    def define_CLF(self, x):
        clf = None
        dclf = None
        return clf, dclf 

