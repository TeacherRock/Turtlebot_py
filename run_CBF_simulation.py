import numpy as np
import time
import keyboard

from utils.save_Data import *

from utils.Turtlebot.Turtlebot import *


def main_Turtlebot():
    Sys = Dynamics_Sys[sys_type]
    Sys.use_CLF = True
    Sys.use_CBF = True
    Sys.use_input_bound = True
    Sys.use_slack = True
    state_1, state_2, state_3, t = Sys.initialize()
    states = np.concatenate((state_1.T, state_2.T, state_3.T), axis=1)
    print("states : ", states)
    sampT = Sys.params.sampT
    Bs = np.zeros((1, 1))
    dBs = np.zeros((1, 1))
    Vs = np.zeros((1, 1))
    dVs = np.zeros((1, 1))
    us = np.zeros((1, 2))
    errors = np.zeros((1, 2))

    ts = np.zeros((1, 1))

    ts = time.time()
    while Sys.check_flag():
        t_itr_start = time.time()
        ts = np.append(ts, t)

        Sys.set_ctrl_input(Sys)
        u_ref = Sys.u_ref(Sys.ctrl_input)
        qp_constraints = Sys.create_constraints(u_ref)
        try:
            u_modified = Sys.ctrl_qp(qp_constraints)
        except:
            print("Can not solve the QP")
            return states

        state = Sys.step(u_modified)
        states = np.concatenate((states, state.T), axis=0)

        t  = t + sampT
        print(f"This iteration spend {time.time()-t_itr_start} s")
        if keyboard.is_pressed("esc"):
            break

    return states

if __name__ == "__main__":
    ctrl_type = 'Turtlebot'
    sys_type = 'Turtlebot'

    Dynamics_Sys = {
        'Turtlebot' : Turtlebot(ctrl_type)
    }

    record = main_Turtlebot()
    save_state_to_txt(record, sys_type, sys_type+'_'+ctrl_type)
    sim_animation_Turtlebot(sys_type+'_'+ctrl_type)
