import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from PIL import Image

def sim_animation_Turtlebot(file_name):
    data = np.loadtxt('./data/output/' + 'Turtlebot' + '/record_' + file_name + '.txt')

    x = data[:, 0]
    y = data[:, 1]

    t = np.arange(0.001, len(x) * 0.001 + 0.001, 0.001)

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.axis('equal')

    obstacle_circle = Circle([-0.38, 0.2625], 0.08, fill=False, edgecolor='b')
    safe_circle = Circle([-0.38, 0.2625], 0.105+0.09, fill=False, edgecolor='r', linestyle='dotted')
    obstacle_circle2 = Circle([-0.725, 0.76], 0.16, fill=False, edgecolor='b')
    safe_circle2 = Circle([-0.725, 0.76], 0.1625+0.09, fill=False, edgecolor='r', linestyle='dotted')
    obstacle_circle3 = Circle([-0.22, 0.935], 0.08, fill=False, edgecolor='b')
    safe_circle3 = Circle([-0.22, 0.935], 0.105+0.09, fill=False, edgecolor='r', linestyle='dotted')
    ax.add_patch(obstacle_circle)
    ax.add_patch(safe_circle)
    ax.add_patch(obstacle_circle2)
    ax.add_patch(safe_circle2)
    ax.add_patch(obstacle_circle3)
    ax.add_patch(safe_circle3)

    point, = ax.plot([], [], 'ro', label='Turtlebot Position')
    line, = ax.plot([], [], '-', label='Turtlebot Path', linewidth=1)

    # Set the view limits
    view_limits_x = (-1.5,  0.3)
    view_limits_y = ( -0.3, 1.5)
    ax.set_xlim(view_limits_x)
    ax.set_ylim(view_limits_y)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()

    img_step = 10

    def update(frame):
        i = frame * img_step
        line.set_data(x[:i+1], y[:i+1])
        point.set_data([x[i]], [y[i]])

        if frame == 0:
            ax.plot(0.0,  0.0, 'go', label='Initial Position')
            ax.plot(-1.2,  1.2, 'bo', label='Goal Position')
            origin_path_x = np.linspace(0.0, -1.2, 1000)
            origin_path_y = np.linspace(0.0,  1.2, 1000)
            ax.plot(origin_path_x,  origin_path_y, 'r-', label='Origin Path')

    num_frames = len(t) // img_step

    ani = FuncAnimation(fig, update, frames=num_frames, blit=False)

    ani.save('./data/output/' + 'Turtlebot/'  + file_name + '_animation.gif', writer='pillow', fps=1000)

def save_state_to_txt(xs, system, file_name):
    with open('./data/output/' + system + '/record_' + file_name + '.txt', 'w') as file:
        for i in range(xs[:, 1].shape[0]):
            line = " ".join(map(str, xs[i, :]))
            file.write(line + '\n')