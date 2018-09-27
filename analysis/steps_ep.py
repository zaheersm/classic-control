import os
import numpy as np


from sweeper import Sweeper
from visualizer import RunLines


def parse_steps_log(log_path, max_steps):
    """
    Retrieves num_steps at which episodes were completed
    and returns an array where the value at index i represents
    the number of episodes completed until step i
    """
    with open(log_path, "r") as f:
        lines = f.readlines()
    eps = 0
    prev_num_steps = 0
    eps_over_time = np.zeros(max_steps)
    for line in lines:
        if 'Episode' not in line:
            continue
        num_steps = int(line.split("|")[1].split(":")[1])
        if num_steps > max_steps:
            break
        eps = int(line.split("|")[2].split(":")[1])
        eps_over_time[prev_num_steps:num_steps] = eps
        eps += 1
        prev_num_steps = num_steps
    eps_over_time[prev_num_steps:] = eps
    return eps_over_time


def parse_ep_log(log_path, max_steps):
    """
    Retrieves num_steps at which episodes were completed
    and returns an array where the value at index i represents
    the number of episodes completed until step i
    """
    with open(log_path, "r") as f:
        lines = f.readlines()
    eps = 0
    prev_num_steps = 0
    eps_over_time = np.zeros(max_steps)
    for line in lines:
        if 'Episode' not in line:
            continue
        num_steps = int(line.split("|")[2].split(":")[1])
        eps_over_time[prev_num_steps:num_steps] = eps
        eps += 1
        prev_num_steps = num_steps
    eps_over_time[prev_num_steps:] = eps
    return eps_over_time


if __name__ == '__main__':

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    log_name = "ep_log"
    parser_func = parse_ep_log if log_name == "ep_log" else parse_steps_log
    path_formatters = []

    config_files = [("config_files/sw_actor_critic.json", 13)]

    for cf, best_setting in config_files:
        swp = Sweeper(os.path.join(project_root, cf))
        cfg = swp.parse(best_setting)      # Creating a cfg with an arbitrary run id
        logdir_format = cfg.get_logdir_format()
        path_format = os.path.join(logdir_format, log_name
                                   )
        path_formatters.append(path_format)

    runs = [10]
    num_datapoints= [40000]
    labels = ['Actor-Critic']
    v = RunLines(path_formatters, runs, num_datapoints,
                 labels, parser_func=parser_func,
                 save_path="mountain_car.png", xlabel="Number of steps", ylabel="Episodes Completed")
    v.draw()




