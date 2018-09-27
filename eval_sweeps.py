import os
import numpy as np

from sweeper import Sweeper


if __name__ == '__main__':

    start_idx = 0
    end_idx = 640
    config_file = 'config_files/sw_actor_critic.json'
    max_steps = 40000

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    sweeper = Sweeper(os.path.join(project_root, config_file))
    eval = []
    for k in range(sweeper.total_combinations):
        eval.append([])

    for idx in range(start_idx, end_idx):
        cfg = sweeper.parse(idx)
        log_dir = cfg.get_logdir()
        log_path = os.path.join(log_dir, 'steps_log')
        with open(log_path, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            continue
        # ugly parse based on the log_file format
        num_steps = int(lines[-1].split("|")[1].split(":")[1])
        episodes = int(lines[-1].split("|")[2].split(":")[1])
        if cfg.max_steps == num_steps:
            assert idx % sweeper.total_combinations == cfg.param_setting
            eval[idx % sweeper.total_combinations].append(episodes)

    summary = list(map(lambda x: (x[0], np.mean(x[1]), np.std(x[1]), len(x[1])), enumerate(eval)))
    summary = sorted(summary, key=lambda s: s[1], reverse=True)

    for idx, mean, std, num_runs in summary:
        print("Param Setting # {:>3d} | Average num of episodes: {:>10.2f} +/- {:>5.2f} ({:>2d} runs) {} | ".format(
            idx, mean, std, num_runs, sweeper.param_setting_from_id(idx)))
