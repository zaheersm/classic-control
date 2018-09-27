import argparse
import os
import time

from environments.mountain_car import MountainCar
import agents.agent
from utils.experiment import Experiment
from utils.helper_functions import ensure_dirs, setup_logger, memory_usage_psutil
from sweeper import Sweeper


def run():
    start = time.time()

    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--idx', default=0, type=int, help='identifies run number and configuration')
    parser.add_argument('--config-file', default='config_files/actor_critic.json')

    args = parser.parse_args()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    sweeper = Sweeper(os.path.join(project_root, args.config_file))
    cfg = sweeper.parse(args.idx)

    cfg.env_instance = MountainCar(cfg)
    agent_class = getattr(agents.agent, cfg.agent_class)
    agent = agent_class(cfg)

    log_dir = cfg.get_logdir()
    ensure_dirs([log_dir])
    steps_log = os.path.join(log_dir, 'steps_log')
    steps_logger = setup_logger(steps_log, stdout=True)
    cfg.log_config(steps_logger)
    ep_log = os.path.join(log_dir, 'ep_log')
    ep_logger = setup_logger(ep_log, stdout=False)
    cfg.log_config(ep_logger)

    exp = Experiment(agent, cfg.env_instance, max_steps=cfg.max_steps, seed=args.idx,
                     steps_log=steps_log, ep_log=ep_log)
    exp.run_step_mode()

    print("Memory used: {:5} MB".format(memory_usage_psutil()))
    print("Time elapsed: {:5.2} minutes".format((time.time() - start) / 60))


if __name__ == '__main__':
    run()
