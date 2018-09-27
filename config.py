import os
from utils.helper_functions import ensure_dirs, setup_logger


class Config(object):
    def __init__(self):
        # Experiment parameters
        self.exp_name = "John-Doe"
        self.run = 0
        self.param_setting = 0
        self.max_steps = 1000000

        # Problem parameters
        self.env_name = "MountainCar-v0"
        self.action_noise = 0.0
        self.env_instance = None
        self.gamma = 1.0

        self.representation = "tile_code"
        # Tile-coding parameters
        self.tiles = 8  # Each tiling consists of 8 x 8 grid
        self.num_tilings = 8
        self.mem_size = 4096
        self.tile_combinations = False

        # Setting up data, log & checkpoint paths
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        self.data_root = os.path.join(project_root, 'data', 'output')
        ensure_dirs([self.data_root])
        self.out_path = None

    def get_logdir_format(self):
        return os.path.join(self.data_root, self.exp_name, "{}_run", "{}_param_setting".format(self.param_setting))

    def get_logdir(self):
        return os.path.join(self.data_root, self.exp_name,
                            "{}_run".format(self.run),
                            "{}_param_setting".format(self.param_setting))

    def log_config(self, logger):
        attrs = self.get_attrs()
        for param, value in sorted(attrs.items(), key=lambda x: x[0]):
            logger.info('{}: {}'.format(param, value))

    def get_attrs(self):
        attrs = dict(self.__dict__)
        # deleting params that I don't want to make part of outpath_name
        for k in ['exp_name', 'run', 'env_instance', 'data_root',
                  'out_path', 'tile_combinations']:
            del attrs[k]
        return attrs


class QConfig(Config):
    def __init__(self):
        super(QConfig, self).__init__()
        # Learning agent's parameters
        self.agent_class = "Q"
        self.alpha = 0.1
        self.epsilon = 0.0
        self.param_init = 0.0


class ActorCriticConfig(Config):
    def __init__(self):
        super(ActorCriticConfig, self).__init__()
        # Learning agent's parameters
        self.agent_class = "ActorCritic"
        self.alpha_w = 0.1
        self.alpha_theta = 0.1
        self.param_init = 0.0


