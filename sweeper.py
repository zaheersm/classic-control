import json

import config


class Sweeper(object):
    """
    The purpose of this class is to take an index, identify a configuration
    of hyper-parameters and create a Config object

    Important: parameters part of the sweep are provided in a list
    """
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config_dict = json.load(f)
        self.total_combinations = 1
        self.set_total_combinations()

    def set_total_combinations(self):
        if 'sweep_parameters' in self.config_dict:
            sweep_params = self.config_dict['sweep_parameters']
            # calculating total_combinations
            tc = 1
            for params, values in sweep_params.items():
                tc = tc * len(values)
            self.total_combinations = tc

    def parse(self, idx):
        config_class = getattr(config, self.config_dict['config_class'])
        cfg = config_class()

        # Populating fixed parameters
        fixed_params = self.config_dict['fixed_parameters']
        for param, value in fixed_params.items():
            setattr(cfg, param, value)

        cumulative = 1

        # Populating sweep parameters
        if 'sweep_parameters' in self.config_dict:
            sweep_params = self.config_dict['sweep_parameters']
            for param, values in sweep_params.items():
                num_values = len(values)
                setattr(cfg, param, values[int(idx/cumulative) % num_values])
                cumulative *= num_values
        cfg.run = int(idx/cumulative)
        cfg.param_setting = idx % cumulative
        self.total_combinations = cumulative
        return cfg

    def param_setting_from_id(self, idx):
        sweep_params = self.config_dict['sweep_parameters']
        param_setting = {}
        cumulative = 1
        for param, values in sweep_params.items():
            num_values = len(values)
            param_setting[param] = values[int(idx/cumulative) % num_values]
            cumulative *= num_values
        return param_setting



if __name__ == '__main__':
    sweeper = Sweeper("config_files/actor_critic_dyna.json")
    sweeper.parse(28)

