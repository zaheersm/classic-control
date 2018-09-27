class Environment(object):
    def __init__(self):
        self.seed = 0

    def reset(self):
        raise NotImplementedError

    def get_string(self):
        s = self.__class__.__name__
        return s

    def step(self, a):
        raise NotImplementedError

    def num_obs(self):
        raise NotImplementedError

    def num_actions(self):
        raise NotImplementedError

    def get_combinations(self):
        raise NotImplementedError

    def set_seed(self, seed):
        pass
