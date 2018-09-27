import numpy as np

import utils.tiles3 as TC


class Representation(object):
    """
    Dummy representation interface
    """
    def __init__(self, cfg):
        self.num_actions = cfg.env_instance.num_actions()
        self.num_obs = cfg.env_instance.num_obs()

    def get_num_features(self):
        return self.acts * self.obs

    def get_representation(self, obs, action):
        """
        Examples:
        obs: [2, 3] with action 1 (out of 3) returns:
            [0, 0, 2, 3, 0, 0]

        obs: [2, 3] with action 2 returns:
            [0, 0, 0, 0, 2, 3]
        """
        rep = np.zeros(self.num_obs * self.num_actions)
        rep[action * self.num_obs: (action + 1) * self.num_obs] = obs
        return rep


class TileCoder(Representation):
    """
    Tile Coding representation
    Config should have three attributes:
        mem_size: maximum size of the hash table memory
        tiles: number of tiles to separate each dimension into
        tilings: number of tilings of each dimension
        tile_combinations: if combinations of the state components are
                           to be tiled separately
    """
    def __init__(self, cfg):
        super(TileCoder, self).__init__(cfg)
        self.mem_size = cfg.mem_size
        self.tiles = cfg.tiles
        self.tilings = cfg.num_tilings
        self.tile_combinations = cfg.tile_combinations
        self.combinations = []
        self.env_instance = cfg.env_instance

        if self.tile_combinations:
            self.combinations = self.env_instance.get_combinations()
            assert self.mem_size % len(self.combinations) == 0
            self.com_mem = self.mem_size / len(self.combinations)
            self.iht = []
            for k in range(len(self.combinations)):
                self.iht.append(TC.IHT(self.com_mem))
        else:
            self.iht = TC.IHT(self.mem_size)

    def get_num_features(self):
        return self.mem_size

    def get_representation(self, state, action=None):
        # Assumption: state components are 0, 1 normalized
        ones = []
        if self.tile_combinations:
            for k, p in enumerate(self.combinations):
                ints = [] if action is None else [action]
                tc = TC.tiles(
                        self.iht[k], self.tilings,
                        float(self.tiles) * np.array([state[m] for m in p]),
                        ints=ints)
                tc = [x + (self.com_mem * k) for x in tc]
                ones.extend(list(tc))
        else:
            ints = [] if action is None else [action]
            tc = TC.tiles(self.iht, self.tilings, float(self.tiles) * state, ints=ints)
            ones.extend(list(tc))

        rep = np.zeros(self.mem_size)
        rep[ones] = 1
        return rep


if __name__ == '__main__':
    from environments.mountain_car import MountainCar
    import config
    cfg = config.Config()
    mc = MountainCar()
    cfg.env_instance = mc

    r = TileCoder(cfg)

    o = mc.reset()
    r.get_representation(o, 1)