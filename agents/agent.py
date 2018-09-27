import random
import numpy as np

from agents.representation import TileCoder


class Agent(object):
    def __init__(self, cfg):
        self.env = cfg.env_instance
        self.num_actions = self.env.num_actions()
        self.state_dim = self.env.num_obs()

    def start(self, s):
        return self.get_action(s)

    def get_action(self, s):
        raise NotImplementedError

    def get_value(self, s, a):
        raise NotImplementedError

    def update(self, s, ns, r, a, done):
        raise NotImplementedError

    def set_seed(self, seed):
        pass

    def arg_max(self, action_values):
        max_ids = np.where(action_values == np.max(action_values))[0]
        assert len(max_ids) >= 1
        return np.random.choice(max_ids)

    @staticmethod
    def td_error(phi_s, phi_ns, r, w, gamma, done):
        if not done:
            return r + gamma*phi_ns.dot(w) - phi_s.dot(w)
        else:
            return r - phi_s.dot(w)


class Q(Agent):
    def __init__(self, cfg):
        super(Q, self).__init__(cfg)
        self.alpha = cfg.alpha
        self.rep, self.num_features, self.alpha = self.setup_representation(cfg)

        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon
        self.w = np.full(self.num_features, cfg.param_init)

    def setup_representation(self, cfg):
        if cfg.representation == "tile_code":
            rep = TileCoder(cfg)
            num_features = rep.get_num_features()
            alpha = self.alpha/cfg.num_tilings
            if cfg.tile_combinations == 1:
                alpha /= len(cfg.env.get_combinations())
            return rep, num_features, alpha
        else:
            raise NotImplementedError

    def get_action(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions-1)
        else:
            action_vals = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                phi = self.rep.get_representation(s, a)
                action_vals[a] = phi.dot(self.w)
            return self.arg_max(action_vals)

    def get_value(self, s, a):
        phi = self.rep.get_representation(s, a)
        return phi.dot(self.w)

    def learn(self, s, a, r, ns, done):
        phi_s = self.rep.get_representation(s, a)
        phi_ns = self.rep.get_representation(ns, self.max_action(ns))
        tde = Q.td_error(phi_s, phi_ns, r, self.w, self.gamma, done)
        self.w = self.w + self.alpha * tde * phi_s

    def max_action(self, s):
        action_vals = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            phi = self.rep.get_representation(s, a)
            action_vals[a] = phi.dot(self.w)
        return self.arg_max(action_vals)

    def update(self, s, ns, r, a, done):
        self.learn(s, ns, r, a, done)

    def reset(self):
        pass

    def start(self, s):
        return self.get_action(s)


class ActorCritic(Agent):
    def __init__(self, cfg):
        super(ActorCritic, self).__init__(cfg)
        self.alpha_w = cfg.alpha_w
        self.rep, self.num_features, self.alpha_w = self.setup_representation(cfg)
        self.alpha_theta = cfg.alpha_theta  # TQ: Why don't we divide the learning rate with # tilings
        self.gamma = cfg.gamma

        self.w = np.full(self.num_features, cfg.param_init)
        self.theta = np.zeros(self.num_features)
        self.I = 1

    def setup_representation(self, cfg):
        if cfg.representation == "tile_code":
            rep = TileCoder(cfg)
            num_features = rep.get_num_features()
            alpha = self.alpha_w/cfg.num_tilings
            if cfg.tile_combinations:
                alpha /= len(cfg.env.get_combinations())
            return rep, num_features, alpha
        else:
            raise NotImplementedError

    def softmax_policy(self, s):
        # Getting representations of the state for all actions
        X = np.zeros((self.num_actions, self.num_features))
        for a in range(self.num_actions):
            X[a, :] = self.rep.get_representation(s, a)

        # Computing the numerical preference
        h = X.dot(self.theta)

        # Computing the softmax (numerically stable)
        p = np.exp(h - np.max(h)) / np.sum(np.exp(h - np.max(h)))

        return p

    def get_action(self, s):
        p = self.softmax_policy(s)

        # Sampling an action according to the policy
        action = np.random.choice(self.num_actions, 1, p=p)[0]
        return action

    def get_value(self, s, a):
        phi = self.rep.get_representation(s, a)
        return phi.dot(self.w)

    def learn(self, s, a, r, ns, done):
        phi_s = self.rep.get_representation(s)
        phi_ns = self.rep.get_representation(ns)
        tde = Agent.td_error(phi_s, phi_ns, r, self.w, self.gamma, done)

        self.w = self.w + self.alpha_w * self.I * tde * phi_s
        self.theta = self.theta + self.alpha_theta * self.I * tde * self.gradient_policy(s, a)
        self.I = self.gamma*self.I

    def gradient_policy(self, s, a):
        p = self.softmax_policy(s)
        phi_sa = self.rep.get_representation(s, a)
        diff = np.zeros(phi_sa.shape)
        for b in range(self.num_actions):
            phi_sb = self.rep.get_representation(s, b)
            diff += p[b] * phi_sb
        grad = phi_sa - diff

        return np.squeeze(grad)

    def max_action(self, s):
        action_vals = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            phi = self.rep.get_representation(s, a)
            action_vals[a] = phi.dot(self.w)
        return self.arg_max(action_vals)

    def update(self, s, ns, r, a, done):
        self.learn(s, ns, r, a, done)

    def reset(self):
        pass

    def start(self, s):
        self.I = 1
        return self.get_action(s)
