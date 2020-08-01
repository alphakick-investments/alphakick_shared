import random as rand

import numpy as np

from indicators import *
from col_refs import *


class QLearner(object):

    def __init__(self, num_states=100, num_actions=4, alpha=0.2, gamma=0.9, rar=0.98, radr=0.999, dyna=0,
                 verbose=False):

        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose

        self.s = 0
        self.a = 0

        self.Q = np.zeros(shape=(num_states, num_actions))
        self.R = np.zeros(shape=(num_states, num_actions))

        self.T = np.zeros((num_states, num_actions, num_states))
        self.Tc = np.zeros((num_states, num_actions, num_states))

        self.num_actions = num_actions
        self.num_states = num_states

    def querysetstate(self, s):
        rand.seed(0)
        np.random.seed(0)

        if np.random.uniform() < self.rar:
            action = rand.randint(0, self.num_actions - 1)

        else:
            action = self.Q[s, :].argmax()

        self.s = s
        self.a = action

        if self.verbose: print("s =", s, "a =", action)

        return action

    def query(self, s_prime, r):
        rand.seed(0)
        np.random.seed(0)

        if np.random.uniform() < self.rar:
            action = rand.randint(0, self.num_actions - 1)

        else:
            action = self.Q[s_prime, :].argmax()

        r_fut = self.Q[s_prime, self.Q[s_prime, :].argmax()]

        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha * (r + (self.gamma * r_fut))

        self.rar = self.rar * self.radr

        if self.dyna != 0:
            self.model_update(self.s, self.a, s_prime, r)

            self.hallucinate()

        self.s = s_prime
        self.a = action

        if self.verbose: print("s =", s_prime, "a =", action, "r =", r)

        return action

    def model_update(self, s, a, s_prime, r):
        self.Tc[s, a, s_prime] += 1
        self.T = self.Tc / self.Tc.sum(axis=2, keepdims=True)
        self.R[s, a] = ((1 - self.alpha) * self.R[s, a]) + (self.alpha * r)

    def hallucinate(self):
        rand.seed(0)
        np.random.seed(0)

        for i in range(0, self.dyna):
            s_rnd = rand.randint(0, self.num_states - 1)
            a_rnd = rand.randint(0, self.num_actions - 1)

            s_prime = np.random.multinomial(100, self.T[s_rnd, a_rnd, :]).argmax()

            r_rnd = self.R[s_rnd, a_rnd]
            r_fut = self.Q[s_prime, self.Q[s_prime, :].argmax()]

            self.Q[s_rnd, a_rnd] = (1 - self.alpha) * self.Q[s_rnd, a_rnd] + self.alpha * (r_rnd + (self.gamma * r_fut))
