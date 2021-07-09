# todo
# Implement TD, UPGO, V-trace

import numpy as np
from collections import deque

def vtrace(values, returns, rewards, gamma, rhos, cs):
    v_t_plus_1 = np.concatenate((values[1:], returns[-1:]))
    deltas = rhos * (rewards + gamma * v_t_plus_1 - values)
    vs_minus_v_xs = deque([deltas[-1]])
    for i in range(len(values) - 2, -1, -1): 
        vs_minus_v_xs.appendleft(deltas[i] + gamma * cs[i] * vs_minus_v_xs[0])
    
    vs = np.array(vs_minus_v_xs) + np.array(values)
    vs_t_plus_1 = np.concatenate((vs[1:], returns[-1:]))
    advantages = rewards + gamma * vs_t_plus_1 - values
    return vs, advantages