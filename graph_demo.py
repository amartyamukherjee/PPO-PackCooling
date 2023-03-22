# Used to test the graphing library

import numpy as np
from environments.PackCooling import PackCooling

env = PackCooling()
env.reset()
for _ in range(20):
    env.step(np.random.randint(2))
env.render()