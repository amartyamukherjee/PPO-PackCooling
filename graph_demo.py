# Used to test the graphing library

import numpy as np
from environments.PackCooling import PackCooling

env = PackCooling()
env.reset()
for _ in range(2048):
    env.step(np.random.randint(-1,2))
    # env.step(-1)
env.render()