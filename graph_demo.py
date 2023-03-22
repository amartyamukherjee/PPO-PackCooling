# Used to test the graphing library

from environments.PackCooling import PackCooling

env = PackCooling()
env.reset()
for _ in range(20):
    env.step(0)
env.render()