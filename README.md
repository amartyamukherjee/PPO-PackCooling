# PPO-PackCooling

Forked from PPO-PyTorch by Nihkil Baharte

![](/HJBPPO_figs/HJBPPO_Graph_0.png)

Source code for the paper "Actor-Critic Methods using Physics-Informed Neural Networks: Control of a 1D PDE Model for Fluid-Cooled Battery Packs" by Amartya Mukherjee and Jun Liu, 2023

Implementation of HJBPPO on HJBPPO.py

Implementation of the PackCooling environment is in environments/PackCooling.py

Implementation of the rendering feature in the PackCooling environment is in environments/PackCoolingGraph.py

To train HJBPPO on the environment, set the prob_optimal_control parameter in HJBPPO.HJBPPO to 0.5
To train HJB value iteration on the environment, set the prob_optimal_control paper in HJBPPO.HJBPPO to 1.0

## References

- [PPO paper](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning up](https://spinningup.openai.com/en/latest/)
- [PPO pack cooling paper](https://arxiv.org/abs/2305.10952)


