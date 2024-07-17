# AlphaTensor
The **AlphaTensor project**, inspired by AlphaZero, aims to decompose specific tensors into rank-1 tensors (CP rank). This decomposition can be utilized in fast matrix multiplication algorithms 
![alt text](./image/alphatensor.png)

**Implementing AlphaTensor relies mainly on two components:**  
1.A neural network that takes input state information (specific tensor) and outputs the corresponding policy and value.  
2.Monte Carlo Tree Search (MCTS) guided by the neural network.

**The main steps to implement AlphaTensor (similar to AlphaZero):**  
1.Generate a dataset using MCTS under the guidance of the old neural network. Due to the specific nature of the AlphaTensor task, we can generate a large dataset ourselves (as is well known, tensor decomposition into rank-1 matrices is difficult, but assembling a tensor from many rank-1 matrices is simple; we just need to perform summation operations).  
2.Train the old neural network on the generated dataset to produce a new neural network.  
3.Evaluate whether the new neural network is better than the old one. If so, return to the first step and continue self-reinforcement learning.




## Helpful Resources
[AlphaZeroFromScratch](https://github.com/foersterrobert/AlphaZeroFromScratch) 

[nebuly](https://github.com/nebuly-ai/nebuly/)

[First Open Source Implementation of DeepMind’s AlphaTensor](https://www.kdnuggets.com/2023/03/first-open-source-implementation-deepmind-alphatensor.html)

## reference:
[Discovering faster matrix multiplication algorithms with reinforcement learning](https://www.nature.com/articles/s41586-022-05172-4)

[Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)

[Learning and Planning in Complex Action Spaces](https://arxiv.org/abs/2104.06303)

[Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961)


