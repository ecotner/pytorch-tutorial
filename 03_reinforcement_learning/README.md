# Reinforcement learning
In this tutorial, we're going to create a Deep Q-Network (DQN) to learn to play CartPole, a game in which a cart tries to keep a pole balanced on end.

An introduction to OpenAI Gym is [here](https://gym.openai.com/docs/).
A description of CartPole is [here](https://gym.openai.com/envs/CartPole-v0/).

The actions control how much force to apply (left or right) to the cart at any given moment, the state is the numerical difference between consecutive images of the screen, and the reward is a constant +1 for every timestep the pole stays balanced (angle from vertical below some threshold). There is a good deal of feature engineering that can be done here (like using the provided angles and positions rather than raw pixels).