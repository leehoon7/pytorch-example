import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import gym
from collections import deque
import random


class ReplayMemory:

    def __init__(self, length):
        self.memory = deque(maxlen=length)

    def __len__(self):
        return len(self.memory)

    def save_memory(self, transition):
        self.memory.append(transition)

    def get_sample(self, size):
        batch = random.sample(self.memory, size)
        print(batch)


class DQNAgent(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DQNAgent, self).__init__()

        self.replay_memory = ReplayMemory(length=5000)
        self.gamma = 0.99
        self.batch_size = 5

        hidden_dim = 64
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.active1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.active1(x)
        x = self.layer2(x)
        return x

    def save_memory(self, transition):
        self.replay_memory.save_memory(transition)

    def train_start(self):
        return len(self.replay_memory) >= self.batch_size

    def train(self):
        return 0


if __name__ == "__main__":

    env = gym.make("CartPole-v0")

    state_dim = 4
    action_dim = 2
    Agent = DQNAgent(state_dim, action_dim)

    epsilon = 0.0
    epsilon_min = 0.005
    decay_rate = 0.001

    episode = 10
    loss_list = []
    reward_list = []

    for i in range(episode):

        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).reshape(-1, state_dim)
        done = False
        reward_epi = []

        while not done:

            # make an action based on epsilon greedy action
            action = None
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                action = Agent(obs)
                action = np.argmax(action.detach().numpy())

            # save a state before do action
            bef_obs = obs

            # do action and get new state, reward and done
            obs, reward, done, _ = env.step(action)
            obs = torch.tensor(obs, dtype=torch.float32).reshape(-1, state_dim)
            reward_epi.append(reward)

            # make a transition and save to replay memory
            transition = [bef_obs, action, reward, obs, done]
            Agent.save_memory(transition)

            if Agent.train_start():
                loss = Agent.train()
                loss_list.append(loss)

        env.close()

        if epsilon > epsilon_min:
            epsilon -= decay_rate
        else:
            epsilon = epsilon_min

        reward_list.append(sum(reward_epi))

    plt.plot(loss_list)
    plt.show()
    plt.close('all')

    plt.plot(reward_list)
    plt.show()
    plt.close('all')
