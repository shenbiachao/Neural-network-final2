from pyvirtualdisplay import Display
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym

if __name__ == "__main__":
    # environment preparing
    virtual_display = Display(visible=0, size=(1400, 900))
    virtual_display.start()
    env = gym.make('LunarLander-v2')

    # net constructing
    class PolicyGradientNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 4)

        def forward(self, state):
            hid = torch.tanh(self.fc1(state))
            hid = torch.tanh(self.fc2(hid))
            return F.softmax(self.fc3(hid), dim=-1)


    class PolicyGradientAgent():
        def __init__(self, network, opt):
            self.network = network
            self.optimizer = opt

        def learn(self, log_probs, rewards):
            loss = (-log_probs * rewards).sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        def sample(self, state):
            action_prob = self.network(torch.FloatTensor(state))
            action_dist = Categorical(action_prob)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            return action.item(), log_prob


    lr = 0.002
    network = PolicyGradientNetwork()
    network.load_state_dict(torch.load("./Network.param"))
    agent = PolicyGradientAgent(network, optim.Adam(network.parameters(), lr=lr))

    def test(num):
        success = 0
        total_reward = 0
        for i in range(0, num):
            agent.network.eval()
            state = env.reset()

            done = False
            final_reward = 0
            while not done:
                action, _ = agent.sample(state)
                state, reward, done, _ = env.step(action)
                final_reward = reward
                total_reward += reward

            if final_reward != -100:
                success = success + 1
        print("Success rate: {}, average reward: {}.".format(success / num, total_reward / num))


    test(10)