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

    # training
    def train_plot(num_batch, episode_per_batch):
        agent.network.train()
        for batch in range(num_batch):
            log_probs, rewards = [], []
            total_rewards, final_rewards = [], []

            for episode in range(episode_per_batch):

                state = env.reset()
                total_reward, total_step = 0, 0

                while True:

                    action, log_prob = agent.sample(state)
                    next_state, reward, done, _ = env.step(action)

                    log_probs.append(log_prob)
                    state = next_state
                    total_reward += reward
                    total_step += 1

                    if done:
                        final_rewards.append(reward)
                        total_rewards.append(total_reward)
                        rewards.append(np.full(total_step, total_reward))
                        break

            avg_total_reward = sum(total_rewards) / len(total_rewards)
            avg_final_reward = sum(final_rewards) / len(final_rewards)
            avg_total_rewards.append(avg_total_reward)
            avg_final_rewards.append(avg_final_reward)
            print("Batch {},\tTotal Reward = {:.1f},\tFinal Reward = {:.1f}".format(batch + 1, avg_total_reward,
                                                                                    avg_final_reward))

            if avg_final_reward == 100:
                print("Success!")
                break
            rewards = np.concatenate(rewards, axis=0)
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)
            agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))

        torch.save(agent.network.state_dict(), "Network.param")
        plt.plot(avg_total_rewards)
        plt.title("Total Rewards")
        plt.savefig("Total Rewards.jpg")
        plt.show()
        plt.plot(avg_final_rewards)
        plt.title("Final Rewards")
        plt.savefig("Final Rewards.jpg")
        plt.show()


    avg_total_rewards, avg_final_rewards = [], []
    num_batch = 1000
    episode_per_batch = 10
    lr = 0.002
    network = PolicyGradientNetwork()
    agent = PolicyGradientAgent(network, optim.Adam(network.parameters(), lr=lr))
    train_plot(num_batch, episode_per_batch)


    def test(num, show):
        success = 0
        total_reward = 0
        for i in range(0, num):
            agent.network.eval()
            state = env.reset()

            if show == True:
                img = plt.imshow(env.render(mode='rgb_array'))
            done = False
            final_reward = 0
            while not done:
                action, _ = agent.sample(state)
                state, reward, done, _ = env.step(action)
                final_reward = reward
                total_reward += reward

                if show == True:
                    img = plt.imshow(env.render(mode='rgb_array'))
                    display.display(plt.gcf())
                    display.clear_output(wait=True)
            if final_reward != -100:
                success = success + 1
        print("Success rate: {}, average reward: {}.".format(success / num, total_reward / num))


    test(10, False)