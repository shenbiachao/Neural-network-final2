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
    class ActorNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 64)
            self.dp1 = nn.Dropout(p=0.25)
            self.fc2 = nn.Linear(64, 64)
            self.dp2 = nn.Dropout(p=0.25)
            self.fc3 = nn.Linear(64, 4)

        def forward(self, state):
            hid = torch.tanh(self.fc1(state))
            hid = self.dp1(hid)
            hid = torch.tanh(self.fc2(hid))
            hid = self.dp2(hid)
            return F.softmax(self.fc3(hid), dim=-1)


    class CriticNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 64)
            self.dp1 = nn.Dropout(p=0.25)
            self.fc2 = nn.Linear(64, 64)
            self.dp2 = nn.Dropout(p=0.25)
            self.fc3 = nn.Linear(64, 1)

        def forward(self, state):
            hid = torch.tanh(self.fc1(state))
            hid = self.dp1(hid)
            hid = torch.tanh(self.fc2(hid))
            hid = self.dp2(hid)
            return self.fc3(hid)


    class PolicyGradientAgent():
        def __init__(self, actor, critic, opt_actor, opt_critic, steps, factor):
            self.actor = actor
            self.critic = critic
            self.optimizer_actor = opt_actor
            self.optimizer_critic = opt_critic
            self.num_steps = steps
            self.gamma = factor

        def learn(self, log_probs, rewards, values):
            T = len(rewards)
            N = self.num_steps
            R = np.zeros(T, dtype=np.float32)
            loss_actor = 0
            loss_critic = 0
            for t in reversed(range(T)):
                V_end = 0 if (t + N >= T) else values[t + N].data
                R[t] = (self.gamma ** N * V_end) + sum(
                    [self.gamma ** k * rewards[t + k] * 1e-2 for k in range(min(N, T - t))])
            R = Variable(torch.FloatTensor(R), requires_grad=False).to(try_gpu())
            loss_actor = ((R - values.detach()) * -log_probs).mean()
            loss_critic = ((R - values) ** 2).mean()
            loss = loss_actor + loss_critic

            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            loss_actor.backward()
            loss_critic.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()

            return loss_actor, loss_critic, loss

        def sample(self, state):
            action_prob = self.actor(torch.FloatTensor(state).to(try_gpu()))
            action_dist = Categorical(action_prob)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            value = self.critic(torch.FloatTensor(state).to(try_gpu()))

            return action.item(), log_prob, value

    # training
    def try_gpu(i=0):
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f'cuda:{i}')
        return torch.device('cpu')


    def train_plot(num_batch, episode_per_batch):
        agent.actor.train()
        agent.critic.train()
        agent.actor.to(try_gpu())
        agent.critic.to(try_gpu())
        for batch in range(num_batch):
            log_probs, rewards, values = [], [], []
            total_rewards, final_rewards = [], []

            for episode in range(episode_per_batch):

                state = env.reset()
                total_reward, total_step = 0, 0

                while True:

                    action, log_prob, value = agent.sample(state)
                    next_state, reward, done, _ = env.step(action)

                    log_probs.append(log_prob)
                    values.append(value)
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

            rewards = np.concatenate(rewards, axis=0)
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)
            actor_loss, critic_loss, loss = agent.learn(torch.stack(log_probs), torch.from_numpy(rewards),
                                                        torch.stack(values))
            losses.append(loss)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        torch.save(agent.actor.state_dict(), "Actor.param")
        torch.save(agent.critic.state_dict(), "Critic.param")
        plt.plot(avg_total_rewards)
        plt.title("Total Rewards")
        plt.savefig("Total Rewards.jpg")
        plt.show()
        plt.plot(avg_final_rewards)
        plt.title("Final Rewards")
        plt.savefig("Final Rewards.jpg")
        plt.show()
        plt.plot(losses, label="losses")
        plt.plot(actor_losses, label="actor_losses")
        plt.plot(critic_losses, label="critic_losses")
        plt.title("Losses")
        plt.legend(loc="best")
        plt.savefig("Losses.jpg")
        plt.show()


    avg_total_rewards, avg_final_rewards, actor_losses, critic_losses, losses = [], [], [], [], []
    num_batches = 1000
    num_steps = 100
    gamma = 0.99
    episode_per_batch = 10
    lr = 0.002
    actor = ActorNetwork()
    critic = CriticNetwork()
    agent = PolicyGradientAgent(actor, critic, optim.Adam(actor.parameters(), lr=lr),
                                optim.Adam(critic.parameters(), lr=lr), num_steps, gamma)
    train_plot(num_batches, episode_per_batch)


    def test(num, show):
        success = 0
        total_reward = 0
        for i in range(0, num):
            agent.actor.eval()
            agent.critic.eval()
            state = env.reset()

            if show == True:
                img = plt.imshow(env.render(mode='rgb_array'))
            done = False
            final_reward = 0
            while not done:
                action, _, _ = agent.sample(state)
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