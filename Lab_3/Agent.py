import torch
from torch.nn.functional import mse_loss
from torch.distributions import Categorical
import os
from DataStructures import EpisodeData, ReinforceData


class ReinforceAgent:
     def __init__(self, policy, optimizer, env, gamma=0.99, loss=lambda log_probs, returns: (-log_probs * returns).mean(), model_path="./model"):
          self.env = env
          self.policy = policy
          self.opt = optimizer
          self.gamma = gamma
          self.score = loss
          self.model_path = model_path
          if not os.path.exists(model_path):
               os.makedirs(model_path)

     def select_action(self, obs):
          dist = Categorical(self.policy(obs))
          action = dist.sample()
          log_prob = dist.log_prob(action)
          return (action.item(), log_prob.reshape(1))


     def compute_returns(self, rewards):
          discounted_sum = 0
          returns = []
          for r in reversed(rewards):
               discounted_sum = r + self.gamma * discounted_sum
               returns.append(discounted_sum)
          returns.reverse()
          return returns

     # Given an environment and a policy, run it up to the maximum number of steps.
     def run_episode(self, maxlen=500):
          observations = []
          actions = []
          log_probs = []
          rewards = []
          
          (obs, _) = self.env.reset()
          episode_length = 0
          while episode_length != maxlen:
               # Get the current observation, run the policy and select an action.
               obs = torch.tensor(obs, dtype=torch.float32)
               (action, log_prob) = self.select_action(obs)
               observations.append(obs)
               actions.append(action)
               log_probs.append(log_prob)
               
               # Advance the episode by executing the selected action.
               (obs, reward, term, trunc, info) = self.env.step(action)
               rewards.append(reward)
               episode_length += 1
               if term or trunc:
                    break
               
          return (observations, actions, torch.cat(log_probs), rewards, episode_length)
     
     def valuation(self, M):
          episode_rewards = []
          episode_lengths = []
          self.policy.eval()

          with torch.no_grad():
               for i in range(M):
                    obs, action, log_prob, rewards, lenght = self.run_episode()
                    episode_rewards.append(sum(rewards))
                    episode_lengths.append(lenght)
          mean_reward = sum(episode_rewards)/M
          mean_length = sum(episode_lengths)/M
          print(f"Mean reward: {mean_reward}, Mean length: {mean_length}")

          self.policy.train()
          return mean_reward, mean_length
     
     def reinforce(self, env_render=None, num_episodes=10, checkpoint = False, N = 100, M = 10, standardize = False):

          # Track episode rewards in a list.
          running_rewards = [0.0]
          best_reward = -float('inf')
          M_mean_reward, M_mean_length = [], []
          
          # The main training loop.
          self.policy.train()
          for episode in range(num_episodes):
               # Run an episode of the environment, collect everything needed for policy update.
               (observations, actions, log_probs, rewards, _) = self.run_episode()
               
               returns = torch.tensor(self.compute_returns(rewards), dtype=torch.float32)

               # Keep a running average of total discounted rewards for the whole episode.
               running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])
               
               # Standardize returns.
               if standardize:
                    returns = (returns - returns.mean()) / (returns.std()+1e-6) #optimized to prevent division by zero

               if N != 0 and not episode % N:
                    reward_mean, lenght_mean = self.valuation(M)
                    M_mean_reward.append(reward_mean)
                    M_mean_length.append(lenght_mean)
               
               # Make an optimization step
               self.opt.zero_grad()
               self.score(log_probs, returns).backward()
               self.opt.step()

               #save the model if the running reward is the best so far
               if checkpoint and (running_rewards[-1] >= best_reward):
                    best_reward = running_rewards[-1]
                    torch.save(self.policy.state_dict(), self.model_path + '/checkpoint.pt')
               
               # Render an episode after every 100 policy updates.
               if not episode % 100:
                    if env_render:
                         self.policy.eval()
                         _ = self.run_episode()
                         self.policy.train()
                    print(f'Running reward: {running_rewards[-1]}')
          
          if checkpoint:
               self.policy.load_state_dict(torch.load(self.model_path+'/checkpoint.pt'))
          self.policy.eval()
          return (running_rewards, M_mean_reward, M_mean_length)
     

class ReinforceAgentWithBaseline(ReinforceAgent):
     def __init__(self, policy, opt, env, baseline, critic_opt, gamma=0.99, loss=lambda log_probs, returns: (-log_probs * returns).mean(), critic_loss = mse_loss, model_path="./model"):
          super().__init__(policy, opt, env, gamma, loss, model_path)
          self.critic = baseline
          self.critic_opt = critic_opt
          self.critic_loss = critic_loss
     
     def critic_valuation(self, obs):
          return self.critic(obs)
     
     def run_episode(self, maxlen=500):
          observations, actions, log_probs, rewards, state_values = [], [], [], [], []
          
          (obs, _) = self.env.reset()
          episode_length = 0
          while episode_length != maxlen:
               obs_tensor = torch.tensor(obs, dtype=torch.float32)
               (action, log_prob) = self.select_action(obs_tensor)
               
               observations.append(obs_tensor)
               actions.append(action)
               log_probs.append(log_prob)
               
               state_values.append(self.critic_valuation(obs_tensor))
               
               (obs, reward, term, trunc, info) = self.env.step(action)
               rewards.append(reward)
               episode_length += 1

               if term or trunc:
                    break
          
          values_tensor = torch.cat(state_values)
               
          return EpisodeData(observations, actions, torch.cat(log_probs), rewards, episode_length, values_tensor)
     
     def reinforce(self, env_render=None, num_episodes=10, checkpoint = False, N = 100, M = 10, standardize = False):
          running_rewards, M_mean_reward, M_mean_length, loss_values = [0.0], [], [], []
          best_reward = -float('inf')
          
          self.policy.train()
          self.critic.train()
          
          for episode in range(num_episodes):
               # Run an episode of the environment, collect everything needed for policy update.
               episode_data = self.run_episode()
               
               (_, _, log_probs, rewards, _) = episode_data
               
               values_clean = episode_data.state_values
               
               returns = torch.tensor(self.compute_returns(rewards), dtype=torch.float32)

               delta = returns - values_clean.detach()

               # Keep a running average of total discounted rewards for the whole episode.
               running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])

               if standardize:
                    delta = (delta - delta.mean()) / (delta.std()+1e-6) #optimized to prevent division by zero

               if N != 0 and not episode % N:
                    reward_mean, lenght_mean = self.valuation(M)
                    M_mean_reward.append(reward_mean)
                    M_mean_length.append(lenght_mean)
               
               # Make an optimization step
               self.opt.zero_grad()
               self.score(log_probs, delta).backward()
               self.opt.step()

               self.critic_opt.zero_grad()
               loss = self.critic_loss(values_clean, returns)
               loss.backward()
               loss_values.append(loss.item())

               self.critic_opt.step()

               #save the model if the running reward is the best so far
               if checkpoint and (running_rewards[-1] > best_reward):
                    best_reward = running_rewards[-1]
                    torch.save(self.policy.state_dict(), self.model_path + '/checkpoint.pt')
                    torch.save(self.critic.state_dict(), self.model_path + '/baseline_checkpoint.pt')
               
               # Render an episode after every 100 policy updates.
               if not episode % 100:
                    if env_render:
                         self.policy.eval()
                         _ = self.run_episode()
                         self.policy.train()
                    print(f'Running reward: {running_rewards[-1]}')
          
          if checkpoint:
               self.policy.load_state_dict(torch.load(self.model_path+'/checkpoint.pt'))
          self.policy.eval()
          self.critic.eval()
          return ReinforceData(running_rewards, M_mean_reward, M_mean_length, loss_values)