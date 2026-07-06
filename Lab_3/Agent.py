import torch
from torch.distributions import Categorical
import os


class ReinforceAgent:
     def __init__(self, policy, optimizer, env, gamma=0.99, loss=lambda log_probs, returns: (-log_probs * returns).mean(), model_path="./model"):
          self.env = env
          self.policy = policy
          self.opt = optimizer
          self.gamma = gamma
          self.score = loss
          self.model_path = model_path
          if os.path.exists(model_path):
               self.policy.load_state_dict(torch.load(model_path))
               self.policy.eval()
          else:
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
          for i in range(maxlen):
               # Get the current observation, run the policy and select an action.
               obs = torch.tensor(obs)
               (action, log_prob) = self.select_action(obs)
               observations.append(obs)
               actions.append(action)
               log_probs.append(log_prob)
               
               # Advance the episode by executing the selected action.
               (obs, reward, term, trunc, info) = self.env.step(action)
               rewards.append(reward)
               if term or trunc:
                    break
          return (observations, actions, torch.cat(log_probs), rewards)
     
     def reinforce(self, env_render=None, num_episodes=10, checkpoint = False):

          # Track episode rewards in a list.
          running_rewards = [0.0]
          scores = [0.0]
          
          # The main training loop.
          self.policy.train()
          for episode in range(num_episodes):
               # Run an episode of the environment, collect everything needed for policy update.
               (observations, actions, log_probs, rewards) = self.run_episode()
               
               returns = torch.tensor(self.compute_returns(rewards), dtype=torch.float16)

               # Keep a running average of total discounted rewards for the whole episode.
               running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])
               
               # Standardize returns.
               returns = (returns - returns.mean()) / returns.std()
               
               # Make an optimization step
               self.opt.zero_grad()
               score = self.score(log_probs, returns)
               if checkpoint and (score.item() > max(scores)): #stiamo massimizzando la loss, quindi salviamo il modello se la loss è maggiore della precedente
                    torch.save(self.policy.state_dict(), self.model_path + '/checkpoint.pt')
               score.backward()
               scores.append(score.item())
               self.opt.step()
               
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
          return running_rewards, scores