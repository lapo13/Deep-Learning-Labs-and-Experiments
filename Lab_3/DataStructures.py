# Creiamo un contenitore leggero che si comporta come una normale tupla a 5 elementi
class EpisodeData(tuple):
     def __new__(cls, observations, actions, log_probs, rewards, length, state_values):
          return super().__new__(cls, (observations, actions, log_probs, rewards, length))
     
     def __init__(self, observations, actions, log_probs, rewards, length, state_values):
          self.state_values = state_values

class ReinforceData(tuple):
     def __new__(cls, running_rewards, M_mean_reward, M_mean_length, loss_values):
          return super().__new__(cls, (running_rewards, M_mean_reward, M_mean_length))
     
     def __init__(self, running_rewards, M_mean_reward, M_mean_length, loss_values):
          self.loss_values = loss_values
     