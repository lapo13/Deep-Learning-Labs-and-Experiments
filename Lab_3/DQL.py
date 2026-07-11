import torch
import torch.nn.functional as F
import numpy as np
import random
import tqdm as tq
from collections import deque


class DeepQLearningAgent:
    def __init__(self, policy_net, target_net, optimizer, env, memory,
                 loss_fn=F.mse_loss, batch_size=32, gamma=0.99, tau=0.005,
                 double=False, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.policy_net = policy_net.to(self.device)
        self.target_net = target_net.to(self.device)

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.env = env
        self.ReplayMemory = memory          
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau                 
        self.double = double
        self._trained = False  


        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def load_model(self, policy_path, target_path):
        self.policy_net.load_state_dict(torch.load(policy_path, map_location=self.device))
        self.target_net.load_state_dict(torch.load(target_path, map_location=self.device))
        self.policy_net.eval()
        self.target_net.eval()
        self._trained = True

    def save_model(self, policy_path, target_path):
        torch.save(self.policy_net.state_dict(), policy_path)
        torch.save(self.target_net.state_dict(), target_path)

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.env.action_space.n)
        state = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def soft_update(self):
        # theta_target <- tau * theta_policy + (1 - tau) * theta_target
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.data.mul_(1.0 - self.tau)
                tp.data.add_(self.tau * pp.data)

    def replay(self, start_training=1000):
        if len(self.ReplayMemory) < start_training:
            return None

        self.policy_net.train()
        minibatch = random.sample(list(self.ReplayMemory), self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Solo il minibatch viene spostato sul device (il buffer resta su CPU).
        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(actions), dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(np.array(dones), dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            if self.double:
                next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
            else:
                next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        self.optimizer.zero_grad()
        self.loss_fn(current_q, target_q).backward()
        self.optimizer.step()

        self.soft_update()

        return
    
    def play(self, env, episodes=3, verbose=True):
        """
        Mostra l'agente addestrato (policy greedy) su un ambiente con render_mode="human".
        In questa modalita' Gymnasium disegna automaticamente la finestra pygame dentro step(),
        quindi basta far girare gli episodi. Chiude sempre l'ambiente (e la finestra) alla fine.
        """
        self.policy_net.eval()
        try:
            for ep in range(episodes):
                state, _ = env.reset()
                total_reward, done = 0.0, False
                while not done:
                    action = self.get_action(state, epsilon=0.0)
                    state, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                if verbose:
                    print(f"Episodio {ep + 1}: reward = {total_reward:.1f}")
        finally:
            env.close()
            self.policy_net.train()

    def _valuation(self, M, run, step):
        episode_rewards, episode_lengths = [], []
        self.policy_net.eval()

        with torch.no_grad():
            for _ in range(M):
                state = self.env.reset()[0]
                total_reward, length = 0, 0
                while True:
                    action = self.get_action(state, epsilon=0.0)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    total_reward += reward
                    length += 1
                    state = next_state
                    if terminated or truncated:
                        break
                episode_rewards.append(total_reward)
                episode_lengths.append(length)

        mean_reward = sum(episode_rewards) / M
        mean_length = sum(episode_lengths) / M

        run.log({"mean_reward": mean_reward, "mean_length": mean_length}, step=step)

        self.policy_net.train()
        return 

    def DeepQLearning(self, episodes, epsilon=0.7, epsilon_min=0.3, epsilon_decay=0.001,
                  max_steps=500, start_training=1000, N=20, M=10, seed=None,
                  run=None):
        self._trained = True
        global_step = 0
        avg_window = int(episodes*0.1)  # dimensione della finestra per le medie mobili
        print(f"Dimensione della finestra per le medie mobili: {avg_window}")

        # finestre scorrevoli per le medie mobili
        reward_window = deque(maxlen=avg_window)
        length_window = deque(maxlen=avg_window)

        for episode in tq.tqdm(range(episodes), desc="Training"):
            state = self.env.reset(seed=seed)[0]
            episode_reward = 0.0
            episode_length = 0

            for step in range(max_steps):
                action = self.get_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.ReplayMemory.append((state, action, reward, next_state, terminated))
                state = next_state

                episode_reward += reward
                episode_length += 1

                self.replay(start_training=start_training)

                global_step += 1
                if terminated or truncated:
                    break

            # aggiorna le finestre e calcola le medie mobili
            reward_window.append(episode_reward)
            length_window.append(episode_length)
            avg_reward = sum(reward_window) / len(reward_window)
            avg_length = sum(length_window) / len(length_window)

            if run is not None:
                run.log({
                    "train/avg_reward": avg_reward,
                    "train/avg_length": avg_length,
                }, step=global_step)

            if N != 0 and episode % N == 0 and run is not None:
                self._valuation(M, run, step=global_step)

            if epsilon > epsilon_min:
                epsilon = max(epsilon_min, epsilon - epsilon_decay)
                if run is not None:
                    run.log({"train/epsilon": epsilon}, step=global_step)

        return