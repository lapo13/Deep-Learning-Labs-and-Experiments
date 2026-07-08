import torch
import torch.nn.functional as F
import numpy as np
import random
import tqdm as tq


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
        self.memory = memory          
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau                 
        self.double = double            


        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

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
        if len(self.memory) < start_training:
            return None

        self.policy_net.train()
        minibatch = random.sample(list(self.memory), self.batch_size)
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

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update()

        return loss.item()

    def valuation(self, M):
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

        self.policy_net.train()
        return mean_reward, mean_length

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

    def DeepQLearning(self, episodes, epsilon=0.7, epsilon_min=0.3, epsilon_decay=0.001,
                      max_steps=500, start_training=1000, N=20, M=10):
        rewards, lengths, losses = [], [], []

        for episode in tq.tqdm(range(episodes), desc="Training"):
            state = self.env.reset()[0]

            for step in range(max_steps):
                action = self.get_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                self.memory.append((state, action, reward, next_state, terminated))
                state = next_state

                l = self.replay(start_training=start_training)
                if l is not None:
                    losses.append(l)

                if terminated or truncated:
                    break

            if N != 0 and episode % N == 0:
                reward_mean, length_mean = self.valuation(M)
                rewards.append(reward_mean)
                lengths.append(length_mean)

            if epsilon > epsilon_min:
                epsilon = max(epsilon_min, epsilon - epsilon_decay)

        return rewards, lengths, losses