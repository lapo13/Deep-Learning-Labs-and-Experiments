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
    

class DeepQLearningMultiAgent(DeepQLearningAgent):
    def __init__(self, policy_net, target_net, optimizer, envs, eval_env, memory,
                 loss_fn=F.mse_loss, batch_size=32, gamma=0.99, tau=0.005,
                 double=False, device=None):
        super().__init__(policy_net, target_net, optimizer, envs, memory,
                         loss_fn, batch_size, gamma, tau, double, device)
        self.eval_env = eval_env   # ambiente SERIALE, usato solo per la valutazione

    def get_action(self, state, epsilon):
        num_envs = self.env.num_envs
        n_actions = self.env.single_action_space.n

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)              # (num_envs, n_actions)
        greedy = q_values.argmax(dim=1).cpu().numpy()        # (num_envs,)

        random_a = np.random.randint(0, n_actions, size=num_envs)
        explore = np.random.random(num_envs) < epsilon       # una scelta per env
        return np.where(explore, random_a, greedy)           # (num_envs,)

    def _valuation(self, M, run, step):
        episode_rewards, episode_lengths = [], []
        self.policy_net.eval()
        with torch.no_grad():
            for _ in range(M):
                state, _ = self.eval_env.reset()
                total_reward, length = 0.0, 0
                while True:
                    state_t = torch.as_tensor(
                        state, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    action = self.policy_net(state_t).argmax(dim=1).item()
                    state, reward, terminated, truncated, _ = self.eval_env.step(action)
                    total_reward += reward
                    length += 1
                    if terminated or truncated:
                        break
                episode_rewards.append(total_reward)
                episode_lengths.append(length)
        mean_reward = sum(episode_rewards) / M
        mean_length = sum(episode_lengths) / M
        if run is not None:
            run.log({"eval/mean_reward": mean_reward,
                     "eval/mean_length": mean_length}, step=step)
        self.policy_net.train()
        return 

    def DeepQLearning(self, episodes, epsilon=0.7, epsilon_min=0.3, epsilon_decay=0.001,
                      max_steps=500, start_training=1000, N=20, M=10, seed=None,
                      run=None):
        self._trained = True
        num_envs = self.env.num_envs
        avg_window = 100

        # budget: stesse transizioni totali della versione seriale (episodes*max_steps),
        # ma ogni iterazione del loop ne raccoglie num_envs -> divido.
        total_steps = (episodes * max_steps) // num_envs
        evaluate_every = max(1, (N * max_steps) // num_envs)

        # accumulatori dell'episodio in corso, uno per env
        current_reward = np.zeros(num_envs, dtype=np.float32)
        current_length = np.zeros(num_envs, dtype=np.int32)
        # medie mobili aggregate sugli ultimi avg_window episodi conclusi
        reward_window = deque(maxlen=avg_window)
        length_window = deque(maxlen=avg_window)

        state, _ = self.env.reset(seed=seed)
        episode_start = np.zeros(num_envs, dtype=bool)

        for global_step in tq.tqdm(range(total_steps), desc="Training"):
            action = self.get_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            current_reward += reward
            current_length += 1

            # salva le transizioni, saltando quelle "a ponte" dopo l'autoreset
            for i in range(num_envs):
                if not episode_start[i]:
                    self.ReplayMemory.append(
                        (state[i], action[i], reward[i], next_state[i], terminated[i])
                    )

            done = np.logical_or(terminated, truncated)
            # per ogni env appena finito: chiudi episodio, aggiorna media, resetta
            for i in range(num_envs):
                if done[i]:
                    reward_window.append(current_reward[i])
                    length_window.append(current_length[i])
                    current_reward[i] = 0.0
                    current_length[i] = 0

            episode_start = done
            state = next_state

            self.replay(start_training=start_training)

            # logging periodico delle medie mobili aggregate
            if run is not None and global_step % 100 == 0 and len(reward_window) > 0:
                run.log({
                    "train/avg_reward": np.mean(reward_window),
                    "train/avg_length": np.mean(length_window),
                }, step=global_step)

            # valutazione periodica sull'ambiente seriale
            if global_step % evaluate_every == 0 and run is not None:
                self._valuation(M, run, step=global_step)

            # decay di epsilon (attenzione: qui è PER STEP del loop, non per episodio)
            if epsilon > epsilon_min:
                epsilon = max(epsilon_min, epsilon - epsilon_decay)
                if run is not None:
                    run.log({"train/epsilon": epsilon}, step=global_step)

        return