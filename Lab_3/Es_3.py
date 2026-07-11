import os, torch, wandb, random, numpy as np, gymnasium as gym
from dotenv import load_dotenv
from DQL import DeepQLearningAgent
from Nets import DQN, DCQN
from collections import deque


exp_config_cp = {
    "env_name": "CartPole-v1",
    "lr": 3e-4,
    "loss_fn": "huber",  # "mse" or "huber"
    "batch_size": 32,
    "gamma": 0.99,
    "tau": 0.005,
    "double": True,
    "episodes": 1000,
    "memory_size": 10000,
    "epsilon": 1.0,
    "epsilon_min": 0.3,
    "epsilon_decay": 0.002,
    "max_steps": 500,
    "start_training": 1000,
    "N": 50,
    "M": 10,
    "seed": 2112,
    "save_path": "Lab_3/models/",
    "save_model": True
}

exp_config_ll = {
    "env_name": "LunarLander-v3",
    "lr": 3e-4,
    "loss_fn": "huber",  # "mse" or "huber"
    "batch_size": 32,
    "gamma": 0.99,
    "tau": 0.005,
    "double": True,
    "episodes": 2000,
    "memory_size": 100000,
    "epsilon": 1.0,
    "epsilon_min": 0.3,
    "epsilon_decay": 0.002,
    "max_steps": 1000,
    "start_training": 10000,
    "N": 50,
    "M": 10,
    "seed": 2112,
    "save_path": "Lab_3/models",
    "save_model": True
}

loss_fn = {
    "mse": torch.nn.MSELoss(),
    "huber": torch.nn.SmoothL1Loss()
}

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_conf():
    load_dotenv()  # carica il file .env nella working directory
    api_key = os.environ.get("API_KEY")
    if api_key is None:
        raise ValueError("API_KEY non trovata: controlla il file .env")
    return api_key

def create_env(env_name, seed=None):
    env = gym.make(env_name)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env

def create_agent(env, config):
     device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
     state_size = env.observation_space.shape[0]
     action_size = env.action_space.n
     
     policy_net = DQN(state_size, action_size)
     target_net = DQN(state_size, action_size)
     target_net.eval()
     
     optimizer = torch.optim.Adam(policy_net.parameters(), lr=config["lr"])
     memory = deque(maxlen=config["memory_size"])
     
     agent = DeepQLearningAgent(
          policy_net=policy_net,
          target_net=target_net,
          optimizer=optimizer,
          env=env,
          memory=memory,
          loss_fn=loss_fn[config["loss_fn"]],
          batch_size=config["batch_size"],
          gamma=config["gamma"],
          tau=config["tau"],
          double=config["double"], 
          device=device
     )
     return agent

def exp(agent, config, save_model=False):
    epsilon = config["epsilon"]
    epsilon_min = config["epsilon_min"]
    epsilon_decay = config["epsilon_decay"]
    max_steps = config["max_steps"]
    start_training = config["start_training"]
    N = config["N"]
    M = config["M"]

    run = wandb.init(
        project="DeepQLearning",
        name=f"{agent.env.spec.id}_{config['loss_fn']}_seed{config['seed']}",
        config=config,
    )

    agent.DeepQLearning(
        episodes=config["episodes"],
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        max_steps=max_steps,
        start_training=start_training,
        N=N,
        M=M,
        run=run,
        seed=config["seed"])
    
    if save_model:
        policy_path = config["save_path"]+f"policy_net_{agent.env.spec.id}_{config['loss_fn']}_seed{config['seed']}.pth"
        target_path = config["save_path"]+f"target_net_{agent.env.spec.id}_{config['loss_fn']}_seed{config['seed']}.pth"
        agent.save_model(policy_path, target_path)
        print(f"Modelli salvati: {policy_path}, {target_path}")
    run.finish()
    
def main():
    print("Starting experiment")
    api_key = load_conf()
    wandb.login(key=api_key)
    exp_config = exp_config_ll  # or exp_config_ll for LunarLander

    set_seed(exp_config["seed"])

    env = create_env(exp_config["env_name"], seed=exp_config["seed"])
    agent = create_agent(env, exp_config)
    exp(agent, config=exp_config, save_model=exp_config["save_model"])


if __name__ == "__main__":
     main()