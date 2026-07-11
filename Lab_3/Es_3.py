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
    "batch_size": 64,
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
    "save_path": "Lab_3/models/",
    "save_model": True
}

exp_config_cr = {
    "env_name": "CarRacing-v3",
    "num_envs": 8,                
    "lr": 3e-4,
    "loss_fn": "huber",
    "batch_size": 128,
    "gamma": 0.99,
    "tau": 0.005,
    "double": True,
    "episodes": 5000,
    "memory_size": 100000,
    "epsilon": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 3e-6,         
    "max_steps": 1000,
    "start_training": 10000,
    "N": 50, "M": 10,
    "seed": 2112,
    "save_path": "Lab_3/models/",
    "save_model": True,
}

loss_fn = {"mse": torch.nn.MSELoss(), "huber": torch.nn.SmoothL1Loss()}

def set_seed(seed):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

def load_conf():
    load_dotenv()
    api_key = os.environ.get("API_KEY")
    if api_key is None:
        raise ValueError("API_KEY non trovata: controlla il file .env")
    return api_key

def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def build_env(env_name):
    """Costruisce UN env col preprocessing adatto (senza seedare)."""
    if env_name == "CarRacing-v3":
        env = gym.make(env_name, continuous=False)                 # azioni discrete (5)
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=False) # 96x96
        env = gym.wrappers.ResizeObservation(env, (84, 84))          # 84x84
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)  # (4, 84, 84)
    else:
        env = gym.make(env_name)
    return env

def create_single_env(env_name, seed=None):
    env = build_env(env_name)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env

def create_vec_env(env_name, num_envs, seed=None):
    envs = gym.vector.AsyncVectorEnv([lambda: build_env(env_name) for _ in range(num_envs)]) #l'esplicitazine è necessaria per fare il wrap di tutti gli env
    #il metodo base gym.make_vec permette di passare i wrapper ma almeno in questo modo posso rendere esplicito l'ordine di applicazione dei wrapper
    if seed is not None:
        envs.reset(seed=seed)
    return envs

def build_nets(obs_shape, action_size):
    """Sceglie DCQN (immagini, shape 3D) o DQN (stati vettoriali, shape 1D)."""
    if len(obs_shape) == 3:
        return DCQN(obs_shape, action_size), DCQN(obs_shape, action_size)
    return DQN(obs_shape[0], action_size), DQN(obs_shape[0], action_size)

def create_agent(config, vectorized, envs=None, eval_env=None, serial_env=None):
    device = pick_device()
    if vectorized:
        obs_shape = envs.single_observation_space.shape
        action_size = envs.single_action_space.n
    else:
        obs_shape = serial_env.observation_space.shape
        action_size = serial_env.action_space.n

    policy_net, target_net = build_nets(obs_shape, action_size)
    target_net.eval()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=config["lr"])
    memory = deque(maxlen=config["memory_size"])

    common = dict(policy_net=policy_net, target_net=target_net, optimizer=optimizer,
                  memory=memory, loss_fn=loss_fn[config["loss_fn"]],
                  batch_size=config["batch_size"], gamma=config["gamma"],
                  tau=config["tau"], double=config["double"], device=device)

    if vectorized:
        return DeepQLearningMultiAgent(envs=envs, eval_env=eval_env, **common)
    return DeepQLearningAgent(env=serial_env, **common)

def exp(agent, config, save_model=False):
    run = wandb.init(
        project="DeepQLearning",
        name=f"{config['env_name']}_{config['loss_fn']}_seed-{config['seed']}",  # nome dal config, robusto per vec
        config=config,
    )
    agent.DeepQLearning(
        episodes=config["episodes"], epsilon=config["epsilon"],
        epsilon_min=config["epsilon_min"], epsilon_decay=config["epsilon_decay"],
        max_steps=config["max_steps"], start_training=config["start_training"],
        N=config["N"], M=config["M"], run=run, seed=config["seed"])

    if save_model:
        suffix = f"{config['env_name']}_{config['loss_fn']}_seed{config['seed']}.pth"
        os.makedirs(config["save_path"], exist_ok=True)
        policy_path = os.path.join(config["save_path"], f"policy_net_{suffix}")
        target_path = os.path.join(config["save_path"], f"target_net_{suffix}")
        agent.save_model(policy_path, target_path)
        print(f"Modelli salvati: {policy_path}, {target_path}")
    run.finish()

def main():
    print("Starting experiment")
    api_key = load_conf()
    wandb.login(key=api_key)

    exp_config = exp_config_cr
    set_seed(exp_config["seed"])

    vectorized = (exp_config["env_name"] == "CarRacing-v3")
    if vectorized:
        envs = create_vec_env(exp_config["env_name"], exp_config["num_envs"], seed=exp_config["seed"])
        eval_env = create_single_env(exp_config["env_name"], seed=exp_config["seed"])
        agent = create_agent(exp_config, vectorized=True, envs=envs, eval_env=eval_env)
    else:
        serial_env = create_single_env(exp_config["env_name"], seed=exp_config["seed"])
        agent = create_agent(exp_config, vectorized=False, serial_env=serial_env)

    exp(agent, config=exp_config, save_model=exp_config["save_model"])

if __name__ == "__main__":
    main()