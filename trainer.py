import os
from unityagents import UnityEnvironment

from agents import *
from qnetwork import *
from replay_buffer import *
import matplotlib.pyplot as plt
from collections import namedtuple
import json, sys, logging


logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    stream=sys.stdout
)

ENVIRONMENT_BINARY = os.environ['DRLUD_P1_ENV']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(agent, environment, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
          solution_score=100.0):
    """Deep Q-Learning.
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        env_info = environment.reset(train_mode=True)[agent.name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = environment.step(action)[agent.name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores.append(score)
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        last_100_steps_mean = np.mean(scores[-100:])
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, last_100_steps_mean), end="")
        if i_episode % 1000 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, last_100_steps_mean))

        if last_100_steps_mean >= solution_score:
            print(f'\nEnvironment solved in {i_episode:d} episodes!\tAverage Score: {last_100_steps_mean:.2f}')
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores


def prepare_environment():
    return UnityEnvironment(file_name=ENVIRONMENT_BINARY)

def infer_environment_properties(environment):
    brain_name = environment.brain_names[0]
    brain = environment.brains[brain_name]
    action_size = brain.vector_action_space_size
    env_info = environment.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    state_size = len(state)
    return (brain_name, action_size, state_size)


def prepare_dqn_agent(environment, agent_config, seed=0):
    return DQNAgent(agent_config=agent_config,
                     network_builder=lambda: QNetwork(
                         agent_config.state_size, 
                         agent_config.action_size, 
                         agent_config.hidden_neurons, 
                         seed
                     ).to(device),
                     replay_buffer=ReplayBuffer(
                         agent_config.action_size, 
                         agent_config.buffer_size, 
                         agent_config.batch_size, 
                         device, 
                         seed
                     ),
                     device=device,
                     seed=seed)


def prepare_ddqn_agent(environment, agent_config, seed=0):
    return DDQNAgent(agent_config=agent_config,
                     network_builder=lambda: QNetwork(
                         agent_config.state_size, 
                         agent_config.action_size, 
                         agent_config.hidden_neurons, 
                         seed
                     ).to(device),
                     replay_buffer=ReplayBuffer(
                         agent_config.action_size, 
                         agent_config.buffer_size, 
                         agent_config.batch_size, 
                         device, 
                         seed
                     ),
                     device=device,
                     seed=seed)

def prepare_dueling_agent(environment, agent_config, seed=0):
    return DDQNAgent(agent_config=agent_config,
                     network_builder=lambda: DuelQNetwork(
                         agent_config.state_size, 
                         agent_config.action_size, 
                         agent_config.hidden_neurons,
                         16,
                         64, 
                         seed
                     ).to(device),
                     replay_buffer=ReplayBuffer(
                         agent_config.action_size, 
                         agent_config.buffer_size, 
                         agent_config.batch_size, 
                         device, 
                         seed
                     ),
                     device=device,
                     seed=seed)



def run_training_session(agent_factory, lr, update_interval, batch_size, buffer_size, gamma, tau, hidden_neurons, times=1):
    env = prepare_environment()
    (brain_name, action_size, state_size) = infer_environment_properties(env)
    agent_config=AgentConfig(
        brain_name, state_size, action_size, lr, hidden_neurons, update_interval, batch_size, buffer_size, gamma, tau)
    scores = []
    for seed in range(times):
        agent = agent_factory(env, agent_config, seed)
        scores.append(train(agent, env, solution_score=100.0))
    env.close()
    return scores

hparm = namedtuple("hparm", ["lr", "update_rate", "batch_size", "memory_size", "gamma", "tau", "times", "hidden_layer_size", "algorithm"])

path_prefix = "./hp_search_results/"

simulation_hyperparameter_reference = {
    1:  hparm(5e-4, 4,  64,  int(1e5), 0.99, 1e-3, 10,  36, "ddqn"      ),
    2:  hparm(5e-3, 4,  64,  int(1e5), 0.99, 1e-3, 10,  36, "ddqn"      ),
    3:  hparm(5e-2, 4,  64,  int(1e5), 0.99, 1e-3, 10,  36, "ddqn"      ),
    4:  hparm(5e-4, 8,  64,  int(1e5), 0.99, 1e-3, 10,  36, "ddqn"      ),
    5:  hparm(5e-4, 16, 64,  int(1e5), 0.99, 1e-3, 10,  36, "ddqn"      ),
    6:  hparm(5e-4, 4,  64,  int(1e5), 0.99, 1e-2, 10,  36, "ddqn"      ),
    7:  hparm(5e-4, 4,  64,  int(1e5), 0.99, 5e-2, 10,  36, "ddqn"      ),
    8:  hparm(5e-5, 4,  64,  int(1e5), 0.99, 1e-3, 10,  36, "ddqn"      ),
    9:  hparm(5e-4, 4,  64,  int(1e4), 0.99, 1e-3, 10,  36, "ddqn"      ),
    10: hparm(5e-4, 4,  64,  int(1e3), 0.99, 1e-3, 10,  36, "ddqn"      ),
    11: hparm(5e-4, 4,  32,  int(1e5), 0.99, 1e-3, 10,  36, "ddqn"      ),
    12: hparm(5e-4, 4,  16,  int(1e5), 0.99, 1e-3, 10,  36, "ddqn"      ),
    13: hparm(5e-4, 4,  128, int(1e5), 0.99, 1e-3, 10,  36, "ddqn"      ),
    14: hparm(5e-4, 8,  64,  int(1e5), 0.99, 1e-3, 10,  36, "ddqn"      ),
    15: hparm(5e-4, 4,  64,  int(1e5), 0.99, 1e-3, 100, 36, "ddqn"      ),
    16: hparm(5e-6, 4,  64,  int(1e5), 0.99, 1e-3, 100, 36, "ddqn"      ),
    17: hparm(5e-4, 4,  64,  int(1e5), 0.99,  0.5, 10,  36, "ddqn"      ),
    18: hparm(5e-4, 4,  64,  int(1e5), 0.99, 1e-3, 10,  36, "dqn"       ),
    19: hparm(5e-4, 2,  64,  int(1e5), 0.99, 1e-3, 10,  36, "ddqn"      ),
    20: hparm(5e-4, 1,  64,  int(1e5), 0.99, 1e-3, 10,  36, "ddqn"      ),
    21: hparm(5e-4, 2,  64,  int(1e5), 0.99, 1e-3, 10,  36, "dqn"       ),
    22: hparm(5e-4, 8,  64,  int(1e5), 0.99, 1e-3, 10,  36, "dqn"       ),
    23: hparm(5e-4, 4,  64,  int(1e5), 0.99, 1e-3, 10,  36, "dueling"   ),
    24: hparm(5e-4, 4,  64,  int(1e5), 0.99, 1e-3, 10,  36, "dueling"   ),
    25: hparm(5e-4, 4,  64,  int(1e5), 0.99, 1e-3, 10,  36, "dueling"   )
}

algorithm_factories = {
    "dqn": prepare_dqn_agent,
    "ddqn": prepare_ddqn_agent,
    "dueling": prepare_dueling_agent
}

def ensure_training_run(id: int, parm: hparm):
    if(os.path.isfile(f"{path_prefix}set{id}_results.json")):
        logging.info(f"Skipping configuration {id} with following parameters {parm}")
    else:
        logging.info(f"Running {id} with following parameters {parm}")
        run_result = run_training_session(
            algorithm_factories[parm.algorithm],
            parm.lr,
            parm.update_rate,
            parm.batch_size,
            parm.memory_size,
            parm.gamma,
            parm.tau,
            parm.hidden_layer_size,
            parm.times
        )
        with open(f"{path_prefix}set{id}_results.json", "w") as fp:
            json.dump(run_result, fp)

if __name__ == "__main__":
    for parm_id in simulation_hyperparameter_reference:
        ensure_training_run(parm_id, simulation_hyperparameter_reference[parm_id])
