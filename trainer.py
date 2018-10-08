import os
from unityagents import UnityEnvironment

from agents import *
from qnetwork import *
from replay_buffer import *
import matplotlib.pyplot as plt
import json


ENVIRONMENT_BINARY = os.environ['DRLUD_P1_ENV']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(agent, environment, n_episodes=50, max_t=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
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

def prepare_prio_agent(environment, agent_config, seed=0):
    return DQNPAgent(agent_config=agent_config,
                     network_builder=lambda: QNetwork(
                         agent_config.state_size, 
                         agent_config.action_size, 
                         agent_config.hidden_neurons, 
                         seed
                     ).to(device),
                     replay_buffer=PrioritizedReplayBuffer(
                         agent_config.action_size, 
                         agent_config.buffer_size, 
                         agent_config.batch_size, 
                         0.5,
                         1e-7,
                         1e-10,
                         device, 
                         seed
                     ),
                     device=device,
                     seed=seed)


def run_training_sessions(agent_factory, lr, update_interval, batch_size, buffer_size, gamma, tau, times=1):
    env = prepare_environment()
    hidden_neurons = 36
    (brain_name, action_size, state_size) = infer_environment_properties(env)
    agent_config=AgentConfig(
        brain_name, state_size, action_size, lr, hidden_neurons, update_interval, batch_size, buffer_size, gamma, tau)
    scores = []
    for seed in range(times):
        agent = agent_factory(env, agent_config, seed)
        scores.append(train(agent, env, solution_score=100.0))
    env.close()
    return scores

if __name__ == "__main__":
    
    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64  # minibatch size
    GAMMA = 0.99  # discount factor
    TAU = 1e-3  # for soft update of target parameters
    LR = 5e-4  # learning rate
    UPDATE_EVERY = 4  # how often to update the network

    with open('ddqn_training.txt', 'w') as fp:
        json.dump(run_training_sessions(prepare_prio_agent, LR, UPDATE_EVERY,
                                        BATCH_SIZE, BUFFER_SIZE, GAMMA, TAU,
                                        times=1), fp)

