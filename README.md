Not much to see here.

Solution to the first project of the deep reinforcement learning nanodegree at Udacity.

## Problem definition

The reinforcement learning agent is travelling through a 2d space filled with blue and yellow bananas. The agent is expected to gather the banana if it is yellow or avoid the blue ones. The agent receives a positive reward for every yellow banana it gathers and a negative reward for every blue banana. The state the agent receives comprises of its speed as well as the raytraced positions of the nearest bananas in the field of view, the size of the state space is 37. The agent is able to move forwards and backwards as well as turn left and right, thus the size of the action space is 4. The minimal expected performance of the agent after training is a score of +13 over 100 consecutive episodes.

## Usage

Please ensure you have [Pipenv](https://pipenv.readthedocs.io/en/latest/) installed. Clone the repository and use `pipenv --three install` to create yourself an environment to run the code in. Otherwise just install the packages mentioned in Pipfile.

Due to the transitive dependency to tensorflow that comes from unity ml-agents and the [bug](https://github.com/pypa/pipenv/issues/1716) causing incompatibility to jupyter you might want to either drop the jupyter from the list of dependencies or run `pipenv --three install --skip-lock` to overcome it.

To see the performance of agents using DQN and DDQN with different sets of hyperparameters (lr, batch_size, etc) as well training code example please check the [parameter search notebook](Training_parameter_search.ipynb).

## Implementation details

The neural network architecture is defined in the `qnetwork.py` file. With some trial and error I found the smallest network that gave me reasonable performance in terms of number of episodes to reach a certain threshold.

Implementation of DQN and DDQN agents is located inside of `agents.py`. Both of them rely on the same neural network architecture as well as the replay buffer which is in `replay_buffer.py`.

## Future improvements

TBD

Notes:
* State prediction module
* p.6 from prioritized replay paper
