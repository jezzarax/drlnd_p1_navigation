Not much to see here.

Solution to the first project of the deep reinforcement learning nanodegree at Udacity.

## Usage

Please ensure you have [Pipenv](https://pipenv.readthedocs.io/en/latest/) installed. Clone the repository and use `pipenv --three install` to create yourself an environment to run the code in. Otherwise just install the packages mentioned in Pipfile.

## Implementation details

The neural network architecture is defined in the `qnetwork.py` file. With some trial and error I found the smallest network that gave me reasonable performance in terms of number of episodes to reach a certain threshold.

Implementation of DQN and DDQN agents is located inside of `agents.py`. Both of them rely on the same neural network architecture as well as the replay buffer which is in `replay_buffer.py`.

## Results

TBD
