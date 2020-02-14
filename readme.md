Deep Q Learning Algorithms in pytorch

# Algorithms

DQN, Double DQN, Dual DQN, Dual Double DQN 

- python 3.6
- [gym](https://github.com/openai/gym#installation)
- [pytorch](https://github.com/pytorch/pytorch#from-source) (1.x)

# Usage

To train a model:

```
# see main.py for args details
$ python main.py 

```

The model is defined in `model.py`

The algorithm is defined in `dqn_agent.py` and `ddqn_agent.py`

The running script and hyper-parameters are defined in `main.py`

# Results

Compare the convergence speed of various algorithms: Dual Double DQN > Dual DQN > Double DQN > DQN



