# Gymnasium Examples
Some simple examples of Gymnasium environments and wrappers.
For some explanations of these examples, see the [Gymnasium documentation](https://gymnasium.farama.org).

### Environments
This repository hosts the examples that are shown [on the environment creation documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).
- `ur5_setpoint`: Implementation

### Wrappers
This repository hosts the examples that are shown [on wrapper documentation](https://gymnasium.farama.org/api/wrappers/).
- `ClipReward`: A `RewardWrapper` that clips immediate rewards to a valid range


## Installation

To install it into your new environment, run the following commands in UR5_coppelia directory:

```{shell}
pip install -e .
```

## Usage
Once you have installed the package in virtual environment, head to your project and use the following commands:
```{python}
import gymnasium
import UR5_coppelia # Mandatory, this step will register the environment
env = gym.make('UR5_coppelia/UR5-v0') # To change this registered name, head to __init__.py
```

