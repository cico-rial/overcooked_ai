# Overcooked-AI 🧑‍🍳🤖

<p align="center">
  <!-- <img src="overcooked_ai_js/images/screenshot.png" width="350"> -->
  <img src="./images/layouts.gif" width="100%"> 
  <i>5 of the available layouts. New layouts are easy to hardcode or generate programmatically.</i>
</p>

## Introduction 🥘

Overcooked-AI is a benchmark environment for fully cooperative human-AI task performance, based on the wildly popular video game [Overcooked](http://www.ghosttowngames.com/overcooked/).

The goal of the game is to deliver soups as fast as possible. Each soup requires placing up to 3 ingredients in a pot, waiting for the soup to cook, and then having an agent pick up the soup and delivering it. The agents should split up tasks on the fly and coordinate effectively in order to achieve high reward.

This repo has been forked from [here](https://github.com/HumanCompatibleAI/overcooked_ai) to implement and test the effectiveness of Policy Gradient Methods with a custom implementation in Tensorflow Keras.

## Installation ☑️

### Building from source 🔧

Clone the repo 
```
git clone https://github.com/cico/overcooked_ai.git
```

Create venv using uv (necessary for compatibility)
```
uv venv
```

Install the dependencies
```
uv sync
```

Install tensorflow through **uv pip**(couldn't be possible to add it to the dependencies)
```
uv pip install tensorflow
```

Activate the virtual environment
```
.venv\Scripts\activate
```

## Code Structure Overview 🗺

The relevant files inside overcooked_ai/ for the project are:

- `training.py`: main python file for training the agents
- `display_trained_agents.py`: python file for testing out the agents and graphically display a game
- `report.pdf`: pdf file describing the project work.
- `info/`: folder containing .json files describing the experiment performed.
- `networks/`: folder containing the weights of the neural networks for each experiment.

## Training the agents 

To train the agents you can launch the program with the following command:

```
python training.py --exp-name ppo_exp_1 --algorithm ppo --seed 42 --num-episodes 2000 
```

To set specific hyperparameters, you can specify the following options:

```
usage: training.py [-h] [--exp-name EXP_NAME] [--seed SEED] [--algorithm {ac,ppo}] [--shared-agent SHARED_AGENT] [--num-episodes NUM_EPISODES]
                   [--num-epochs NUM_EPOCHS] [--batch-size BATCH_SIZE] [--prev-action PREV_ACTION] [--delivery-reward DELIVERY_REWARD]
                   [--gamma GAMMA] [--lr-w LR_W] [--lr-t LR_T] [--ppo-epsilon PPO_EPSILON] [--entropy ENTROPY] [--load-weights LOAD_WEIGHTS]
                   [--run-on-colab RUN_ON_COLAB]

options:
  -h, --help            show this help message and exit
  --exp-name EXP_NAME   the name of this experiment. Default = "dummy_experiment".
  --seed SEED           set the seed for reproducibility of the experiment. Default = 42
  --algorithm {ac,ppo}  the name of the algorithm to use. Default = "ppo"
  --shared-agent SHARED_AGENT
                        whether to use the same critic for both agents. Default = True
  --num-episodes NUM_EPISODES
                        number of episodes to train the agent on. Default = 1000
  --num-epochs NUM_EPOCHS
                        number of epochs to train the agent on with SGD. Default = 2
  --batch-size BATCH_SIZE
                        batch size of the training with SGD. Default = 20
  --prev-action PREV_ACTION
                        number of actions prior to actual reward. Default = 5
  --delivery-reward DELIVERY_REWARD
                        reward associated to soup delivery. Default = 5
  --gamma GAMMA         discount factor for rewards and future state value estimations. Default = 0.95
  --lr-w LR_W           learning rate for the critic. Default = 1e-5
  --lr-t LR_T           learning rate for the actor. Default = 1e-6
  --ppo-epsilon PPO_EPSILON
                        epsilon for clipping in PPO. Default = 0.05
  --entropy ENTROPY     whether you want to use entropy-loss. Default = False
  --load-weights LOAD_WEIGHTS
                        whether you want to load previous weights. Default = False
  --run-on-colab RUN_ON_COLAB
                        whether you are running it from colab. Default = False
```

## Testing the agents 

To test your trained agents, you can run the following command:

```
python display_trained_agents.py --exp-name ppo_exp_1 --seed 42 --num-episodes 10 
```

You can specify the following options:
```
usage: display_trained_agents.py [-h] [--exp-name EXP_NAME] [--seed SEED] [--num-episodes NUM_EPISODES] [--refresh-rate REFRESH_RATE]

options:
  -h, --help            show this help message and exit
  --exp-name EXP_NAME   the name of the experiment from which weights will be loaded. Default = "dummy_experiment".
  --seed SEED           set the seed for reproducibility of the experiment. Default = 42.
  --num-episodes NUM_EPISODES
                        number of episodes for which to compute the average reward. Default = 10.
  --refresh-rate REFRESH_RATE
                        refresh-rate for displaying the episode. Default = 250 (ms).
```
