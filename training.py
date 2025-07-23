from utility.utility import set_seed_for_reproducibility, Policy, ValueFunctionApproximator, MyAgent
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from typing import Tuple, List, Dict
import sys
import argparse
import json
import time
import os
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    """
    Parse command line arguments for the experiment configuration.
    
    Returns:
        args (Namespace): Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="dummy_experiment", help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42, help="set the seed for reproducibility of the experiment")
    parser.add_argument("--algorithm", type=str, default='ppo', choices=['ac', 'ppo'], help="the name of the algorithm to use")
    parser.add_argument("--shared-agent", type=lambda x: (str(x).lower() == "true"), default=True, help="whether to use the same critic for both agents")
    parser.add_argument("--num-episodes", type=int, default=1000, help="number of episodes to train the agent on")
    parser.add_argument("--num-epochs", type=int, default=2, help="number of epochs to train the agent on with SGD")
    parser.add_argument("--batch-size", type=int, default=20, help="batch size of the training with SGD")
    parser.add_argument("--prev-action", type=int, default=5, help="number of actions to reward prior to actual reward (rewarding a trajectory)")
    parser.add_argument("--delivery-reward", type=int, default=5, help="reward associated to soup (< 3 onions) delivery")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor for rewards and future state value estimations")
    parser.add_argument("--lr-w", type=float, default=1e-5, help="learning rate for the critic")
    parser.add_argument("--lr-t", type=float, default=1e-6, help="learning rate for the actor")
    parser.add_argument("--ppo-epsilon", type=float, default=0.05, help="epsilon for clipping in PPO.")
    parser.add_argument("--entropy", type=lambda x: (str(x).lower() == "true"), default=False, help="whether you want to use entropy-loss")
    parser.add_argument("--load-weights", type=lambda x: (str(x).lower() == "true"), default=False, help="whether you want to load previous weights")
    parser.add_argument("--run-on-unix-like", type=lambda x: (str(x).lower() == "true"), default=True, help="whether you are running it from a unix like system (e.g. linux, macos)")

    args = parser.parse_args()

    return args


def check_if_continue():
    """
    Check if the user wants to continue the experiment even if there are warnings or errors.
    """
    command = input("Do you want to continue anyway? (y/n) ")
    if "y" not in command:
        exit("Exiting.")


def load_weights():
    """
    Load the weights of the neural networks if they exist.
    If the weights do not exist, training will start from scratch.
    """
    global LOAD_WEIGHTS
    
    if LOAD_WEIGHTS:
        if SHARED_AGENT:
            condition = os.path.exists(PATH_ACTOR) and os.path.exists(PATH_CRITIC)
        else:
            condition = os.path.exists(PATH_ACTOR) and os.path.exists(PATH_CRITIC) and os.path.exists(PATH_SECOND_CRITIC)

        if condition:
            print("")
            print("Loading previous weights...")
            try:
                actor.load_weights(PATH_ACTOR)
                critic.load_weights(PATH_CRITIC)
                if not SHARED_AGENT:
                    second_critic.load_weights(PATH_SECOND_CRITIC)
                print("Weights successfully loaded.")
            except:
                print("Error: loading weights has failed.")
                check_if_continue()
                print("Overriding weights.")
                print("")
                LOAD_WEIGHTS = False
            
        else:
            print("")
            print("Warning: previous weights not found.")
            check_if_continue()
            print("Starting from scratch.")
            print("")
            LOAD_WEIGHTS = False
    else:
        condition = os.path.exists(PATH_ACTOR) or os.path.exists(PATH_CRITIC) or os.path.exists(PATH_SECOND_CRITIC)

        if condition:
            print("")
            print("Warning: There exist already weights with this name.")
            check_if_continue()
            print("Overriding weights.")
            print("")
            LOAD_WEIGHTS = False


def load_experiment_info():
    """
    Load the experiment's info from a json file or create a new one if it doesn't exist.

    Returns:
        experiment_info (dict): the experiment's info.
    """
    try:
        if os.path.exists(PATH_EXPERIMENT_INFO) and LOAD_WEIGHTS:
            print("Loading previous experiment's info...")
            with open(PATH_EXPERIMENT_INFO, 'r') as f:
                experiment_info = json.load(f)
                print("Experiment's info successfully loaded.")
                print("")
        else:
            experiment_info = {
                "exp_name": EXP_NAME, 
                "seed": SEED, 
                "algorithm": ALGORITHM, 
                "shared_agent": SHARED_AGENT, 
                "load_weights": LOAD_WEIGHTS,
                "lr_critic": LR_CRITIC, 
                "lr_actor": LR_ACTOR, 
                "number_of_episodes": NUMBER_OF_EPISODES, 
                "number_of_epochs": NUMBER_OF_EPOCHS,
                "batch_size": BATCH_SIZE, 
                "prev_action_to_reward": PREV_ACTION_TO_REWARD,  
                "delivery_reward": DELIVERY_REWARD, 
                "gamma": GAMMA,
                "ppo_epsilon": PPO_EPSILON,
                "entropy_loss": ENTROPY,
                "average_reward" : 0,
                "best_avg" : 0,
                "avg_reward_list" : [],
                "stats" : {"soup_delivery" : [],
                           "useful_onion_pickup" : [],
                           "potting_onion" : [],
                           "useful_dish_pickup": [],
                           "soup_pickup": []
                           }
            }

    except:
        print(f"Error: unable to load experiment's info.")
        check_if_continue()
        print("Overriding experiment's info.")
        print("")

    return experiment_info


def save_experiment_info(experiment_info: Dict):
    """ 
    Save the experiment's info to a json file.
    If the file already exists, it will be overwritten.

    Args:
        experiment_info (dict): the experiment's info to save.
    Returns:
        None
    """
    try:
        with open(PATH_EXPERIMENT_INFO, 'w') as f:
                json.dump(experiment_info, f)
        print(f"Experiment's info successfully saved at {PATH_EXPERIMENT_INFO}.")
    except:
        print("Error: unable to save experiment's info.")


def save_weights():
    """
    Save the weights of the neural networks to a file.
    If the file already exists, it will be overwritten.
    """
    try:
        critic.save_weights(PATH_CRITIC)
        actor.save_weights(PATH_ACTOR)
        if not SHARED_AGENT:
            second_critic.save_weights(PATH_SECOND_CRITIC)
        print("Weights successfully saved.")
    except:
        print("Error: unable to save weights.")


def get_old_policy():
    """
    Get the old policy for PPO training.

    Returns:
        old_policy (Policy): the old policy to use for PPO training.
    """
    if ALGORITHM == "ppo":
        old_policy = Policy(input_shape=input_shape, num_actions=Action.NUM_ACTIONS, optimizer=None, entropy_loss=ENTROPY)
    else:
        old_policy = None
    return old_policy


def get_second_critic():
    """
    Get the second critic for the second agent.
    If SHARED_AGENT is True, the critic is shared.

    Returns:
        second_critic (ValueFunctionApproximator): the second critic to use for the shared agent.
    """
    if not SHARED_AGENT:
        second_critic = ValueFunctionApproximator(input_shape=input_shape, optimizer=Adam(learning_rate=LR_CRITIC))
    else:
        second_critic = critic
    return second_critic


if __name__ == "__main__":
    args = parse_args()

    # algorithm specifications
    EXP_NAME = args.exp_name
    SEED = args.seed
    ALGORITHM = args.algorithm
    SHARED_AGENT = args.shared_agent
    LOAD_WEIGHTS = args.load_weights
    ENTROPY = args.entropy
    RUN_ON_UNIX = args.run_on_unix_like

    # hyperparameters
    LR_CRITIC = args.lr_w
    LR_ACTOR = args.lr_t
    NUMBER_OF_EPISODES = args.num_episodes
    NUMBER_OF_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    PREV_ACTION_TO_REWARD = args.prev_action
    DELIVERY_REWARD = args.delivery_reward
    GAMMA = args.gamma
    PPO_EPSILON = args.ppo_epsilon

    # paths for saving and loading weights and info for the experiment
    PATH_ACTOR = os.path.join("networks", "actor", "actor_" + EXP_NAME + ".weights.h5") 
    PATH_CRITIC = os.path.join("networks","critic", "critic_" + EXP_NAME + ".weights.h5") 
    PATH_SECOND_CRITIC = os.path.join("networks","second_critic", "second_critic_" + EXP_NAME + ".weights.h5")
    PATH_EXPERIMENT_INFO = os.path.join("info", EXP_NAME + ".json") 

    if RUN_ON_UNIX:
        sys.path.append('/content/overcooked_ai/src') # necessary to import the modules from the src folder 

    print("")
    print("EXPERIMENT INFO.")
    print(f"Experiment Name: {EXP_NAME}")
    print(f"Seed: {SEED}")
    print(f"Algorithm: {ALGORITHM}")
    print(f"Shared Agent: {SHARED_AGENT}")
    print(f"Loading previous weights: {LOAD_WEIGHTS}")
    print(f"Entropy Loss: {ENTROPY}")
    print(f"Running on unix-like system: {RUN_ON_UNIX}")
    print(f"Number of Episodes: {NUMBER_OF_EPISODES}")
    print(f"Number of Epochs: {NUMBER_OF_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Previous Action to Reward: {PREV_ACTION_TO_REWARD}")
    print(f"Delivery Reward: {DELIVERY_REWARD}")
    print(f"Gamma: {GAMMA}")
    print(f"PPO Epsilon: {PPO_EPSILON}")
    print(f"Learning Rate Critic: {LR_CRITIC}")
    print(f"Learning Rate Actor: {LR_ACTOR}")
    print("")
    print(f"Weights will be saved and loaded from the following paths:")
    print(f"Path actor: {PATH_ACTOR}")
    print(f"Path critic: {PATH_CRITIC}")
    if not SHARED_AGENT:
        print(f"Path second critic: {PATH_SECOND_CRITIC}")
    print("")

    set_seed_for_reproducibility(SEED)

    # initializing the environment
    number_of_frames = 400
    layout_name = "cramped_room"
    base_mdp = OvercookedGridworld.from_layout_name(layout_name=layout_name)
    base_env = OvercookedEnv.from_mdp(base_mdp, info_level=0, horizon=number_of_frames)
    env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

    input_shape = env.observation_space._shape

    # initializing the actor and critic
    actor = Policy(
        input_shape=input_shape,
        num_actions=Action.NUM_ACTIONS,
        optimizer=Adam(learning_rate=LR_ACTOR),
        entropy_loss=ENTROPY,
        epsilon=PPO_EPSILON
        )

    critic = ValueFunctionApproximator(
        input_shape=input_shape,
        optimizer=Adam(learning_rate=LR_CRITIC)
        )
    
    old_policy = get_old_policy()

    second_critic = get_second_critic()
    
    load_weights()

    # creating the agents
    agent_1 = MyAgent(
        actor=actor,
        old_policy=old_policy,
        critic=critic,
        idx=0,
        base_env=base_env
    )
    agent_2 = MyAgent(
        actor=actor,
        old_policy=old_policy,
        critic=second_critic,
        idx=1,
        base_env=base_env
    )

    # setting up the experiment info
    experiment_info = load_experiment_info()
        
    try:
        # episode rollout and training
        for episode in range(1, NUMBER_OF_EPISODES + 1):
            actions = []
            observations = []
            new_observations = []
            rewards = []

            t = 0
            obs = env.reset()
            obs = obs['both_agent_obs'] 
            
            done = False
            episodic_reward = 0

            start_episode = time.time()

            while not done:
                # getting the actions from the agents
                action_1_idx = agent_1.action(obs)
                action_2_idx = agent_2.action(obs)
                agent_1_action = Action.ACTION_TO_INDEX[action_1_idx[0]]
                agent_2_action = Action.ACTION_TO_INDEX[action_2_idx[0]]
                action = (agent_1_action, agent_2_action)

                actions.append(action)
                observations.append(obs)

                # performing the action and getting the results
                new_obs, reward, done, env_info = env.step(action)

                # calculating the rewards
                shaped_reward = sum(env_info['shaped_r_by_agent']) 
                shaped_reward_1 = env_info['shaped_r_by_agent'][0] 
                shaped_reward_2 = env_info['shaped_r_by_agent'][1]

                sparse_reward = reward 
                sparse_reward_1 = env_info['sparse_r_by_agent'][0]
                sparse_reward_2 = env_info['sparse_r_by_agent'][1]

                total_reward = reward + shaped_reward 
                total_reward_1 = shaped_reward_1 + sparse_reward_1
                total_reward_2 = shaped_reward_2 + sparse_reward_2

                episodic_reward += total_reward

                if SHARED_AGENT:
                    # appending the sum of the rewards if the critic is shared
                    rewards.append(total_reward)

                    # rewarding a successful trajectory of PREV_ACTION_TO_REWARD actions
                    if PREV_ACTION_TO_REWARD > 0 and total_reward > 0:
                        if t > PREV_ACTION_TO_REWARD:
                            for i in range(t-1, t-PREV_ACTION_TO_REWARD-1, -1):
                                rewards[i] += (GAMMA**(t-i))*total_reward
                        else:
                            for i in range(t-1,-1,-1):
                                rewards[i] += total_reward
                else:
                    # appending the individual rewards if the critic is NOT shared
                    rewards.append([total_reward_1, total_reward_2])
                    
                    # rewarding a successful trajectory of PREV_ACTION_TO_REWARD actions
                    if PREV_ACTION_TO_REWARD > 0:
                        if total_reward_1 > 0:
                            if t > PREV_ACTION_TO_REWARD:
                                for i in range(t-1, t-PREV_ACTION_TO_REWARD-1, -1):
                                    rewards[i][0] += (GAMMA**(t-i))*total_reward_1
                            else:
                                for i in range(t-1,-1,-1):
                                    rewards[i][0] += total_reward_1

                        if total_reward_2 > 0:
                            if t > PREV_ACTION_TO_REWARD:
                                for i in range(t-1, t-PREV_ACTION_TO_REWARD-1, -1):
                                    rewards[i][1] += (GAMMA**(t-i))*total_reward_2
                            else:
                                for i in range(t-1,-1,-1):
                                    rewards[i][1] += total_reward_2
            
                new_obs = new_obs['both_agent_obs']
                new_observations.append(new_obs)

                # update state
                obs = new_obs

                t += 1
            
            # getting some stats
            t_useful_onion_pickup = env_info.get('episode',{}).get('ep_game_stats',{}).get('useful_onion_pickup',[[],[]])
            useful_onion_pickup = sum([len(agent) for agent in t_useful_onion_pickup])

            t_potting_onion = env_info.get('episode',{}).get('ep_game_stats',{}).get('potting_onion',[[],[]])
            potting_onion = sum([len(agent) for agent in t_potting_onion])

            t_useful_dish_pickup = env_info.get('episode',{}).get('ep_game_stats',{}).get('useful_dish_pickup',[[],[]])
            useful_dish_pickup = sum([len(agent) for agent in t_useful_dish_pickup])

            t_soup_pickup = env_info.get('episode',{}).get('ep_game_stats',{}).get('soup_pickup',[[],[]])
            soup_pickup = sum([len(agent) for agent in t_soup_pickup])

            t_soup_delivery = env_info.get('episode',{}).get('ep_game_stats',{}).get('soup_delivery',[[],[]])
            soup_delivery = sum([len(agent) for agent in t_soup_delivery])

            # rewarding soup delivery (even if not 3-onions soup)
            if DELIVERY_REWARD > 0:
                for agent in range(len(t_soup_delivery)):
                    for delivery_timestep in t_soup_delivery[agent]:
                        for i in range(delivery_timestep, delivery_timestep-PREV_ACTION_TO_REWARD-1, -1):
                            if SHARED_AGENT:
                                rewards[i] += (GAMMA**(delivery_timestep-i))*DELIVERY_REWARD
                            else:
                                rewards[i][agent] += (GAMMA**(delivery_timestep-i))*DELIVERY_REWARD

            # computing the average episodic reward achieved so far
            epsiodes_so_far = len(experiment_info["avg_reward_list"])
            experiment_info['average_reward'] = 1/(epsiodes_so_far+1)*( episodic_reward + (epsiodes_so_far)*experiment_info['average_reward'])
            experiment_info["avg_reward_list"].append(round(experiment_info['average_reward'],2))

            # saving the stats
            experiment_info["stats"]['useful_onion_pickup'].append(useful_onion_pickup)
            experiment_info["stats"]['potting_onion'].append(potting_onion)
            experiment_info["stats"]['useful_dish_pickup'].append(useful_dish_pickup)
            experiment_info["stats"]['soup_pickup'].append(soup_pickup)
            experiment_info["stats"]['soup_delivery'].append(soup_delivery)
            
            end_episode = time.time()

            print(f"Episode [{episode:>3d}] terminated at timestep {t}. " 
                f"cumulative reward: {episodic_reward:>3d}. "
                f"avg reward: {round(experiment_info['average_reward'], 2)}. "
                f"soups delivered: {soup_delivery:>3d}. "
                f"execution time: {round(end_episode - start_episode, 2)} seconds.")
            
            # performing the training with SGD
            start_training = time.time()
            for epoch in range(1, NUMBER_OF_EPOCHS + 1):
                num_batches = len(actions) // BATCH_SIZE
                shuffled_indices = tf.random.shuffle(tf.range(len(actions)))
                for batch in range(num_batches):
                    if batch == num_batches: 
                        # last batch we take the remaining elements
                        idx = shuffled_indices[batch*BATCH_SIZE:]
                    else:
                        idx = shuffled_indices[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]

                    observations_batch = tf.gather(observations, idx)
                    new_observations_batch = tf.gather(new_observations, idx)
                    critic_values_batch = tf.squeeze(critic.call(observations_batch))
                    critic_new_values_batch = tf.squeeze(critic.call(new_observations_batch))

                    if not SHARED_AGENT:
                        # if the critic is not shared, we compute the expected values also for the second agent
                        second_critic_values_batch = tf.squeeze(second_critic.call(observations_batch))
                        second_critic_new_values_batch = tf.squeeze(second_critic.call(new_observations_batch))
                        # stacking the critic values and new values for both agents in a (400,2) tensor
                        critic_values_batch = tf.stack([critic_values_batch,second_critic_values_batch], axis=1)
                        critic_new_values_batch = tf.stack([critic_new_values_batch,second_critic_new_values_batch], axis=1)

                    rewards_batch = tf.gather(tf.constant(rewards, dtype=float), idx)
                    # deltas_batch = tf.gather(deltas, idx)
                    deltas_batch = rewards_batch + GAMMA*critic_new_values_batch - critic_values_batch
                    actions_batch = tf.gather(actions, idx)

                    if ALGORITHM == 'ac':
                        # training the critic(s)
                        if SHARED_AGENT:
                            critic.train_batch(deltas_batch, observations_batch)
                        else:
                            critic.train_batch(deltas_batch[:,0], observations_batch)
                            second_critic.train_batch(deltas_batch[:,1], observations_batch)

                        # training the actor
                        actor.train_batch(deltas_batch, observations_batch, actions_batch)

                    elif ALGORITHM == 'ppo':
                        # training the critic(s)
                        if SHARED_AGENT:
                            critic.train_batch_PPO(rewards_batch, observations_batch, new_observations_batch, GAMMA)
                        else:
                            critic.train_batch_PPO(rewards_batch[:,0], observations_batch, new_observations_batch, GAMMA)
                            second_critic.train_batch_PPO(rewards_batch[:,1], observations_batch, new_observations_batch, GAMMA)

                        # training the actor
                        actor.train_batch_PPO(deltas_batch, observations_batch, actions_batch, old_policy)

            end_training = time.time()
            print(f"Training ended in {round(end_training - start_training, 2)} seconds")

            agent_1.update_old_policy()
            # agent_2.update_old_policy() # the old_policy is shared, so we don't need to update it again

            if episode > 20 and experiment_info["average_reward"] > experiment_info["best_avg"]:
                # saving the weights if the average reward is better than the best one so far
                experiment_info["best_avg"] = experiment_info['average_reward']
                save_weights()
                last_saved_weights_episode = episode

        save_experiment_info(experiment_info)
    
    
    except KeyboardInterrupt:
        print("")
        print(f"User interrupted the experiment.")
        save_experiment_info(experiment_info)
    
    except Exception as e:
        print("")
        print(f"An unexpected error occurred: {e}")
        save_experiment_info(experiment_info)
