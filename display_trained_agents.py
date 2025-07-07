from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.agents.agent import Agent, AgentPair, RandomAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, Concatenate
from tensorflow.keras.models import Model
from scipy.stats import entropy
from tqdm.notebook import tqdm
from typing import Tuple, List, Dict
import sys
import argparse
import json
import time
import pygame
import os
# import warnings
# warnings.filterwarnings('ignore')
from utility.utility import set_seed_for_reproducibility, visualize_states, Policy, MyAgent

def parse_args():
    """
    Parse command line arguments for the experiment configuration.
    
    Returns:
        args (Namespace): Parsed command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="dummy_experiment", help="the name of the experiment from which weights will be loaded")
    parser.add_argument("--seed", type=int, default=42, help="set the seed for reproducibility of the experiment")
    parser.add_argument("--num-episodes", type=int, default=10, help="number of episodes for which to compute the average reward")
    parser.add_argument("--refresh-rate", type=int, default=250, help="refresh-rate for displaying the episode")

    args = parser.parse_args()

    return args


def load_weights():
    """
    Load the weights of the policy if they exist.
    If the weights do not exist, the program will terminate.
    """
    try:
        actor.load_weights(PATH_ACTOR)
        print("")
        print("Weights successfully loaded.")
        print("")
    except:
        print("")
        print("Error: loading weights has failed.")
        exit("Exiting...")
        print("")


if __name__ == "__main__":
    args = parse_args()

    # algorithm specifications
    EXP_NAME = args.exp_name
    NUMBER_OF_EPISODES = args.num_episodes
    REFRESH_RATE = args.refresh_rate
    SEED = args.seed

    PATH_ACTOR = os.path.join("networks", "actor", "actor_" + EXP_NAME + ".weights.h5") 

    print("")
    print("EXPERIMENT INFO.")
    print(f"Experiment Name: {EXP_NAME}")
    print(f"Number of episodes: {NUMBER_OF_EPISODES}")
    print(f"Seed: {SEED}")

    print(f"Weights will be loaded from the following path:")
    print(f"Path actor: {PATH_ACTOR}")
    print("")

    set_seed_for_reproducibility(SEED)

    # initializing the environment
    number_of_frames = 400
    layout_name = "cramped_room"
    base_mdp = OvercookedGridworld.from_layout_name(layout_name=layout_name) #or other layout
    base_env = OvercookedEnv.from_mdp(base_mdp, info_level=0, horizon=number_of_frames)
    env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

    input_shape = env.observation_space._shape

    # initializing the agents
    if EXP_NAME == "random_agent":
        agent_1 = RandomAgent(all_actions=True)
        agent_2 = RandomAgent(all_actions=True)

    elif os.path.exists(PATH_ACTOR):
        actor = Policy(input_shape=input_shape, num_actions=Action.NUM_ACTIONS)
        load_weights()
        agent_1 = MyAgent(
            actor=actor,
            old_policy=None,
            critic=None,
            idx=0,
            base_env=base_env
        )
        agent_2 = MyAgent(
            actor=actor,
            old_policy=None,
            critic=None,
            idx=1,
            base_env=base_env
        )
    else:
        print(f"Couldn't find actor weights for the following experiment: '{EXP_NAME}'")
        exit("Exiting...")

    cumulative_sparse_rewards = [] # list of cumulative sparse rewards for each episode
    cumulative_shaped_rewards = [] # list of cumulative shaped rewards for each episode
    useful_onion_pickups = []
    potting_onions = []
    useful_dish_pickups = []
    soup_pickups = []
    soup_deliveries = []

    try:
        for episode in range(1, NUMBER_OF_EPISODES + 1):
            
            states = []

            t = 0
            obs = env.reset()
            done = False

            episode_sparse_rewards = [0] # list of sparse rewards for the current episode for visualization
            episode_shaped_rewards = [0] # list of shaped rewards for the current episode for visualization

            start_episode = time.time()

            while not done:
                # getting the actions from the agents
                action_1_idx = agent_1.action(obs['both_agent_obs'] )
                action_2_idx = agent_2.action(obs['both_agent_obs'] )
                agent_1_action = Action.ACTION_TO_INDEX[action_1_idx[0]]
                agent_2_action = Action.ACTION_TO_INDEX[action_2_idx[0]]
                action = (agent_1_action, agent_2_action)
                
                states.append(obs['overcooked_state'])

                # performing the action and getting the results
                new_obs, reward, done, env_info = env.step(action)

                # calculating the rewards
                shaped_reward = sum(env_info['shaped_r_by_agent']) 
                shaped_reward_1 = env_info['shaped_r_by_agent'][0] 
                shaped_reward_2 = env_info['shaped_r_by_agent'][1]

                sparse_reward = reward # the reward is the sparse reward
                sparse_reward_1 = env_info['sparse_r_by_agent'][0]
                sparse_reward_2 = env_info['sparse_r_by_agent'][1]

                total_reward = reward + shaped_reward 
                total_reward_1 = shaped_reward_1 + sparse_reward_1
                total_reward_2 = shaped_reward_2 + sparse_reward_2

                episode_sparse_rewards.append(sparse_reward)
                episode_shaped_rewards.append(total_reward)

                obs = new_obs

                t += 1
            
            # getting some stats
            cumulative_sparse_rewards.append(sum(episode_sparse_rewards))
            cumulative_shaped_rewards.append(sum(episode_shaped_rewards))

            t_useful_onion_pickup = env_info.get('episode',{}).get('ep_game_stats',{}).get('useful_onion_pickup',[[],[]])
            t_useful_onion_pickup = t_useful_onion_pickup[0] + t_useful_onion_pickup[1]
            useful_onion_pickup = len(t_useful_onion_pickup)
            useful_onion_pickups.append(useful_onion_pickup)

            t_potting_onion = env_info.get('episode',{}).get('ep_game_stats',{}).get('potting_onion',[[],[]])
            t_potting_onion = t_potting_onion[0] + t_potting_onion[1]
            potting_onion = len(t_potting_onion)
            potting_onions.append(potting_onion)

            t_useful_dish_pickup = env_info.get('episode',{}).get('ep_game_stats',{}).get('useful_dish_pickup',[[],[]])
            t_useful_dish_pickup = t_useful_dish_pickup[0] + t_useful_dish_pickup[1]
            useful_dish_pickup = len(t_useful_dish_pickup)
            useful_dish_pickups.append(useful_dish_pickup)

            t_soup_pickup = env_info.get('episode',{}).get('ep_game_stats',{}).get('soup_pickup',[[],[]])
            t_soup_pickup = t_soup_pickup[0] + t_soup_pickup[1]
            soup_pickup = len(t_soup_pickup)
            soup_pickups.append(soup_pickup)

            t_soup_delivery = env_info.get('episode',{}).get('ep_game_stats',{}).get('soup_delivery',[[],[]])
            t_soup_delivery = t_soup_delivery[0] + t_soup_delivery[1]
            soup_delivery = len(t_soup_delivery)
            soup_deliveries.append(soup_delivery)

            # computing average stats
            average_sparse_reward = round(sum(cumulative_sparse_rewards)/len(cumulative_sparse_rewards), 3)
            average_shaped_reward = round(sum(cumulative_shaped_rewards)/len(cumulative_shaped_rewards), 3)
            average_useful_onion_pickups = round(sum(useful_onion_pickups)/len(useful_onion_pickups), 3)
            average_potting_onions = round(sum(potting_onions)/len(potting_onions), 3)
            average_useful_dish_pickups = round(sum(useful_dish_pickups)/len(useful_dish_pickups), 3)
            average_soup_pickups = round(sum(soup_pickups)/len(soup_pickups), 3)
            average_soup_deliveries = round(sum(soup_deliveries)/len(soup_deliveries), 3)
            
            end_episode = time.time()

            print(f"Episode [{episode:>3d}] terminated at timestep {t}. " 
                f"cumulative sparse reward: {sum(episode_sparse_rewards):>3d}. "
                f"cumulative shaped reward: {sum(episode_shaped_rewards):>3d}. "
                f"soups delivered: {soup_delivery:>3d}. "
                f"execution time: {round(end_episode - start_episode, 2)} seconds.")   

        time.sleep(2) 

        print("")
        print(f"Average results in {NUMBER_OF_EPISODES} episodes:")
        print(f"avg sparse reward: {average_sparse_reward}. ")
        print(f"avg shaped reward: {average_shaped_reward}. ")
        print(f"avg (useful) onion pickups: {average_useful_onion_pickups}. ")
        print(f"avg potting onions: {average_potting_onions}. ")
        print(f"avg (useful) dish pickups: {average_useful_dish_pickups}. ")
        print(f"avg soup pickups: {average_soup_pickups}. ")
        print(f"avg soups delivered: {average_soup_deliveries}. ")

        print("")
        print("Opening pygame...")
        time.sleep(2)
        visualize_states(states=states, 
                         rewards=[episode_sparse_rewards,episode_shaped_rewards], 
                         deliveries=t_soup_delivery, base_mdp=base_mdp, refresh_rate=REFRESH_RATE)               
        print("Closing pygame...")

    except KeyboardInterrupt:
        print("")
        print(f"User interrupted the experiment.")
