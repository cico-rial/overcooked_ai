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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="dummy_experiment", help="the name of this experiment. weights will be loaded from it")
    parser.add_argument("--seed", type=int, default=42, help="set the seed for reproducibility of the experiment")
    parser.add_argument("--num-episodes", type=int, default=10, help="number of episodes for which to compute the average reward")
    parser.add_argument("--refresh-rate", type=int, default=250, help="refresh-rate for displaying the episode")

    args = parser.parse_args()

    return args


def visualize_states(states: list[OvercookedState], rewards:list[list], deliveries: list, base_mdp: OvercookedGridworld, refresh_rate=500):

    pygame.init()
    pygame.display.init()

    UPDATE_INTERVAL = refresh_rate  # milliseconds
    UPDATE_EVENT = pygame.USEREVENT + 1

    visualizer = StateVisualizer()
    rendered_states = [
        visualizer.render_state(state, grid=base_mdp.terrain_mtx)
        for state in states
    ]

    rendered_state = rendered_states[0]
    screen = pygame.display.set_mode(
        rendered_state.get_size(),
        flags=pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
    )
    screen.blit(rendered_state, (0, 0))
    pygame.display.flip()

    pygame.time.set_timer(UPDATE_EVENT, UPDATE_INTERVAL)
    pygame.display.set_caption("OvercookedAI")

    # Initialize font
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 12)
    
    rendered_state_index = 1
    n_rendered_states = len(rendered_states)
    running = True

    soups_delivered = 0
    cumulative_sparse_reward =  0
    cumulative_shaped_reward =  0
    end_soups = len(deliveries) == 0

    while running:
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                running = False

            elif event.type == UPDATE_EVENT:
                if rendered_state_index < n_rendered_states:
                    rendered_state = rendered_states[rendered_state_index]
                    screen.blit(rendered_state, (0, 0))

                    if not end_soups and rendered_state_index == deliveries[soups_delivered]:
                        soups_delivered += 1
                        end_soups = len(deliveries) == soups_delivered

                    cumulative_sparse_reward += rewards[0][rendered_state_index]
                    cumulative_shaped_reward += rewards[1][rendered_state_index]

                    # Draw timestep t
                    t = rendered_state_index
                    timestep = font.render(f"Timestep: {t}", True, (0, 0, 0))  # Red color
                    soup = font.render(f"Soups count: {soups_delivered}", True, (0, 0, 0))  # Red color
                    sparse_reward = font.render(f"Sparse reward: {cumulative_sparse_reward}", True, (0, 0, 0))  # Red color
                    shaped_reward = font.render(f"Shaped reward: {cumulative_shaped_reward}", True, (0, 0, 0))  # Red color
                    screen.blit(timestep, (10, 10))  # Position (x=10, y=10)
                    screen.blit(soup, (10, 25))  # Position (x=10, y=10)
                    screen.blit(sparse_reward, (10, 40))  # Position (x=10, y=10)
                    screen.blit(shaped_reward, (10, 55))  # Position (x=10, y=10)

                    pygame.display.flip()
                    rendered_state_index += 1
                else:
                    running = False

    pygame.time.wait(2000)
    pygame.quit()


def set_seed_for_reproducibility(SEED):
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)   
    tf.config.experimental.enable_op_determinism()


def load_weights():
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


class Policy(Model):
    def __init__(self, input_shape, num_actions, optimizer=None, entropy_loss=None, epsilon = 0.05):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.optimizer = optimizer
        self.entropy_loss = entropy_loss
        self.epsilon = epsilon
        self.input_a = Input(shape=(self.input_shape))
        self.input_b = Input(shape=(self.input_shape))
        self.dense_1 = layers.Dense(128, activation='tanh')
        self.dense_2 = layers.Dense(256, activation='tanh')
        self.dense_3 = layers.Dense(256, activation='tanh')
        self.dense_4 = layers.Dense(128, activation='tanh')
        # self.dense_1 = layers.Dense(64, activation='tanh')
        # self.dense_2 = layers.Dense(128, activation='tanh')
        # self.dense_3 = layers.Dense(64, activation='tanh')
        self.policy_a = layers.Dense(self.num_actions, activation='softmax', name="policy_a")
        self.policy_b = layers.Dense(self.num_actions, activation='softmax', name="policy_b")
        # self.printt = True
        self.build_model()

    
    def preprocess(self, obs):
        if isinstance(obs, Tuple):
            obs = [obs] # to handle the case where obs_batch is a single observation

        obs_1, obs_2 = zip(*obs)
        obs_batch = tf.concat([tf.stack(obs_1), tf.stack(obs_2)], axis=-1)
        return obs_batch


    def call(self, obs, training=False):
        x = self.preprocess(obs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        policy_a = self.policy_a(x)
        policy_b = self.policy_b(x)
        return (policy_a, policy_b)

    def build_model(self):
        # computing a forward pass in order to automatically build the model
        dummy_input = (
            tf.zeros((1, 96)),
            tf.zeros((1, 96))
        )
        _ = self(dummy_input)

    def train_step(self, delta, obs: Tuple, action: Tuple[int,int]):
        # update t with t + alpha_t*delta*grad_pi^(A|S) where A is the action taken before reaching St+1
        with tf.GradientTape() as tape:
            pi = self.call(obs, training=True)
            log_pi = tf.math.log(pi)
            pi_a = log_pi[0][..., action[0]] + log_pi[1][..., action[1]] # π(A|S), computing the sum of the probability of the best actions

        grad_pi_a = tape.gradient(pi_a, self.trainable_weights)
        processed_gradient = [-tf.squeeze(delta)*grad for grad in grad_pi_a]
        self.optimizer.apply_gradients(zip(processed_gradient, self.trainable_weights))
    
    def train_batch(self, deltas_batch: tf.Tensor, obs_batch, actions_batch):
        # update t with t + alpha_t*delta*grad_pi^(A|S) where A is the action taken before reaching St+1
        with tf.GradientTape() as tape:
            pi = self.call(obs_batch, training=True)
            log_pi = tf.math.log(pi)
            pi_a1 = tf.gather(log_pi[0], actions_batch[:, 0], axis=1, batch_dims=1)
            pi_a2 = tf.gather(log_pi[1], actions_batch[:, 1], axis=1, batch_dims=1)
            if tf.rank(deltas_batch) == 1:
                deltas_batch = tf.stack((deltas_batch,deltas_batch), axis=1)
            stacked_pi_a = tf.stack((pi_a1,pi_a2), axis=1)
            # Now compute the weighted sum over the batch
            pi_a = -tf.reduce_sum(deltas_batch*stacked_pi_a)
            loss = pi_a
            
            if self.entropy_loss:
                entropy_1 = -tf.reduce_sum(pi[0] * tf.math.log(pi[0] + 1e-8), axis=1)
                entropy_2 = -tf.reduce_sum(pi[1] * tf.math.log(pi[1] + 1e-8), axis=1)
                entropy_l = tf.reduce_mean((entropy_1 + entropy_2) / 2)
                loss -= entropy_l # minus because we need to maximize the entropy

        grad_pi_a = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grad_pi_a, self.trainable_weights))

    def train_batch_PPO(self, deltas_batch: tf.Tensor, obs_batch, actions_batch, old_policy):
        if tf.rank(deltas_batch) == 1:
            # in this way i have either 2 identical deltas or specific for each agent.
            deltas_batch = tf.stack([deltas_batch,deltas_batch], axis=1)
        with tf.GradientTape() as tape:
            pi = self.call(obs_batch, training=True)
            old_pi = old_policy.call(obs_batch)
            pi_ratio_1 = pi[0] / (old_pi[0] + 1e-8) # to avoid numerical instability
            pi_ratio_2 = pi[1] / (old_pi[1] + 1e-8) # to avoid numerical instability
            pi_clipped_ratio_1 = tf.clip_by_value(pi_ratio_1, 1 - self.epsilon, 1 + self.epsilon)
            pi_clipped_ratio_2 = tf.clip_by_value(pi_ratio_2, 1 - self.epsilon, 1 + self.epsilon)
            pi_ratio_advantage_1 = pi_ratio_1*deltas_batch[:,:1] # to preserve the second dimension
            pi_ratio_advantage_1 = tf.gather(pi_ratio_advantage_1, actions_batch[:, 0], axis=1, batch_dims=1)
            pi_ratio_advantage_2 = pi_ratio_2*deltas_batch[:,1:] # to preserve the second dimension
            pi_ratio_advantage_2 = tf.gather(pi_ratio_advantage_2, actions_batch[:, 1], axis=1, batch_dims=1)
            pi_clipped_ratio_advantage_1 = pi_clipped_ratio_1*deltas_batch[:,:1] # to preserve the second dimension
            pi_clipped_ratio_advantage_1 = tf.gather(pi_clipped_ratio_advantage_1, actions_batch[:, 0], axis=1, batch_dims=1)
            pi_clipped_ratio_advantage_2 = pi_clipped_ratio_2*deltas_batch[:,1:] # to preserve the second dimension
            pi_clipped_ratio_advantage_2 = tf.gather(pi_clipped_ratio_advantage_2, actions_batch[:, 1], axis=1, batch_dims=1)
            min_pi_ratio_1 = tf.minimum(pi_ratio_advantage_1, pi_clipped_ratio_advantage_1)
            min_pi_ratio_2 = tf.minimum(pi_ratio_advantage_2, pi_clipped_ratio_advantage_2)
            loss = - tf.reduce_sum(min_pi_ratio_1+min_pi_ratio_2)

            if self.entropy_loss:
                entropy_1 = -tf.reduce_sum(pi[0] * tf.math.log(pi[0] + 1e-8), axis=1)
                entropy_2 = -tf.reduce_sum(pi[1] * tf.math.log(pi[1] + 1e-8), axis=1)
                entropy_l = tf.reduce_mean((entropy_1 + entropy_2) / 2)
                loss -= entropy_l # minus because we need to maximize the entropy

        grad_loss = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grad_loss, self.trainable_weights))
    

    def train(self, deltas_batch: tf.Tensor, obs_batch, actions_batch, old_policy, algorithm='ac'):
        if algorithm == 'ppo':
            self.train_batch_PPO(deltas_batch, obs_batch, actions_batch, old_policy)

        elif algorithm == 'ac':
            self.train_batch(deltas_batch, obs_batch, actions_batch)

        else:
            raise KeyError("The algorithm can only be 'ac' or 'ppo'.")


class ValueFunctionApproximator(Model):
    def __init__(self, input_shape, optimizer=None):
        super().__init__()
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.input_a = Input(shape=(self.input_shape))
        self.input_b = Input(shape=(self.input_shape))
        self.dense_1 = layers.Dense(128, activation='tanh')
        self.dense_2 = layers.Dense(256, activation='tanh')
        self.dense_3 = layers.Dense(256, activation='tanh')
        self.dense_4 = layers.Dense(128, activation='tanh')
        # self.dense_1 = layers.Dense(64, activation='tanh')
        # self.dense_2 = layers.Dense(128, activation='tanh')
        # self.dense_3 = layers.Dense(64, activation='tanh')
        self.value_function = layers.Dense(1, name="value_function")
        self.build_model()

    
    def preprocess(self, obs):
        if isinstance(obs, Tuple):
            obs = [obs] # to handle the case where obs_batch is a single observation

        obs_1, obs_2 = zip(*obs)
        obs_batch = tf.concat([tf.stack(obs_1), tf.stack(obs_2)], axis=-1)
        return obs_batch


    def call(self, obs: Tuple, training=False):
        x = self.preprocess(obs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        value_function = self.value_function(x)
        return value_function

    def build_model(self):
        # computing a forward pass in order to automatically build the model
        dummy_input = (
            tf.zeros((1, 96)),
            tf.zeros((1, 96))
        )
        _ = self(dummy_input)

    def train_step(self, delta, obs: Tuple):
        # update w with w + alpha_w*delta*grad_v^(St)
        with tf.GradientTape() as tape:
            state_value = self.call(obs, training=True)

        grad_state_value = tape.gradient(state_value, self.trainable_weights)
        processed_gradient = [-tf.squeeze(delta)*grad for grad in grad_state_value]
        self.optimizer.apply_gradients(zip(processed_gradient, self.trainable_weights))

    def train_batch(self, deltas_batch: tf.Tensor, obs_batch): # deltas is a tf.Tensor of shape (batch_size,1)
        # update w with w + alpha_w*grad_v^(St)*delta
        with tf.GradientTape() as tape:
            state_value = self.call(obs_batch, training=True)
            processed_state_value = -deltas_batch * state_value

        grad_state_value = tape.gradient(processed_state_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grad_state_value, self.trainable_weights))


class MyAgent(Agent):
    """
    This class is more a couple of actors since we use shared networks and the output are 2!!!
    For now let's treat it like a single player identified by self.index
    """
    def __init__(self, actor, old_policy, critic, idx, base_env: OvercookedEnv):
        super().__init__()
        self.actor = actor
        self.old_policy = old_policy
        self.critic = critic
        self.idx = idx
        if not self.idx in [0,1]:
            raise AssertionError("The index of the agent must be either 0 or 1!")
        self.base_env = base_env
        self.update_old_policy()

    def action(self, obs):
        """
        obs: preprocessed observation (or overcookedstate)
        We want to output the action given the state. can use a NN!
        should return a tuple (Action, Dict)
        Dict should contain info about the action ('action_probs': numpy array)
        """
        if isinstance(obs, OvercookedState):
            # this is useful for translating the OvercookedState
            # into observation that can be fed into the NN.
            state = obs
            obs_from_state = self.base_env.featurize_state_mdp(state)
            obs = (obs_from_state[0],obs_from_state[1])

        action_probs = self.actor.call(obs)[self.idx].numpy()
        action = Action.sample(np.squeeze(action_probs))
        
        return (action, {'action_probs': action_probs})

    def actions(self, obss):
        """
        Look at the documentation of the Agent class
        """
        pass

    def update(self, obs, reward):
        """
        What do we need to update?
        """
        pass

    def update_old_policy(self):
        if self.old_policy is not None:
            self.old_policy.set_weights(self.actor.get_weights())


if __name__ == "__main__":
    args = parse_args()

    # algorithm specifications
    EXP_NAME = args.exp_name
    NUMBER_OF_EPISODES = args.num_episodes
    REFRESH_RATE = args.refresh_rate
    SEED = args.seed

    # PATH_ACTOR = os.path.join("networks", "actor", "actor_" + EXP_NAME + ".weights.h5") 
    PATH_ACTOR = "networks/actor/shared_actor_best_2.weights.h5" 

    print("")
    print("EXPERIMENT INFO.")
    print(f"Experiment Name: {EXP_NAME}")
    print(f"Number of episodes: {NUMBER_OF_EPISODES}")
    print(f"Seed: {SEED}")

    print(f"Weights will be loaded from the following path:")
    print(f"Path actor: {PATH_ACTOR}")
    print("")

    set_seed_for_reproducibility(SEED)

    number_of_frames = 400
    layout_name = "cramped_room"
    base_mdp = OvercookedGridworld.from_layout_name(layout_name=layout_name) #or other layout
    base_env = OvercookedEnv.from_mdp(base_mdp, info_level=0, horizon=number_of_frames)
    env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

    input_shape = env.observation_space._shape

    if os.path.exists(PATH_ACTOR):
        actor = Policy(input_shape=input_shape, num_actions=Action.NUM_ACTIONS)
        load_weights()
    else:
        print(f"Couldn't find actor weights for the following experiment: '{EXP_NAME}'")
        exit("Exiting...")

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

    cumulative_sparse_rewards = []
    cumulative_shaped_rewards = []
    useful_onion_pickups = []
    potting_onions = []
    useful_dish_pickups = []
    soup_pickups = []
    soup_deliveries = []

    try:
        for episode in range(1, NUMBER_OF_EPISODES + 1):
            
            t = 0
            obs = env.reset()
            
            done = False

            episode_cumulative_sparse_reward = 0
            episode_cumulative_shaped_reward = 0

            states = []
            episode_sparse_rewards = [episode_cumulative_sparse_reward]
            episode_shaped_rewards = [episode_cumulative_shaped_reward]

            start_episode = time.time()

            while not done:
                action_1_idx = agent_1.action(obs['both_agent_obs'] )
                action_2_idx = agent_2.action(obs['both_agent_obs'] )
                agent_1_action = Action.ACTION_TO_INDEX[action_1_idx[0]]
                agent_2_action = Action.ACTION_TO_INDEX[action_2_idx[0]]
                action = (agent_1_action, agent_2_action)
                
                states.append(obs['overcooked_state'])

                new_obs, reward, done, env_info = env.step(action)

                shaped_reward = sum(env_info['shaped_r_by_agent']) 
                shaped_reward_1 = env_info['shaped_r_by_agent'][0] 
                shaped_reward_2 = env_info['shaped_r_by_agent'][1]

                sparse_reward = reward # the reward is the sparse reward
                sparse_reward_1 = env_info['sparse_r_by_agent'][0]
                sparse_reward_2 = env_info['sparse_r_by_agent'][1]

                total_reward = reward + shaped_reward 
                total_reward_1 = shaped_reward_1 + sparse_reward_1
                total_reward_2 = shaped_reward_2 + sparse_reward_2

                episode_cumulative_sparse_reward += sparse_reward
                episode_cumulative_shaped_reward += total_reward

                episode_sparse_rewards.append(sparse_reward)
                episode_shaped_rewards.append(total_reward)


                # if REWARD_TYPE == "shaped":
                #     rewards.append(total_reward)
                #     episode_cumulative_reward += total_reward
                # elif REWARD_TYPE == "sparse":
                #     rewards.append(sparse_reward)
                #     episode_cumulative_reward += sparse_reward

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
            
            average_sparse_reward = round(sum(cumulative_sparse_rewards)/len(cumulative_sparse_rewards), 3)
            average_shaped_reward = round(sum(cumulative_shaped_rewards)/len(cumulative_shaped_rewards), 3)
            average_useful_onion_pickups = round(sum(useful_onion_pickups)/len(useful_onion_pickups), 3)
            average_potting_onions = round(sum(potting_onions)/len(potting_onions), 3)
            average_useful_dish_pickups = round(sum(useful_dish_pickups)/len(useful_dish_pickups), 3)
            average_soup_pickups = round(sum(soup_pickups)/len(soup_pickups), 3)
            average_soup_deliveries = round(sum(soup_deliveries)/len(soup_deliveries), 3)
            
            end_episode = time.time()

            print(f"Episode [{episode:>3d}] terminated at timestep {t}. " 
                f"cumulative sparse reward: {episode_cumulative_sparse_reward:>3d}. "
                f"cumulative shaped reward: {episode_cumulative_shaped_reward:>3d}. "
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
        print("Showing the trajectory...")
        visualize_states(states=states, rewards=[episode_sparse_rewards,episode_shaped_rewards], deliveries=t_soup_delivery, base_mdp=base_mdp, refresh_rate=REFRESH_RATE)
        print("Closing pygame...")

    except KeyboardInterrupt:
        print("")
        print(f"User interrupted the experiment.")
