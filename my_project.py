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
from tqdm.notebook import tqdm
from typing import Tuple, List, Dict
import sys
import argparse
import json
import time
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="dummy_experiment", help="the name of this experiment")
    parser.add_argument("--algorithm", type=str, default='ac', choices=['ac', 'ppo'], help="the name of the algorithm to use")
    parser.add_argument("--shared-agent", type=lambda x: (str(x).lower() == "true"), default=True, help="whether to use the same agent or not")
    parser.add_argument("--num-episodes", type=int, default=1000, help="number of episodes to train the agent on")
    parser.add_argument("--num-epochs", type=int, default=2, help="number of epochs to train the agent on")
    parser.add_argument("--batch-size", type=int, default=20, help="batch size of the training")
    parser.add_argument("--prev-action", type=int, default=5, help="number of actions prior to actual reward. necessary at first")
    parser.add_argument("--prev-action-limit", type=int, default=5000, help="number of actions prior to actual reward. necessary at first")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor for estimating the future state value")
    parser.add_argument("--lr-w", type=float, default=1e-5, help="learning rate for the critic")
    parser.add_argument("--lr-t", type=float, default=1e-6, help="learning rate for the actor")
    parser.add_argument("--epsilon-greedy", type=lambda x: (str(x).lower() == "true"), default=False, help="whether you want to use epsilon-greedy")
    parser.add_argument("--load-weights", type=lambda x: (str(x).lower() == "true"), default=False, help="whether you want to load previous weights")
    parser.add_argument("--run-on-colab", type=lambda x: (str(x).lower() == "true"), default=False, help="whether you are running it from colab")
    # parser.add_argument("--name-weights", type=str, help="name of the experiment's weights")

    args = parser.parse_args()

    if args.shared_agent in ["f","F","false","False","FALSE"]:
        args.shared_agent = False
    
    if args.load_weights in ["t","T","true","True","TRUE"]:
        args.load_weights = True
 
    return args


class Policy(Model):
    def __init__(self, input_shape, num_actions, optimizer, epsilon = 0.05):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.input_a = Input(shape=(self.input_shape))
        self.input_b = Input(shape=(self.input_shape))
        self.dense_1 = layers.Dense(128, activation='tanh')
        self.dense_2 = layers.Dense(256, activation='tanh')
        self.dense_3 = layers.Dense(256, activation='tanh')
        self.dense_4 = layers.Dense(128, activation='tanh')
        self.policy_a = layers.Dense(self.num_actions, activation='softmax', name="policy_a")
        self.policy_b = layers.Dense(self.num_actions, activation='softmax', name="policy_b")
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

        grad_pi_a = tape.gradient(pi_a, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grad_pi_a, self.trainable_weights))

    def train_batch_PPO(self, deltas_batch: tf.Tensor, obs_batch, actions_batch, old_policy):
        if tf.rank(deltas_batch) == 1:
            # in this way i have either 2 identical deltas or specific for each agent.
            deltas_batch = tf.stack([deltas_batch,deltas_batch], axis=1)
        with tf.GradientTape() as tape:
            pi = self.call(obs_batch, training=True)
            old_pi = old_policy.call(obs_batch)
            pi_ratio_1 = pi[0] / old_pi[0] + 1e-8 # to avoid numerical instability
            pi_ratio_2 = pi[1] / old_pi[1] + 1e-8 # to avoid numerical instability
            pi_clipped_ratio_1 = tf.clip_by_value(pi_ratio_1, 1 - self.epsilon, 1 + self.epsilon)
            pi_clipped_ratio_2 = tf.clip_by_value(pi_ratio_2, 1 - self.epsilon, 1 + self.epsilon)
            pi_ratio_advantage_1 = pi_ratio_1*deltas_batch[:,:1] # to preserve the second dimension
            pi_ratio_advantage_2 = pi_ratio_2*deltas_batch[:,1:] # to preserve the second dimension
            pi_clipped_ratio_advantage_1 = pi_clipped_ratio_1*deltas_batch[:,:1] # to preserve the second dimension
            pi_clipped_ratio_advantage_2 = pi_clipped_ratio_2*deltas_batch[:,1:] # to preserve the second dimension
            L = 0
            for i in range(len(actions_batch)):
                L += min(pi_ratio_advantage_1[i][actions_batch[i][0]], pi_clipped_ratio_advantage_1[i][actions_batch[i][0]])
                L += min(pi_ratio_advantage_2[i][actions_batch[i][1]], pi_clipped_ratio_advantage_2[i][actions_batch[i][1]])
            loss = -L

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
    def __init__(self, input_shape, optimizer):
        super().__init__()
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.input_a = Input(shape=(self.input_shape))
        self.input_b = Input(shape=(self.input_shape))
        self.dense_1 = layers.Dense(128, activation='tanh')
        self.dense_2 = layers.Dense(256, activation='tanh')
        self.dense_3 = layers.Dense(256, activation='tanh')
        self.dense_4 = layers.Dense(128, activation='tanh')
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
    def __init__(self, actor, old_policy, critic, idx, base_env: OvercookedEnv, epsilon):
        super().__init__()
        self.actor = actor
        self.old_policy = old_policy
        self.critic = critic
        self.idx = idx
        if not self.idx in [0,1]:
            raise AssertionError("The index of the agent must be either 0 or 1!")
        self.base_env = base_env
        self.epsilon = epsilon
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
        
        if np.random.random() > self.epsilon:
            action = Action.sample(np.squeeze(action_probs))
        else:
            action_idx = np.random.choice(range(0,6), size=1)[0]
            action = Action.INDEX_TO_ACTION[action_idx] # random exploration
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

    EXP_NAME = args.exp_name
    ALGORITHM = args.algorithm
    SHARED_AGENT = args.shared_agent
    LOAD_WEIGHTS = args.load_weights
    RUN_ON_COLAB = args.run_on_colab

    LR_CRITIC = args.lr_w
    LR_ACTOR = args.lr_t
    NUMBER_OF_EPISODES = args.num_episodes
    NUMBER_OF_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    PREV_ACTION_TO_REWARD = args.prev_action
    PREV_ACTION_LIMIT = args.prev_action_limit
    GAMMA = args.gamma
    EPSILON_GREEDY = args.epsilon_greedy

    PATH_ACTOR = os.path.join("networks", "actor", "actor_" + EXP_NAME + ".weights.h5") 
    PATH_CRITIC = os.path.join("networks","critic", "critic_" + EXP_NAME + ".weights.h5") 
    PATH_SECOND_CRITIC = os.path.join("networks","second_critic", "second_critic_" + EXP_NAME + ".weights.h5")
    PATH_EXPERIMENT_INFO = os.path.join("info", EXP_NAME + ".json") 

    print("")
    print(f"Experiment Name: {EXP_NAME}")
    print(f"Algorithm: {ALGORITHM}")
    print(f"Shared Agent: {SHARED_AGENT}")
    print(f"Number of Episodes: {NUMBER_OF_EPISODES}")
    print(f"Number of Epochs: {NUMBER_OF_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Previous Action to Reward: {PREV_ACTION_TO_REWARD}")
    print(f"Previous Action Limit: {PREV_ACTION_LIMIT}")
    print(f"Gamma: {GAMMA}")
    print(f"Epsilon Greedy: {EPSILON_GREEDY}")
    print(f"Learning Rate Critic: {LR_CRITIC}")
    print(f"Learning Rate Actor: {LR_ACTOR}")
    print(f"Loading previous weights: {LOAD_WEIGHTS}")
    print(f"Running on colab: {RUN_ON_COLAB}")
    print("")
    print(f"Weights will be saved and loaded from the following paths:")
    print(f"Path actor: {PATH_ACTOR}")
    print(f"Path critic: {PATH_CRITIC}")
    print(f"Path second critic: {PATH_SECOND_CRITIC}")
    print("")

    if RUN_ON_COLAB:
        sys.path.append('/content/overcooked_ai/src')

    number_of_frames = 400
    layout_name = "cramped_room"
    base_mdp = OvercookedGridworld.from_layout_name(layout_name=layout_name) #or other layout
    base_env = OvercookedEnv.from_mdp(base_mdp, info_level=0, horizon=number_of_frames)
    env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

    input_shape = env.observation_space._shape

    actor = Policy(
        input_shape=input_shape,
        num_actions=Action.NUM_ACTIONS,
        optimizer=Adam(learning_rate=LR_ACTOR)
        )

    critic = ValueFunctionApproximator(
        input_shape=input_shape,
        optimizer=Adam(learning_rate=LR_CRITIC)
        )
    
    if ALGORITHM == "ppo":
        old_policy = Policy(input_shape=input_shape, num_actions=Action.NUM_ACTIONS, optimizer=Adam(learning_rate=LR_ACTOR))
    else:
        old_policy = None
    
    if not SHARED_AGENT:
        second_critic = ValueFunctionApproximator(input_shape=input_shape,optimizer=Adam(learning_rate=LR_CRITIC))
    else:
        second_critic = critic

    if LOAD_WEIGHTS:
        if SHARED_AGENT:
            condition = os.path.exists(PATH_ACTOR) and os.path.exists(PATH_CRITIC)
        else:
            condition = os.path.exists(PATH_ACTOR) and os.path.exists(PATH_CRITIC) and os.path.exists(PATH_SECOND_CRITIC)

        if condition:
            print("")
            print("Loading previous weights...")
            print("")
            actor.load_weights(PATH_ACTOR)
            critic.load_weights(PATH_CRITIC)
            if not SHARED_AGENT:
                second_critic.load_weights(PATH_SECOND_CRITIC)
        else:
            print("")
            print("Previous weights not found.")
            command = input("Do you want to continue anyway? (y/n) ")
            if "y" not in command:
                exit("Exiting.")
            print("Starting from scratch.")
            print("")
    else:
        condition = os.path.exists(PATH_ACTOR) or os.path.exists(PATH_CRITIC) or os.path.exists(PATH_SECOND_CRITIC)

        if condition:
            print("")
            print("There exist already weights with this name.")
            command = input("Do you want to continue anyway? (y/n) ")
            if "y" not in command:
                exit("Exiting.")
            print("Overriding weights.")
            print("")
    
    if EPSILON_GREEDY:
        epsilon = 0.1
    else:
        epsilon = 0.0

    agent_1 = MyAgent(
        actor=actor,
        old_policy=old_policy,
        critic=critic,
        idx=0,
        base_env=base_env,
        epsilon=epsilon
    )
    agent_2 = MyAgent(
        actor=actor,
        old_policy=old_policy,
        critic=second_critic,
        idx=1,
        base_env=base_env,
        epsilon=epsilon
    )

    # loading previous experiment's info
    if os.path.exists(PATH_EXPERIMENT_INFO) and LOAD_WEIGHTS:
        with open(PATH_EXPERIMENT_INFO, 'r') as f:
            experiment_info = json.load(f)
            print("Experiment's info loaded.") 

    else:
        experiment_info = {
            "exp_name": EXP_NAME, 
            "algorithm": ALGORITHM, 
            "shared_agent": SHARED_AGENT, 
            "load_weights": LOAD_WEIGHTS,
            "lr_critic": LR_CRITIC, 
            "lr_actor": LR_ACTOR, 
            "number_of_episodes": NUMBER_OF_EPISODES, 
            "number_of_epochs": NUMBER_OF_EPOCHS,
            "batch_size": BATCH_SIZE, 
            "prev_action_to_reward": PREV_ACTION_TO_REWARD, 
            "prev_action_limit": PREV_ACTION_LIMIT, 
            "gamma": GAMMA,
            "epsilon_greedy": EPSILON_GREEDY,
            "average_reward" : 0,
            "best_avg" : 0,
            "avg_reward_list" : [],
        }
        

    try:
        for episode in range(1, NUMBER_OF_EPISODES + 1):
            actions = []
            observations = []
            rewards = []

            t = 0
            obs = env.reset()
            obs = obs['both_agent_obs'] 
            
            done = False
            cumulative_reward = 0

            start_episode = time.time()

            while not done:
                action1 = agent_1.action(obs)
                action2 = agent_2.action(obs)
                player_1_action = Action.ACTION_TO_INDEX[action1[0]]
                player_2_action = Action.ACTION_TO_INDEX[action2[0]]
                action = (player_1_action, player_2_action)

                actions.append(action)
                observations.append(obs)

                new_obs, reward, done, env_info = env.step(action)

                shaped_reward = sum(env_info['shaped_r_by_agent']) # let's use shaped reward for learning how to play first.
                shaped_reward_1 = env_info['shaped_r_by_agent'][0] 
                shaped_reward_2 = env_info['shaped_r_by_agent'][1]

                sparse_reward = reward # the reward is the sparse reward
                sparse_reward_1 = env_info['sparse_r_by_agent'][0]
                sparse_reward_2 = env_info['sparse_r_by_agent'][1]

                total_reward = reward + shaped_reward 
                total_reward_1 = shaped_reward_1 + sparse_reward_1
                total_reward_2 = shaped_reward_2 + sparse_reward_2

                cumulative_reward += total_reward

                if SHARED_AGENT:
                    rewards.append(total_reward)

                    if total_reward > 0 and episode < PREV_ACTION_LIMIT:
                        if t > PREV_ACTION_TO_REWARD:
                            for i in range(t-1, t-PREV_ACTION_TO_REWARD-1, -1):
                                rewards[i] += total_reward
                        else:
                            for i in range(t-1,-1,-1):
                                rewards[i] += total_reward
                else:
                    rewards.append([total_reward_1, total_reward_2])

                    if episode < PREV_ACTION_LIMIT:

                        if total_reward_1 > 0:
                            if t > PREV_ACTION_TO_REWARD:
                                for i in range(t-1, t-PREV_ACTION_TO_REWARD-1, -1):
                                    rewards[i][0] += total_reward_1
                            else:
                                for i in range(t-1,-1,-1):
                                    rewards[i][0] += total_reward_1

                        if total_reward_2 > 0:
                            if t > PREV_ACTION_TO_REWARD:
                                for i in range(t-1, t-PREV_ACTION_TO_REWARD-1, -1):
                                    rewards[i][1] += total_reward_2
                            else:
                                for i in range(t-1,-1,-1):
                                    rewards[i][1] += total_reward_2
            
                new_obs = new_obs['both_agent_obs']

                # update state (obs = new_obs)
                obs = new_obs

                # think about training the critic by itself for a while
                t += 1
            
            critic_values = tf.squeeze(critic.call(observations))
            critic_new_values = tf.squeeze(critic.call(observations[1:])) # it represent the estimation of the next observation
            critic_new_values = tf.concat([critic_new_values, tf.constant([0.0])], axis=0) # the last one is 0
            
            if not SHARED_AGENT:
                second_critic_values = tf.squeeze(second_critic.call(observations))
                second_critic_new_values = tf.squeeze(second_critic.call(observations[1:])) # it represent the estimation of the next observation
                second_critic_new_values = tf.concat([second_critic_new_values, tf.constant([0.0])], axis=0) # the last one is 0

                critic_values = tf.stack([critic_values,second_critic_values], axis=1)
                critic_new_values = tf.stack([critic_new_values,second_critic_new_values], axis=1)
            
            deltas = rewards + GAMMA*critic_new_values - critic_values

            # experiment_info['average_reward'] = 1/(episode)*( cumulative_reward + (episode-1)*experiment_info['average_reward'])
            avg_denominator = len(experiment_info["avg_reward_list"]) + 1
            experiment_info['average_reward'] = 1/(avg_denominator)*( cumulative_reward + (avg_denominator-1)*experiment_info['average_reward'])
            experiment_info["avg_reward_list"].append(round(experiment_info['average_reward'],2))
            
            end_episode = time.time()

            # print(f"agent_1 critic = {agent_1.critic(obs)}")
            # print(f"agent_2 critic = {agent_2.critic(obs)}")

            print(f"Episode [{episode:>3d}] terminated at timestep {t}." 
                f"cumulative reward: {cumulative_reward:>3d}. avg reward: {round(experiment_info['average_reward'], 3)}. "
                f"execution time: {round(end_episode - start_episode, 2):>3f} seconds")
            
            print(f"Performing stocastic gradient descent with {NUMBER_OF_EPOCHS} epochs.")
            start_training = time.time()
            for epoch in range(1, NUMBER_OF_EPOCHS + 1):
                num_batches = len(actions) // BATCH_SIZE
                shuffled_indices = tf.random.shuffle(tf.range(len(actions)))
                for batch in range(num_batches):
                    if batch == num_batches: # last batch
                        idx = shuffled_indices[batch*BATCH_SIZE:]
                    else:
                        idx = shuffled_indices[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]

                    deltas_batch = tf.gather(deltas, idx)
                    actions_batch = tf.gather(actions, idx)
                    observations_batch = tf.gather(observations, idx)

                    if SHARED_AGENT:
                        critic.train_batch(deltas_batch, observations_batch)
                    else:
                        critic.train_batch(deltas_batch[:,0], observations_batch)
                        second_critic.train_batch(deltas_batch[:,1], observations_batch)

                    actor.train(deltas_batch, observations_batch, actions_batch, old_policy, algorithm=args.algorithm)

                print(f"Epoch {epoch} terminated.")

            end_training = time.time()
            print(f"Training ended in {round(end_training - start_training, 2)} seconds")

            agent_1.update_old_policy()
            # agent_2.update_old_policy()

            if episode > 20 and experiment_info["average_reward"] > experiment_info["best_avg"]:
                experiment_info["best_avg"] = experiment_info['average_reward']
                critic.save_weights(PATH_CRITIC)
                actor.save_weights(PATH_ACTOR)
                if not SHARED_AGENT:
                    second_critic.save_weights(PATH_SECOND_CRITIC)
                print("Weights saved.")
        
        # print(f"agent_1 critic = {agent_1.critic(obs)}")
        # print(f"agent_2 critic = {agent_2.critic(obs)}")
    
    except KeyboardInterrupt:
        print(f"User interrupted the experiment.")
        print(f"Saving weights and experiment's info.")
        critic.save_weights(PATH_CRITIC)
        actor.save_weights(PATH_ACTOR)
        if not SHARED_AGENT:
            second_critic.save_weights(PATH_SECOND_CRITIC)
        
        # saving experiment info
        with open(PATH_EXPERIMENT_INFO, 'w') as f:
            json.dump(experiment_info, f)

    with open(PATH_EXPERIMENT_INFO, 'w') as f:
        json.dump(experiment_info, f)
