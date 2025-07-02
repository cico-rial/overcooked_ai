from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.agents.agent import Agent
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import pygame
from typing import Tuple, List, Dict, Union


def visualize_states(states: list[OvercookedState], rewards: list[list], deliveries: list, base_mdp: OvercookedGridworld, refresh_rate=500):
    """
    NOTE: Heavily adapted from Luca Napoli's code. 

    Visualizes the trajectory of states in a pygame window.

    Args:
        states (list[OvercookedState]): List of OvercookedState objects representing the trajectory.
        rewards (list[list]): List of rewards received in the episode, where each sublist contains sparse and shaped rewards.
        deliveries (list): List of delivery timesteps to highlight in the visualization.
        base_mdp (OvercookedGridworld): The base MDP used for rendering the states.
        refresh_rate (int): Refresh rate for the visualization in milliseconds.
    
    Returns:
        None
    """
    print("Showing the trajectory...")

    pygame.init()
    pygame.display.init()

    UPDATE_INTERVAL = refresh_rate  
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

                    t = rendered_state_index
                    timestep = font.render(f"Timestep: {t}", True, (0, 0, 0))  
                    soup = font.render(f"Soups count: {soups_delivered}", True, (0, 0, 0))  
                    sparse_reward = font.render(f"Sparse reward: {cumulative_sparse_reward}", True, (0, 0, 0))  
                    shaped_reward = font.render(f"Shaped reward: {cumulative_shaped_reward}", True, (0, 0, 0))  
                    screen.blit(timestep, (10, 10))  
                    screen.blit(soup, (10, 25))  
                    screen.blit(sparse_reward, (10, 40))  
                    screen.blit(shaped_reward, (10, 55))  

                    pygame.display.flip()
                    rendered_state_index += 1
                else:
                    running = False

    pygame.time.wait(2000)
    pygame.quit()


def set_seed_for_reproducibility(SEED: int):
    """ Sets the seed for reproducibility in TensorFlow and NumPy.

    Args:
        SEED (int): The seed value to set for reproducibility.
    
    Returns:
        None
    """
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)   
    tf.config.experimental.enable_op_determinism()


class Policy(Model):
    def __init__(self, input_shape: Tuple, num_actions: int, optimizer: optimizers = None, entropy_loss: bool = None, epsilon: float = 0.05):
        """
        Keras Model subclassing tensorflow.keras.models. It represents the policy network for the overcooked environment.
        It takes as input a tuple of observations (one for each agent) and outputs the action probabilities for each agent.

        Args:
            input_shape (Tuple): The shape of the input observations for each agent.
            num_actions (int): The number of actions available for each agent.
            optimizer (optimizers, optional): The optimizer to use for training the model. Defaults to None.
            entropy_loss (bool, optional): Whether to include an entropy loss term in the training. Defaults to None.
            epsilon (float, optional): The epsilon value for clipping in PPO. Defaults to 0.05.
        Returns:
            None
        """
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.optimizer = optimizer
        self.entropy_loss = entropy_loss
        self.epsilon = epsilon
        self.input_a = Input(shape=(self.input_shape))
        self.input_b = Input(shape=(self.input_shape))
        self.dense_1 = layers.Dense(64, activation='tanh')
        self.dense_2 = layers.Dense(128, activation='tanh')
        self.dense_3 = layers.Dense(64, activation='tanh')
        self.policy_a = layers.Dense(self.num_actions, activation='softmax', name="policy_a")
        self.policy_b = layers.Dense(self.num_actions, activation='softmax', name="policy_b")
        self.build_model()

    
    def preprocess(self, obs: Union[Tuple[np.array, np.array], tf.Tensor]) -> tf.Tensor:
        """
        Preprocesses the observations by stacking them along the last dimension.

        Args:
            obs (Tuple or tf.Tensor): The observations to preprocess. Can be:
                - A tuple of two np.array (first and second agent's observations)
                - A tf.Tensor of shape (batch_size, 2, input_shape) (for batched input)
        Returns:
            tf.Tensor: A tensor of shape (batch_size, input_shape * 2) containing the concatenated observations.
        """
        if isinstance(obs, Tuple):
            obs = [obs] # to handle the case where obs_batch is a single observation

        obs_1, obs_2 = zip(*obs)
        obs_batch = tf.concat([tf.stack(obs_1), tf.stack(obs_2)], axis=-1)
        return obs_batch


    def call(self, obs: Union[Tuple, tf.Tensor], training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass of the model.

        Args:
            obs (Tuple or tf.Tensor): The input observations,  Can be:
                - A tuple of two np.array (first and second agent's observations)
                - A tf.Tensor of shape (batch_size, 2, input_shape) (for batched input)
            training (bool, optional): Whether the model is in training mode. Defaults to False.
        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the action probabilities for each agent.
        """
        x = self.preprocess(obs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        policy_a = self.policy_a(x)
        policy_b = self.policy_b(x)
        return (policy_a, policy_b)

    def build_model(self):
        """
        Builds the model by computing a forward pass with dummy input.
        """
        dummy_input = (
            tf.zeros((1, 96)),
            tf.zeros((1, 96))
        )
        _ = self(dummy_input)

    def train_step(self, delta: tf.Tensor, obs: Tuple, action: Tuple[int,int]):
        """
        WARNING: This method is DEPRECATED and will be removed in future versions.
        Use `train_batch` or `train_batch_PPO` instead for batch training.

        Implements the one-step actor-critic weight update. 

        Args:
            delta (tf.Tensor): The delta value to scale the gradient.
            obs (Tuple): The input observations for the agents.
            action (Tuple[int, int]): The actions taken by the agents.
        Returns:
            None
        """
        
        with tf.GradientTape() as tape:
            pi = self.call(obs, training=True)
            log_pi = tf.math.log(pi)
            pi_a = log_pi[0][..., action[0]] + log_pi[1][..., action[1]] 

        grad_pi_a = tape.gradient(pi_a, self.trainable_weights)
        processed_gradient = [-tf.squeeze(delta)*grad for grad in grad_pi_a]
        self.optimizer.apply_gradients(zip(processed_gradient, self.trainable_weights))
    
    def train_batch(self, deltas_batch: tf.Tensor, observation_batch: tf.Tensor, actions_batch: tf.Tensor):
        """
        Implements the batch actor-critic weight update.
        
        Args:
            deltas_batch (tf.Tensor): A tensor of shape (batch_size,) or (batch_size, 2) containing the deltas for each observation.
            The rank of the tensor depends on whether the delta is shared by the the agents.
            observation_batch (tf.Tensor): A tensor of shape (batch_size, 2, input_shape) containing the observations for each agent.
            actions_batch (tf.Tensor): A tensor of shape (batch_size, 2) containing the actions taken by each agent.
        Returns:
            None
        """
        with tf.GradientTape() as tape:
            pi = self.call(observation_batch, training=True)
            log_pi = tf.math.log(pi)
            pi_a1 = tf.gather(log_pi[0], actions_batch[:, 0], axis=1, batch_dims=1)
            pi_a2 = tf.gather(log_pi[1], actions_batch[:, 1], axis=1, batch_dims=1)
            if tf.rank(deltas_batch) == 1: 
                # check if deltas_batch has a single dimension (shared by the agents)
                # if so, i compute a 2 column identical delta.
                # if not we assume that deltas_batch is already a 2 column tensor.
                deltas_batch = tf.stack((deltas_batch,deltas_batch), axis=1)
            stacked_pi_a = tf.stack((pi_a1,pi_a2), axis=1)

            # Now compute the weighted sum over the batch
            pi_a = -tf.reduce_sum(deltas_batch*stacked_pi_a)
            loss = pi_a
            
            if self.entropy_loss: # entropy loss term if specified
                entropy_1 = -tf.reduce_sum(pi[0] * tf.math.log(pi[0] + 1e-8), axis=1)
                entropy_2 = -tf.reduce_sum(pi[1] * tf.math.log(pi[1] + 1e-8), axis=1)
                entropy_l = tf.reduce_mean((entropy_1 + entropy_2) / 2)
                loss -= entropy_l # minus because we need to maximize the entropy

        grad_pi_a = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grad_pi_a, self.trainable_weights))

    def train_batch_PPO(self, deltas_batch: tf.Tensor, observation_batch: tf.Tensor, actions_batch: tf.Tensor, old_policy: "Policy"):
        """
        Implements the batch PPO weight update.

        Args:
            deltas_batch (tf.Tensor): A tensor of shape (batch_size,) or (batch_size, 2) containing the deltas for each observation.
            The rank of the tensor depends on whether the delta is shared by the the agents.
            observation_batch (tf.Tensor): A tensor of shape (batch_size, 2, input_shape) containing the observations for each agent.
            actions_batch (tf.Tensor): A tensor of shape (batch_size, 2) containing the actions taken by each agent.
            old_policy (Policy): The old policy model used for computing the PPO loss.
        Returns:
            None
        """
        if tf.rank(deltas_batch) == 1:
            # check if deltas_batch has a single dimension (shared by the agents)
            # if so, i compute a 2 column identical delta.
            # if not we assume that deltas_batch is already a 2 column tensor.
            deltas_batch = tf.stack([deltas_batch,deltas_batch], axis=1)

        with tf.GradientTape() as tape:
            pi = self.call(observation_batch, training=True)
            old_pi = old_policy.call(observation_batch)
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
    

class ValueFunctionApproximator(Model):
    def __init__(self, input_shape, optimizer):
        """
        Keras Model subclassing tensorflow.keras.models. It represents the value function network for the agent(s) overcooked environment.
        It takes as input a tuple of observations (one for each agent) and outputs the state value fucntion for the agent(s).
        
        Args:
            input_shape (Tuple): The shape of the input observations for each agent.
            optimizer (optimizers): The optimizer to use for training the model. 
        Returns:
            None
        """
        super().__init__()
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.input_a = Input(shape=(self.input_shape))
        self.input_b = Input(shape=(self.input_shape))
        self.dense_1 = layers.Dense(64, activation='tanh')
        self.dense_2 = layers.Dense(128, activation='tanh')
        self.dense_3 = layers.Dense(64, activation='tanh')
        self.value_function = layers.Dense(1, name="value_function")
        self.build_model()
        self.printt = True

    
    def preprocess(self, obs: Union[Tuple[np.array, np.array], tf.Tensor]) -> tf.Tensor:
        """
        Preprocesses the observations by stacking them along the last dimension.

        Args:
            obs (Tuple or tf.Tensor): The observations to preprocess. Can be:
                - A tuple of two np.array (first and second agent's observations)
                - A tf.Tensor of shape (batch_size, 2, input_shape) (for batched input)
        Returns:
            tf.Tensor: A tensor of shape (batch_size, input_shape * 2) containing the concatenated observations.
        """
        if isinstance(obs, Tuple):
            obs = [obs] # to handle the case where obs_batch is a single observation

        obs_1, obs_2 = zip(*obs)
        obs_batch = tf.concat([tf.stack(obs_1), tf.stack(obs_2)], axis=-1)
        return obs_batch


    def call(self, obs: Union[Tuple, tf.Tensor], training=False) -> tf.Tensor:
        """
        Forward pass of the model.

        Args:
            obs (Tuple or tf.Tensor): The input observations,  Can be:
                - A tuple of two np.array (first and second agent's observations)
                - A tf.Tensor of shape (batch_size, 2, input_shape) (for batched input)
            training (bool, optional): Whether the model is in training mode. Defaults to False.
        Returns:
            tf.Tensor: A tensor of shape (batch_size, 1) containing the state value function for the agent(s).
        """
        x = self.preprocess(obs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        value_function = self.value_function(x)
        return value_function

    def build_model(self):
        """
        Builds the model by computing a forward pass with dummy input.
        """
        dummy_input = (
            tf.zeros((1, 96)),
            tf.zeros((1, 96))
        )
        _ = self(dummy_input)

    def train_step(self, delta: tf.Tensor, obs: Tuple):
        """
        WARNING: This method is DEPRECATED and will be removed in future versions.
        Use `train_batch` or `train_batch_PPO` instead for batch training.

        Implements the one-step semi-gradient weight update.

        Args:
            delta (tf.Tensor): The delta value to scale the gradient.
            obs (Tuple): The input observations for the agents.
        Returns:
            None
        """
        with tf.GradientTape() as tape:
            state_value = self.call(obs, training=True)

        grad_state_value = tape.gradient(state_value, self.trainable_weights)
        processed_gradient = [-tf.squeeze(delta)*grad for grad in grad_state_value]
        self.optimizer.apply_gradients(zip(processed_gradient, self.trainable_weights))

    def train_batch(self, deltas_batch: tf.Tensor, observation_batch: tf.Tensor): # deltas is a tf.Tensor of shape (batch_size,1)
        """
        Implements the batch semi-gradient weight update.

        Args:
            deltas_batch (tf.Tensor): A tensor of shape (batch_size,) containing the deltas for each observation.
            observation_batch (tf.Tensor): A tensor of shape (batch_size, 2, input_shape) containing the observations for each agent.
        Returns:
            None
        """
        with tf.GradientTape() as tape:
            state_value = self.call(observation_batch, training=True)
            processed_state_value = -deltas_batch * state_value

        grad_state_value = tape.gradient(processed_state_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grad_state_value, self.trainable_weights))

        
    def train_batch_PPO(self, rewards_batch: tf.Tensor, observation_batch, new_observation_batch, gamma:float): 
        """
        Implements the batch weight update through the MSE of the advantage function delta.

        Args:
            rewards_batch (tf.Tensor): A tensor of shape (batch_size,) or (batch_size, 2) containing the rewards for each observation.
            The rank of the tensor depends on whether the rewards are computed separately for the agents or not.
            observation_batch (tf.Tensor): A tensor of shape (batch_size, 2, input_shape) containing the observations for each agent.
            new_observation_batch (tf.Tensor): A tensor of shape (batch_size, 2, input_shape) containing the next observations for each agent.
            gamma (float): The discount factor for the estimated future state value.
        Returns:
            None
        """
        with tf.GradientTape() as tape:
            loss = 0.5*(tf.reduce_mean(tf.expand_dims(rewards_batch, axis=1) + gamma*self.call(new_observation_batch) - self.call(observation_batch)))**2
        grad_loss = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grad_loss, self.trainable_weights))


class MyAgent(Agent):
    """
    A custom agent for the Overcooked environment that uses a neural network policy and a value function approximator.
    It inherits from the Agent class.
    """
    def __init__(self, actor: Policy, old_policy: Policy, critic: ValueFunctionApproximator, idx: int, base_env: OvercookedEnv):
        """
        Initializes the agent with the actor netowrk, the old_policy, the critic, the index, and base_env.

        Args:   
            actor (Policy): The policy network for the agent.
            old_policy (Policy): The old policy network for the agent, used for PPO updates.
            critic (ValueFunctionApproximator): The value function approximator for the agent.
            idx (int): The index of the agent (0 or 1).
            base_env (OvercookedEnv): The base environment for the agent.
        """
        super().__init__()
        self.actor = actor
        self.old_policy = old_policy
        self.critic = critic
        self.idx = idx
        if not self.idx in [0,1]:
            raise AssertionError("The index of the agent must be either 0 or 1!")
        self.base_env = base_env
        self.update_old_policy()

    def action(self, obs: Union[Tuple[np.array, np.array], OvercookedState]) -> Tuple[Action, Dict]:
        """
        Computes the action to take based on the current observation and the associated probability distribution.

        Args:
            obs (Tuple(np.array,np.array) | OvercookedState): The observation of the environment. 
                Can be a tuple of two numpy arrays (one for each agent) or the OvercookedState object.
        Returns:
            Tuple[Action, Dict]: A tuple containing the action to take and a dictionary with action probabilities.
                The action is an instance of Action sampled from the policy network.
                The dictionary contains the action probabilities.
        """
        if isinstance(obs, OvercookedState):
            state = obs
            obs_from_state = self.base_env.featurize_state_mdp(state)
            obs = (obs_from_state[0],obs_from_state[1])

        action_probs = self.actor.call(obs)[self.idx].numpy()
        action = Action.sample(np.squeeze(action_probs))
        
        return (action, {'action_probs': action_probs})
    

    def update_old_policy(self):
        """
        Updates the old policy with the current actor's weights.
        This is used for PPO updates to ensure that the old policy 
        is used for computing the loss during the training process.
        """
        if self.old_policy is not None:
            self.old_policy.set_weights(self.actor.get_weights())
