'''
    Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
'''
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

class Actor(keras.Model):
    '''
        The actor in TD3. 
    '''
    def __init__(self, action_dim, max_action):
        super().__init__()

        self.l1 = keras.layers.Dense(256, activation="relu")
        self.l2 = keras.layers.Dense(256, activation="relu")
        self.l3 = keras.layers.Dense(action_dim)
        self.max_action = max_action


    def call(self, state):
        '''
            Returns the tanh normalized action
            Ensures that output <= self.max_action
        '''
        a = self.l1(state)
        a = self.l2(a)
        return self.max_action * keras.activations.tanh(self.l3(a))

class Critic(keras.Model):
    '''
        The critics in TD3. 
    '''
    def __init__(self):
        super().__init__()

        # Q1 architecture
        self.l1 = keras.layers.Dense(256, activation="relu")
        self.l2 = keras.layers.Dense(256, activation="relu")
        self.l3 = keras.layers.Dense(1)

        # Q2 architecture
        self.l4 = keras.layers.Dense(256, activation="relu")
        self.l5 = keras.layers.Dense(256, activation="relu")
        self.l6 = keras.layers.Dense(1)


    def call(self, state, action):
        '''
            Returns the output for both critics. Using during critic training.
        '''
        sa = tf.concat([state, action], 1)

        q1 = self.l1(sa)
        q1 = self.l2(q1)
        q1 = self.l3(q1)

        q2 = self.l4(sa)
        q2 = self.l5(q2)
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        '''
            Returns the output for only critic 1. Used to compute actor loss.
        '''
        sa = tf.concat([state, action], 1)
        
        q1 = self.l1(sa)
        q1 = self.l2(q1)
        q1 = self.l3(q1)
        return q1

    
class TD3():
    '''
        The TD3 main class. 
    '''
    def __init__(
        self,
        state_dim,
        action_dim,
        max_acc,
        max_steering,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):
        self.max_acc = max_acc
        self.max_steering = max_acc
        self.state_dim = state_dim
        self.actor = Actor(action_dim, np.array([max_acc, max_steering]))
        self.action_dim = action_dim
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=3e-4)

        self.critic = Critic()
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=3e-4)
        
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.expl_noise = 0.4

        self.total_it = 0
        self.loss = keras.losses.MeanSquaredError()

    def select_action(self, state):
        '''
            Select action for a single state.
            state: np.array, size (state_dim, )
            output: np.array, size (action_dim, )
        '''
        state = tf.convert_to_tensor(state.reshape(1, -1))
        state = tf.cast(state, tf.float32) 
        acc_expl_noise = np.random.normal(0, self.max_acc * self.expl_noise)
        steering_expl_noise = np.random.normal(0, self.max_acc * self.expl_noise)
        expl_action = self.actor(state).numpy().flatten() + np.array([acc_expl_noise, steering_expl_noise])
        expl_action[0] = max(min(expl_action[0], self.max_acc), -self.max_acc)
        expl_action[1] = max(min(expl_action[1], self.max_acc), -self.max_acc)
        return expl_action

    def select_action_batch(self, state):
        '''
            Select action for a batch of states.
            state: np.array, size (batch_size, state_dim)
            output: np.array, size (batch_size, action_dim)
        '''
        if not tf.is_tensor(state):
            state = tf.convert_to_tensor(state)
        return self.actor(state).numpy()


    def train_on_batch(self, state, action, next_state, reward, not_done):
        '''
            Trains both the actor and the critics on a batch of transitions.
            state: tf tensor, size (batch_size, state_dim)
            action: tf tensor, size (batch_size, action_dim)
            next_state: tf tensor, size (batch_size, state_dim)
            reward: tf tensor, size (batch_size, 1)
            not_done: tf tensor, size (batch_size, 1)
        '''
        # Select action according to policy and add clipped noise
        noise = tf.clip_by_value(tf.random.normal(action.shape) * self.policy_noise,
                                 -self.noise_clip, self.noise_clip)

        next_action = tf.clip_by_value(self.actor_target(next_state) + noise,
                                       [-self.max_acc, -self.max_acc], [self.max_acc, self.max_acc])
        # Compute the target Q value # 
        q1, q2 = self.critic_target.call(next_state, next_action)
        q1 = q1.numpy()
        q2 = q2.numpy()

        target_Q = reward + self.discount * np.minimum(q1, q2) 
    
        with tf.GradientTape() as tape:
            # Get current Q estimates
            current_Q1, current_Q2 = self.critic.call(state, action)
            # Compute critic loss
            loss = self.loss(target_Q, current_Q1) + self.loss(target_Q, current_Q2)
        # Optimize the critic
        gradients = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_weights))     

        # Delayed policy updates
        if (self.total_it % self.policy_freq == 0):
            # Compute actor losses
            with tf.GradientTape() as tape:
                action = self.actor(state)
                current_Q1 = self.critic.Q1(state, action)
                loss = -1 * tf.reduce_mean(current_Q1)   
            gradients = tape.gradient(loss, self.actor.trainable_weights)
            self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_weights))

            # Update the frozen target models
            cur_weights = self.actor.get_weights()
            target_weights = self.actor_target.get_weights()
            new_target_weights = []
            for cur_weight, target_weight in zip(cur_weights, target_weights):
                new_target_weights.append(target_weight*(1-self.tau) + cur_weight*self.tau)
            self.actor_target.set_weights(new_target_weights)
            
            cur_weights = self.critic.get_weights()
            target_weights = self.critic_target.get_weights()
            new_target_weights = []
            for cur_weight, target_weight in zip(cur_weights, target_weights):
                new_target_weights.append(target_weight*(1-self.tau) + cur_weight*self.tau)
            self.critic_target.set_weights(new_target_weights)

        self.total_it += 1