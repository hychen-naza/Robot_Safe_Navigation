from __future__ import print_function
from __future__ import absolute_import

# python modules
import argparse
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras

# project files
import dynamic_obstacle
import bounds
import robot 
import simu_env
import runner
import param
from utils import ReplayBuffer
from td3 import TD3
from turtle_display import TurtleRunnerDisplay
from ssa import SafeSetAlgorithm

class RND(keras.Model):
    '''
        RND
    '''
    def __init__(self):
        super().__init__()
        self.l1 = keras.layers.Dense(128, activation="relu")
        self.l2 = keras.layers.Dense(128)

    def call(self, state):
        '''
            Returns the output for both critics. Using during critic training.
        '''
        if not tf.is_tensor(state):
            state = tf.convert_to_tensor(state)
        q1 = self.l1(state)
        q1 = self.l2(q1)
        return q1


def display_for_name( dname ):
    # choose none display or visual display
    if dname == 'turtle':
        return TurtleRunnerDisplay(800,800)
    else:
        return runner.BaseRunnerDisplay()

def parser():
    prsr = argparse.ArgumentParser()
    prsr.add_argument( '--display',
                       choices=('turtle','text','none'),
                       default='turtle' )
    prsr.add_argument( '--ssa',dest='enable_ssa', action='store_true')
    prsr.add_argument( '--no-ssa',dest='enable_ssa', action='store_false')
    return prsr

def run_kwargs( params ):
    in_bounds = bounds.BoundsRectangle( **params['in_bounds'] )
    goal_bounds = bounds.BoundsRectangle( **params['goal_bounds'] )
    min_dist = params['min_dist']
    ret = { 'field': dynamic_obstacle.ObstacleField(),
            'robot_state': robot.DoubleIntegratorRobot( **( params['initial_robot_state'] ) ),
            'in_bounds': in_bounds,
            'goal_bounds': goal_bounds,
            'noise_sigma': params['noise_sigma'],
            'min_dist': min_dist,
            'nsteps': 1000 }
    return ret

def navigate(display_name, enable_ssa):
    try:
        params = param.params
    except Exception as e:
        print(e)
        return
    display = display_for_name(display_name)
    env_params = run_kwargs(params)
    env = simu_env.Env(display, **(env_params))

    # ssa
    safe_controller = SafeSetAlgorithm(max_speed = env.robot_state.max_speed, dmin = env.min_dist * 2)
    
    # RL policy parameters
    max_steps = int(1e6)
    robot_state_size = 4 #(x,y,v_x,v_y)
    robot_action_size = 2 #(a_x,a_y)
    nearest_obstacle_state_size = 2 #(delta_x, delta_y)
    state_dim = robot_state_size + nearest_obstacle_state_size

    policy_replay_buffer = ReplayBuffer(state_dim = state_dim, action_dim = robot_action_size, max_size=max_steps)
    policy = TD3(state_dim, robot_action_size, env.max_acc, env.max_acc)
    # Random Network Distillation
    rnd_fixed = RND()
    rnd_train = RND()
    rnd_optimizer = keras.optimizers.Adam(learning_rate=3e-4)
    rnd_loss = keras.losses.MeanSquaredError()

    # dynamic model parameters
    episode_num = 0
    collision_num = 0
    failure_num = 0
    success_num = 0
    sensing_range = env.min_dist * 6
    fx = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
    gx = np.array([[1,0],[0,1],[1,0],[0,1]])
    state, done = env.reset(), False

    for t in range(max_steps):
      # train the random network prediction when using rnd
      if (t > 1024):
        with tf.GradientTape() as tape:
          state_batch, action_batch, next_state_batch, reward_batch, not_done_batch =  policy_replay_buffer.sample(256)
          q_fixed = rnd_fixed.call(state_batch)
          q_train = rnd_train.call(state_batch)
          loss = rnd_loss(q_fixed, q_train)
          gradients = tape.gradient(loss, rnd_train.trainable_weights)
          rnd_optimizer.apply_gradients(zip(gradients, rnd_train.trainable_weights))
      action = policy.select_action(state)
      env.display_start()
      obstacle_ids, obstacles = env.detect_obstacles(sensing_range)
      # compute safe control
      if (enable_ssa):
        action, is_safe = safe_controller.get_safe_control(state[:4], obstacles, fx, gx, action)
      else:
         is_safe = False
      s_new, reward, done = env.step(action, is_safe, obstacle_ids) 
      env.display_end()

      if (done and reward == -500):          
        collision_num += 1      
      elif (done and reward == 2000):
        success_num += 1
      elif (done):
        failure_num += 1
      time.sleep(0.05)
      if (done):
        episode_num += 1
        print(f"Train: episode_num {episode_num}, success_num {success_num}, collision_num {collision_num}, step {t}")
        state, done = env.reset(), False
      
      # add the novelty to reward when using rnd
      rnd_state = tf.convert_to_tensor(state.reshape(1, -1))
      q_fixed = rnd_fixed.call(rnd_state)
      q_train = rnd_train.call(rnd_state)                    
      loss = np.sum(np.square(q_fixed - q_train))      
      reward += loss    
      policy_replay_buffer.add(state, action, s_new, reward, done)
      state = s_new

      # train policy
      if (policy_replay_buffer.size > 1024):
        state_batch, action_batch, next_state_batch, reward_batch, not_done_batch =  [np.array(x) for x in policy_replay_buffer.sample(256)]
        policy.train_on_batch(state_batch, action_batch, next_state_batch, reward_batch, not_done_batch)



if __name__ == '__main__':
    args = parser().parse_args()
    navigate(display_name = args.display, enable_ssa = args.enable_ssa)


