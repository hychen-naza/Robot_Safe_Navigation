from __future__ import print_function
from __future__ import absolute_import

# python modules
import argparse
import numpy as np
import time
# project files
import dynamic_obstacle
import bounds
import robot # double integrator robot
import simu_env
import runner
import param
from turtle_display import TurtleRunnerDisplay
from ssa import SafeSetAlgorithm


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
    
    # rl policy
    robot_state_size = 4 #(x,y,v_x,v_y)
    robot_action_size = 2
    nearest_obstacle_state_size = 2 #(delta_x, delta_y)
    state_dim = robot_state_size + nearest_obstacle_state_size
    env = simu_env.Env(display, **(env_params))
    # ssa
    safe_controller = SafeSetAlgorithm(max_speed = env.robot_state.max_speed)

    # parameters
    max_steps = int(1e6)
    start_timesteps = 2e3
    episode_reward = 0
    episode_num = 0
    total_rewards = []
    total_steps = 0
    # dynamic model parameters
    fx = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
    gx = np.array([[1,0],[0,1],[1,0],[0,1]])
    state, done = env.reset(), False
    collision_num = 0
    failure_num = 0
    success_num = 0

    reward_records = []

    for t in range(max_steps):
      action = env.random_action()
      env.display_start()
      # ssa parameters
      unsafe_obstacle_ids, unsafe_obstacles = env.find_unsafe_obstacles(env.min_dist * 6)
      if (enable_ssa):
        action, is_safe, is_unavoidable, danger_obs = safe_controller.get_safe_control(state, unsafe_obstacles, fx, gx, action)
      else:
         is_safe = False
      # take safe action
      s_new, reward, done = env.step(action, is_safe, unsafe_obstacle_ids) 

      env.display_end()
      state = s_new

      if (done and reward == -500):          
        collision_num += 1      
      elif (done and reward == 2000):
        success_num += 1
      elif (done):
        failure_num += 1
      time.sleep(0.05)
      if (done):      
        total_steps += env.cur_step
        print(f"Train: episode_num {episode_num}, total_steps {total_steps}")
        total_rewards.append(episode_reward)
        episode_reward = 0
        episode_num += 1
        state, done = env.reset(), False
    return reward_records



if __name__ == '__main__':
    args = parser().parse_args()
    navigate(display_name = args.display, enable_ssa = args.enable_ssa)


