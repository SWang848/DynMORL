import numpy as np
from minecart import Minecart

import gym
from gym.spaces import Box
import os

class PixelMinecart(gym.ObservationWrapper):

    def __init__(self, env):
        # don't actually display pygame on screen
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        super(PixelMinecart, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(480, 480, 3), dtype=np.uint8)

    def observation(self, obs):
        obs = self.render('rgb_array')
        return obs



# Generate minecart from configuration file (2 ores + 1 fuel objective, 5 mines)
json_file = "mine_config_det.json"
env = Minecart.from_json(json_file)

pixel = PixelMinecart(env)

# # Or alternatively, generate a random instance
# env = Minecart(mine_cnt=5,ore_cnt=2,capacity=1)

# Initial State
s_t = env.reset()
s_t_pixel = pixel.observation(s_t)
print(s_t_pixel.shape)
print(env.action_space())

# Note that s_t is a dictionary containing among others the state's pixels but also the cart's position, velocity, etc...
# s_t = s_t["pixels"]

# flag indicates termination
terminal = False

while not terminal:
  # randomly pick an action
  a_t = np.random.randint(env.a_space)

  # apply picked action in the environment
  s_t1, r_t, terminal, _= env.step(a_t)
  print(env.step(a_t))
  # s_t1 = s_t1["pixels"]
  # print(s_t1)
  # update state
  s_t = s_t1
    
  print("Taking action", a_t, "with reward", r_t)
  
env.reset()
