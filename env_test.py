import gym


env = gym.make('Hopper-v2')

print(env.action_space,env.observation_space)