import sys
sys.path.append("..")

## Random model
# from stable_baselines3 import SAC
# from Learning.agents.random.random_agent import RandomEnv

# env = RandomEnv()
# model = SAC("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1000)
# print(model.predict(env.reset()[0]))

## Line Ref model
# from stable_baselines3 import SAC
# from Learning.environments.ref_line.ref_line_agent import RefLineEnv

#env = RefLineEnv("./data/clean_blocks.csv")
#model = SAC("MultiInputPolicy", env, verbose=1)
#model.learn(total_timesteps=100000)


## Import Naive Model
from Learning.models.naive import NaiveModel
from Learning.environments.naive.naive_env import NaiveEnv


env = NaiveEnv(delta_t = 0.1)       # Temps en secondes
model = NaiveModel(env, weights_path="../Learning/models/without_walls.pth", testing = True) # "../Learning/models/with_time.pth"  
model.learn(total_timesteps=int(1e5))
