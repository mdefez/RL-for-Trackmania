keys = ["W", "A", "S", "D"]
ACTIONS = [
    (1, 0),   # W
    (1, 1),   # WD
    (1, -1),  # WA
    (0, 1),   # D
    (0, -1),  # A
]
n_actions = len(ACTIONS)

link_action_key = {(1, 0) : "W", (1, 1) : "WD", (1, -1) : "WA", (0, 1) : "D", (0, -1) : "A"}


FEATURES = [
    "speed",
    "finished",
    "distance_next_turn",
    "pos_x",
    "pos_z",
    "distance_finish_line",
    "angle_car_direction",
    "direction_x",
    "direction_z",
    "distance_center_line"
]

# Features we feed the NN 
features_nn = ["angle_car_direction", "speed", "distance_closest_wall", "distance_next_turn"]
ref = [0.3, 20, 3, 100]        # to normalize around 1



import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from collections import deque


wandb.init(project="RL-TM")

from setup.trackmania.env import TMEnv

class NaiveModel :
    def __init__(self, env : TMEnv, weights_path = False, testing = False):
        self.env = env
        self.testing = testing
        self.history = [] 
        self.gamma = 0.99

        self.delta_t = env.delta_t      # Time between 2 commands
        self.finish_time = 0            # Save the finish time

        self.history_action = []    # Garde en mémoire les actions effectuées
        self.history_reward = []

        self.replay_buffer = deque(maxlen=30000)  # Taille max du buffer en nombre de steps
        self.batch_size = 32
        self.warmup_steps = 2000

        self.count_log_loss = 0
        self.log_every = 10
 
        self.epsilon_testing = 0
        self.epsilon_min = 0.1
        self.epsilon = 0.6

        self.epsilon_decay = 0.9995        

        self.save_weights_every = 1e3
        self.count_save_weights = 0

        self.q_network = QNetwork(len(self.env.observation_space), n_actions)
        if type(weights_path) == str:
            print("model loaded")
            self.q_network.load_state_dict(torch.load(weights_path))

        self.target_network = QNetwork(len(self.env.observation_space), n_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()



    def learn(self, total_timesteps):
        first_obs, _ = self.env.reset()
        self.history.append(first_obs)
        self.history_reward.append(0)

        terminated = False
        for t in range(total_timesteps):

            if self.testing == False:
                self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

            else:       # Testing
                self.epsilon = self.epsilon_testing   

            ## Compute action
            if terminated:
                if reward < 0:
                    self.finish_time = 60   # Default value if we terminate by hitting a wall
                wandb.log({"Finish time": self.finish_time})
                self.finish_time = 0

                next_action = 0x0E
                first_obs, _ = self.env.reset()
                self.history = [] 
                self.history_action = []
                self.history_reward = []

            else:
                next_action = self.find_action()
                self.finish_time += self.delta_t

            ## Apply action
            obs, reward, terminated, truncated, info = self.env.step(next_action)

            self.history_reward.append(reward)
            self.history.append(obs)

            # Si on a fait une première action, on ajoute la transition d'état dans le buffer
            if len(self.history) >= 2:
                state_t, action_t, reward, state_tp1 = self.history[-2], self.history_action[-1], self.history_reward[-1], self.history[-1]
                self.replay_buffer.append((state_t, action_t, reward, state_tp1, terminated))

            # Si le buffer est suffisament grand, on train
            if len(self.replay_buffer) >= self.warmup_steps:
                self.train_from_buffer()

                # Save the model's weights from time to time
                self.count_save_weights += 1
                if self.count_save_weights == self.save_weights_every:
                    print("model saved")
                    torch.save(self.q_network.state_dict(), "../Learning/models/without_angle.pth")
                    self.count_save_weights = 0



            # De temps en temps on update le target model
            if t % 1000 == 0:
                print("target model updated")
                self.target_network.load_state_dict(self.q_network.state_dict())

    # Convertit l'observation (dictionnaire) en tenseur de float
    def obs_to_tensor(self, obs):
        state = torch.tensor(
            [obs[features_nn[k]]/ref[k] for k in range(len(features_nn))],
            dtype=torch.float32
        )

        return state

    def find_action(self):
        state = self.obs_to_tensor(self.history[-1]).unsqueeze(0)

        if random.random() < self.epsilon:
            action_idx = random.randrange(n_actions)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                action_idx = torch.argmax(q_values).item()

        self.history_action.append(action_idx)

        return link_action_key[ACTIONS[action_idx]]     # output as str, we seek keys
    

    def train_from_buffer(self):
        batch = random.sample(self.replay_buffer, self.batch_size)

        states, actions, rewards, next_states, terminateds = zip(*batch)

        states = torch.stack([self.obs_to_tensor(s) for s in states])
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack([self.obs_to_tensor(s) for s in next_states])
        terminateds = torch.tensor(terminateds, dtype=torch.float32)

        # Q(s,a)
        q_values = self.q_network(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            y = rewards + self.gamma * next_q * (1 - terminateds)   # On ajoute rien si c'est terminé

        loss = F.mse_loss(q_sa, y)

        self.q_network.optimizer.zero_grad()
        loss.backward()
        self.q_network.optimizer.step()





# Prend en entrée les données et renvoie une estimation de la q value pour toutes les actions possibles, c'est donc un tuple
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=128):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_actions)
        self.lr = 1e-4

        self.log_every = 10
        self.count_log_loss = 0

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, state):
        """
        state  : tensor (state_dim)
        """
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return self.out(x)
    
