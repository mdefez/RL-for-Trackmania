from setup.trackmania.env import TMEnv
import gymnasium as gym
from ...utils import new_features
from Learning.utils.new_features import distance_3d
import numpy as np



class NaiveEnv(TMEnv) :
    def init_features(self):
        self.previous_positions = []
        self.distance_with_n_previous = self.distance_treshold + 1  # pour pas que ça soit actif au début

        self.last_distance_to_finish = False
        self.last_position = [528, 688 - 16]
        self.last_direction = [0, 1]        # Initialize at first direction (at the starting point)

        self.progress_between_2_obs = 0

        self.crash = 0
        self.out_of_track = 0
        self.finish = 0

        self.first_obs = True   # If it's the first observation

    def __init__(self, delta_t):
        """         super().__init__(observation_space = gym.spaces.Dict({
            "speed": gym.spaces.Box(low=0, high=300, shape=(1,)),
            "finished": gym.spaces.Discrete(2),
            "distance_next_turn": gym.spaces.Box(low=0, high=1e4, shape=(1,)),
            "pos_x" : gym.spaces.Box(low=300, high=700, shape=(1,)),
            "pos_z" : gym.spaces.Box(low=600, high=1000, shape=(1,)),
            "distance_finish_line" : gym.spaces.Box(low=0, high=1e4, shape=(1,)),
            "angle_car_direction" : gym.spaces.Box(low=-180, high=180, shape=(1,)),
            "direction_x" : gym.spaces.Box(low=-1, high=1, shape=(1,)),
            "direction_z" : gym.spaces.Box(low=-1, high=1, shape=(1,))
        })) """
        super().__init__(observation_space = gym.spaces.Dict({
            "angle_car_direction": gym.spaces.Box(low=-3.14, high=3.14, shape=(1,)),
            "speed": gym.spaces.Box(low=0, high=300, shape=(1,)),
            "distance_closest_wall" : gym.spaces.Box(low=-100, high=100, shape=(1,)),
            "distance_next_turn": gym.spaces.Box(low=0, high=1e4, shape=(1,))
            }))

        self.delta_t = delta_t      # Temps en secondes entre deux obs

        self.treshold_stuck = 1 / self.delta_t      # S'il reste bloqué plus que 1 seconde
        self.distance_treshold = 1
        self.importance_crash = 50

        self.warning_walls = 0.4     # pénalité douce (zone danger)
        self.penalty_walls = 4.0      # pénalité forte (zone critique)

        self.reward_finish = 50
        self.penalty_out_of_track = -50
        self.w_angle = 0        # Weights for the reward, concerning the angle. Should be 2
        self.w_progress = 35    # Progress
        self.w_walls = 5        # Distance to closest walls
        self.w_time = 1         # Pénalité de temps pour forcer l'agent à finir rapidement

        self.init_features()

    def reset(self):
        self.controller.reset()
        self.init_features()

        return self._get_obs(), self._get_info()
    
    def _action_to_key(self, action: gym.spaces.Space) -> list[str]:
        if type(action) == int:
            return action
        return(list(action))
    
    def _get_info(self):
        return {}
    
    def _is_truncated(self, obs):
        return False
    
    def _is_terminated(self, obs):
        if obs["finished"]:
            return True
        
        if self.out_of_track:
            return True
        
        if self.crash:  
            return True

        return False

    def _compute_reward(self, obs) -> float:    
        finished = obs["finished"]
        distance_finish_line = obs["distance_finish_line"]
        pos_x, pos_z = obs["pos_x"], obs["pos_z"]
        angle_car_direction = obs["angle_car_direction"]
        distance_walls = obs["distance_closest_wall"]

        # On met à jour les paramètres paramètres permettant de calculer des parts de la reward
        if not self.first_obs:
            self.progress_between_2_obs =  self.last_distance_to_finish - distance_finish_line

        else:
            self.first_obs = False

        self.last_distance_to_finish = distance_finish_line
        self.last_pos = [pos_x, pos_z]

        # On cape les valeurs si jamais il y a des outliers et on normalise pour que ce soit environ égal à 1
        self.progress_between_2_obs = np.clip(self.progress_between_2_obs, -20, 20) / 20

        # On détecte s'il y a crash
        self.previous_positions.append(self.current_position)
        if len(self.previous_positions) > self.treshold_stuck:
            self.distance_with_n_previous = distance_3d(self.current_position, self.previous_positions[0])
            self.previous_positions.pop(0)
    
            if self.distance_with_n_previous < self.distance_treshold:  # S'il y a crash
                self.crash = 1

        # Pénalité liée à la distance au mur le plus proche
        d_safe = 2.5
        d_crit = 1
        d_abs = abs(distance_walls)

        if d_abs > d_safe:
            reward_walls = 0
        elif d_crit < d_abs <= d_safe:
            x = (d_safe - d_abs) / (d_safe - d_crit)  # x ∈ (0,1)
            reward_walls = self.warning_walls * (x ** 2)
        elif d_abs <= d_crit:
            reward_walls = self.penalty_walls

        reward_walls = - reward_walls / self.penalty_walls        # Normalisation et passage en négatif

        # reward liée à l'angle entre trajectoire classique et direction de la voiture
        reward_angle = - abs(angle_car_direction)

        # Reward exceptionnelle
        if finished:
            self.finish = 1  # gros bonus de fin

        self.out_of_track = new_features.out_of_track([pos_x, pos_z])
        if self.out_of_track == 1:
            self.out_of_track = 1

        reward = (
            self.w_progress  * self.progress_between_2_obs +
            self.w_angle * reward_angle 
            + self.w_walls  * reward_walls 
            - self.importance_crash * self.crash +
            self.reward_finish * self.finish + 
            + self.penalty_out_of_track * self.out_of_track
            - self.w_time

        )

        return reward

    def _data_to_observation(self, data):
        # Build and add new features (make it a cool and packed function)
        position = data["vehicleData"]["position"]
        x, z = position[0], position[2] 
        position = [x, z]

        # Compute the distance to next turn and the turn's orientation
        self.current_position = position
        distance_next_turn = new_features.distance_to_next_turn(position)
        data["distance_next_turn"] = distance_next_turn

        # Compute if it arrived
        arrived = new_features.arrived(position)
        data["arrived"] = arrived

        # Compute the real distance to finish
        distance_finish_line, distance_center_line = new_features.distance_finish_line(position, points_per_segment=100)
        data["distance_finish_line"] = distance_finish_line
        data["distance_center_line"] = distance_center_line

        # Computes the (signed, positive if right negative if left) distance to the closest wall (or point de corde during turns)
        distance_closest_wall = new_features.processed_distance_walls(position)
        data["distance_closest_wall"] = distance_closest_wall

        # Compute the vehicle direction (as cos and sin) and angle between the car and the target direction
        angle_car_direction, current_direction = new_features.difference_angle_vehicle(last_position=self.last_position,
                                                                    position=position, 
                                                                    last_direction=self.last_direction, dt=self.delta_t)
        data["angle_car_direction"] = angle_car_direction
        data["direction_x"] = current_direction[0]
        data["direction_z"] = current_direction[1]

        self.last_direction = current_direction
        self.last_position = position

        # Make it a simple dict with one level of keys/values. At that point every values should be float. WIP
        obs = new_features.keep_relevant_features(data)

        return obs

