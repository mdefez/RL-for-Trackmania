import pandas as pd
import math
import numpy as np

def keep_relevant_features(data):
    vd = data["vehicleData"]
    return {
        "speed": vd["speed"],
        "finished": data["arrived"],
        "distance_next_turn" : data["distance_next_turn"],
        "pos_x" : vd["position"][0],
        "pos_z" : vd["position"][2],
        "distance_finish_line" : data["distance_finish_line"],
        "angle_car_direction" : data["angle_car_direction"],
        "direction_x" : data["direction_x"],
        "direction_z" : data["direction_z"],
        "distance_center_line" : data["distance_center_line"],
        "distance_closest_wall" : data["distance_closest_wall"]
    }


# Calcule où on est sur la map et la distance au prochain virage
def read_map():
    return pd.read_csv("../Learning/utils/ParseTMMap/blocks.csv")

def find_block_index(car_pos, df):
    """
    Identifie le bloc (ligne du DataFrame) dans lequel se trouve la voiture,
    en utilisant uniquement les coordonnées de grille. Suppose une map simple

    car_pos : (x, y, z) coordonnées monde de la voiture (float)
    df      : DataFrame contenant les colonnes X, Y, Z (coordonnées grille)

    Retour :
        index du bloc dans df, ou None si aucun bloc ne correspond
    """

    car_x, car_z = car_pos

    # Conversion monde -> grille
    bx = int(car_x // 32)
    bz = int(car_z // 32)

    # Recherche exacte du bloc
    match = df[
        (df["X"] == bx) &
        (df["Z"] == bz)
    ]

    # If no match, we return the starting block to avoid any error
    if match.empty:
        return 0

    return match.index[0]

def next_curve_center_point_world_pos(current_idx, df):
    """
    Retourne (x, z) monde du début du prochain bloc Curve et la direction du prochain virage
    """

    for i in range(current_idx + 1, len(df)):
        if "Curve" in df.loc[i, "Block"]:

            if df.loc[i, "Dir"] == "East":
                turn = "right"
            else:
                turn = "left"

            return (
                df.loc[i, "cx"],
                df.loc[i, "cz"],
                turn
            )
        
    return None


def distance_3d(p1, p2):
    """
    calcule la distance entre deux points monde
    """

    dx = p2[0] - p1[0]

    dz = p2[1] - p1[1]
    return (dx*dx + dz*dz) ** 0.5


def distance_to_next_turn(car_pos):
    """
    Calcule la distance entre la position actuel et l'entrée du prochain virage (distance > 0 si virage à droite, <0 sinon)
    car_pos : (x, z) coordonnées monde de la voiture (float)
    df      : DataFrame contenant les colonnes X, Z (coordonnées grille)

    """
    # On lit les données
    df = read_map()

    value_no_turn = 100     # infinite

    # On cherche le bloc actuel
    current_idx = find_block_index(car_pos, df)
    

    # entrée du prochain virage
    next_turn = next_curve_center_point_world_pos(current_idx, df)
    if next_turn is None:
        return value_no_turn

    distance_to_center = distance_3d(car_pos, [next_turn[0], next_turn[1]])
    distance_to_entry_of_turn = distance_to_center - 16

    if next_turn[2] == "right":
        return distance_to_entry_of_turn
    
    elif next_turn[2] == "left":
        return - distance_to_entry_of_turn


# Calcule si on est arrivé en fonction des coordonnées

def arrived(position):
    center_finish = (528.0, 912.0)

    if distance_3d(position, center_finish) < 16:
        return True 
    
    return False


def out_of_track(position):
    min_z = 688
    tol = 3
    coord_z = position[1]

    if coord_z + tol < min_z:
        return True
    
    return False



# Calcule la distance restante à l'arrivée

def interpolate(p1, p2, n):
    x = np.linspace(p1[0], p2[0], n)
    y = np.linspace(p1[1], p2[1], n)
    return list(zip(x, y))

# Relie les points de la liste en interpolant avec des points intermédiaires
def generate_path(points, points_per_segment=20):
    path = []

    for i in range(len(points) - 1):
        segment = interpolate(points[i], points[i + 1], points_per_segment)

        if i > 0:
            segment = segment[1:]  # éviter les doublons
        path.extend(segment)


    return path


def closest_point(path, position):  # On trouve le point interpolé le plus proche de la position
    path_np = np.array(path)            # shape (N, 2)
    pos_np = np.array(position)          # shape (2,)

    diff = path_np - pos_np
    dist2 = np.einsum('ij,ij->i', diff, diff)  # distances au carré
    idx = np.argmin(dist2)

    return path[idx], idx, np.sqrt(dist2[idx])

def compute_ds(path):
    p0 = np.array(path[0])
    p1 = np.array(path[1])
    return np.linalg.norm(p1 - p0)

def distance_finish_line(position, points_per_segment): # On regarde le point le plus proche dans le bloc courant (on a besoin du bloc d'avant et d'après pour interpoler)
    position = (position[0], position[1])   # on garde que x et z

    df = read_map()
    idx = find_block_index(position, df)

    remaining_blocks = len(df) - 1 - idx - 1

    if idx == 0:
        points = list(zip(df["cx"], df["cz"]))[0 : 2]
    
    elif idx == len(df)-1:
        points = list(zip(df["cx"], df["cz"]))[len(df)-2 : len(df)]
        remaining_blocks = 0

    else:
        points = list(zip(df["cx"], df["cz"]))[idx - 1 : idx + 2]

    path = generate_path(points, points_per_segment=points_per_segment)
    distance_between_points = compute_ds(path)
    closest_interpolated_point = closest_point(path, position)
    coord_points, idx_in_path, dist_to_centerline = closest_interpolated_point


    remaining_points_in_blocks = len(path) - 1 - idx_in_path


    number_of_points_to_finish_line = remaining_points_in_blocks + remaining_blocks * points_per_segment

    final_distance = number_of_points_to_finish_line * distance_between_points + dist_to_centerline

    return final_distance, dist_to_centerline



# Calcul de l'angle du véhicule et de la vitesse dans la "bonne" direction
def velocity_vector(p_prev, p_curr, dt, eps=1e-8):
    vx = (p_curr[0] - p_prev[0]) / dt
    vz = (p_curr[1] - p_prev[1]) / dt

    norm = np.sqrt(vx*vx + vz*vz)
    if norm < eps:
        return (0.0, 0.0)  

    return (vx, vz)

def vehicle_direction(p_prev, p_curr, last_direction, dt, eps=1e-8):
    velocity = velocity_vector(p_prev, p_curr, dt, eps=1e-8)
    norm = np.linalg.norm(velocity)

    if norm > eps:
        direction = velocity / norm
    else:
        return last_direction

    return direction

# Return the "target" vehicle direction
def good_direction(position):
    df = read_map()

    block_idx = find_block_index(position, df)

    race_direction = df.loc[block_idx, "Dir"]
    if "Curve" in df.loc[block_idx, "Block"]:
        dico =  {"East" : [-1/np.sqrt(2), 1/np.sqrt(2)], "West" : [-1/np.sqrt(2), 1/np.sqrt(2)], "South" : [1/np.sqrt(2), 1/np.sqrt(2)]}
    else:
        dico = {"North" : [0, 1], "East" : [-1, 0], "West" : [1, 0], "South" : [0, -1]}

    return dico[race_direction]

# Return the angle between the target direction and the current direction. Between -180 & 180
def difference_angle_vehicle(last_position, position, last_direction, dt):
    target = good_direction(position)
    current = vehicle_direction(p_prev=last_position, p_curr=position, last_direction = last_direction, dt=dt)

    cross = target[0]*current[1] - target[1]*current[0]
    dot   = target[0]*current[0] + target[1]*current[1]

    return math.atan2(cross, dot), current



### Détecte la distance au mur le plus proche en ligne droite, ou l'opposé du point de corde en virage

def _distance_forward(coord, wall_left, wall_right):   # Intermediate function

    distance_left = abs(coord - wall_left) - 4  # 4 is linked to the widh of the wall
    distance_right = abs(coord - wall_right) - 4

    if distance_left < distance_right:
        return - distance_left

    else:
        return distance_right

# Return the distance and orientation (left or right) of the closest wall in the current idx. The distance is negative if the closest wall is on the left, positive otherwise
def distance_closest_wall(position):
    df = read_map()
    idx = find_block_index(position, df)
    if "Curve" in df.loc[idx, "Block"]: # Renvoie la distance au mur extérieur seulement (le seul vrai danger). En pratique calcule la distance au point de corde (ça marche car le virage forme un cercle quasi parfait)
        cx, cz = df.loc[idx, ["cx", "cz"]]
        x, z = position[0], position[1]

        if df.loc[idx, "Dir"]== "East":
            cx -= 14        # On considère le point de corde, pas le centre du bloc
            cz -= 14 
            turn = "right"
        elif df.loc[idx, "Dir"]== "West":
            cx += 14
            cz += 14
            turn = "left"
        elif df.loc[idx, "Dir"]== "South":
            cx += 14
            cz -= 14
            turn = "left"

        dx = x - cx
        dy = z - cz
        r = math.hypot(dx, dy)  # distance au centre du cercle

        distance_wall = abs(r - 25.5) # 25.5 c'est le rayon du virage

        if turn == "right": # Ca tourne à droite donc le mur est à gauche
            return - distance_wall
    
        elif turn == "left":
            return distance_wall


    
    if df.loc[idx, "Dir"]== "North":    # Les murs sont à x fixé. Mur gauche à x positif
        pos_x = position[0]
        pos_wall_left = df.loc[idx, "cx"] + 16
        pos_wall_right = df.loc[idx, "cx"] - 16

        return _distance_forward(pos_x, pos_wall_left, pos_wall_right)
    
    elif df.loc[idx, "Dir"]== "South":    # Les murs sont à x fixé. Mur gauche à x négatif
        pos_x = position[0]
        pos_wall_left = df.loc[idx, "cx"] - 16
        pos_wall_right = df.loc[idx, "cx"] + 16

        return _distance_forward(pos_x, pos_wall_left, pos_wall_right)
        
    
    elif df.loc[idx, "Dir"] == "East":    # Les murs sont à z fixé. Mur gauche à z positif
        pos_z = position[1]
        pos_wall_left = df.loc[idx, "cz"] + 16
        pos_wall_right = df.loc[idx, "cz"] - 16

        return _distance_forward(pos_z, pos_wall_left, pos_wall_right)
    
    
    elif df.loc[idx, "Dir"] == "West":    # Les murs sont à z fixé. Mur gauche à z négatif
        pos_z = position[1]
        pos_wall_left = df.loc[idx, "cz"] - 16
        pos_wall_right = df.loc[idx, "cz"] + 16

        return _distance_forward(pos_z, pos_wall_left, pos_wall_right)
    

def processed_distance_walls(position): # Clipped distance (irrelevant if too big, the agent should only care when it's small)
    distance = distance_closest_wall(position)
    return np.clip(distance, -5, 5)
