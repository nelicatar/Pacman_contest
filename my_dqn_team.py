import os
import sys

cd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cd)

# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point
import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle

TRAINING = False
NETWORK_DISTANCE = 0.7

def get_action_index(action):
    if action == "South":
        return 0
    if action == "East":
        return 1
    if action == "North":
        return 2
    if action == "West":
        return 3
    return 4

def invert_action(action):
    if action == "South":
        return "North"
    if action == "North":
        return "South"
    if action == "West":
        return "East"
    if action == "East":
        return "West"
    return action

def get_action_name(index):
    actions = ["South", "East", "North", "West", "Stop"]
    return actions[int(index)]

def get_attacker_reward(agent, s1, s2):
    opponents = agent.get_opponents(s1)
    killed = 0
    agent_state_1 = s1.data.agent_states[agent.index]
    agent_state_2 = s2.data.agent_states[agent.index]

    pos2 = agent.intify(agent_state_2.configuration.pos)
    dist2 = agent.closest_food(s2, pos2)[0]
    pos1 = agent.intify(agent_state_1.configuration.pos)
    dist1 = agent.closest_food(s1, pos1)[0]

    food_dist_d =  (dist1 - dist2) * (1/dist2)**1.5

    dist2 = agent.maze_distance(pos2, agent.start)
    dist1 = agent.maze_distance(pos1, agent.start)
    home_dist_d = (dist1 - dist2) 

    moved = agent.maze_distance(pos1, pos2)
    stationary = (1 if moved == 0 else 0)
    died = (1 if agent_state_2.configuration.pos == agent.start and moved > 1 else 0)
    for opp in opponents:
        if died:
            break
        if s1.data.agent_states[opp].configuration is None:
            continue
        if agent.maze_distance(pos1, s1.data.agent_states[opp].configuration.pos) > 2:
            continue
        if s2.agent_distances[opp] - s1.agent_distances[opp] > 20:
            killed += 1

    collected = agent_state_2.num_carrying - agent_state_1.num_carrying
    collected = max(0, collected)
    returned = agent_state_2.num_returned - agent_state_1.num_returned
    
    features = [killed, died, collected, returned, food_dist_d, stationary, home_dist_d * agent_state_1.num_carrying]
    weights = [10, -50, 2, 20, 1, -0.01, 0.05]
    return np.dot(features, weights)

def state_to_feature(agent, game_state):
    distance_to_food = []
    distance_to_home = []
    if not agent.red:
        action_dirs = [[0, -1], [1, 0], [0, 1], [-1, 0]]
    else:
        action_dirs = [[0, 1], [-1, 0], [0, -1], [1, 0]]
    pos = agent.intify(game_state.data.agent_states[agent.index].configuration.pos)
    distance_to_food_curr = agent.closest_food(game_state, pos)[0]
    distance_to_home_curr = agent.maze_distance(pos, agent.start)

    for action in action_dirs:
        npos = (pos[0]+action[0], pos[1]+action[1])
        if game_state.data.layout.walls.data[npos[0]][npos[1]]:
            distance_to_food.append(-1)
            distance_to_home.append(-1)
            continue
        distance_to_food.append((agent.closest_food(game_state, npos)[0] - distance_to_food_curr))
        distance_to_home.append((agent.maze_distance(npos, agent.start) - distance_to_home_curr))
    opps = agent.get_opponents(game_state)
    dist_to_opps = []
    for opp in opps:
        if game_state.data.agent_states[opp].configuration is not None:
            opp_pos = game_state.data.agent_states[opp].configuration.pos
            dist_to_opps.append(1/(agent.maze_distance(pos, opp_pos)+1))
        else:
            dist_to_opps.append(1/(game_state.agent_distances[opp]+1))
    
    extra_features = []
    extra_features.append(game_state.data.agent_states[agent.index].num_carrying)
    extra_features.append(1/(game_state.data.agent_states[agent.index].scared_timer+1))
    extra_features.append(int(game_state.data.agent_states[agent.index].is_pacman))
    extra_features.append(1/(distance_to_food_curr+1))
    extra_features.append(1/(distance_to_home_curr+1))
    extra_features.append((1/(game_state.data.timeleft+1)))
    enemy_scared = max([game_state.data.agent_states[opp].scared_timer for opp in opps])
    extra_features.append(1/(enemy_scared+1))
    return np.array(distance_to_food + distance_to_home + dist_to_opps + extra_features)

def state_to_pic(agent, game_state):
    view_around = (5,5)
    pos = agent.intify(game_state.data.agent_states[agent.index].configuration.pos)
    pic = np.zeros((2, 2*view_around[0]+1, 2*view_around[1]+1))
    walls = game_state.data.layout.walls.data
    foods = agent.get_food(game_state)
    for di in range(-view_around[0], view_around[0]+1):
        for dj in range(-view_around[1], view_around[1]+1):
            npos = (pos[0]+di, pos[1]+dj)
            if npos[0] < 0 or npos[0] >= len(walls) or npos[1] < 0 or npos[1] >= len(walls[0]):
                continue
            pic[0,view_around[0]+di, view_around[1]+dj] = int(walls[npos[0]][npos[1]])
            pic[1,view_around[0]+di, view_around[1]+dj] = int(foods[npos[0]][npos[1]])
    opponents = agent.get_opponents(game_state)
    for opp in opponents:
        agent_s =  game_state.data.agent_states[opp]
        if agent_s.configuration is not None:
            epos = agent.intify(agent_s.configuration.pos)
            if abs(epos[0]-pos[0]) > view_around[0]:
                continue
            if abs(epos[1]-pos[1]) > view_around[1]:
                continue
            pic[1, view_around[0] + epos[0] - pos[0], view_around[1] + epos[1] - pos[1]] = -1
    if agent.red:
        pic[0,:,:] = np.rot90(np.rot90(pic[0,:,:]))
        pic[1,:,:] = np.rot90(np.rot90(pic[1,:,:]))
    return pic

def get_transition(agent):
    if len(agent.observation_history) < 2:
        return None
    s1 = agent.observation_history[-2]
    s2 = agent.observation_history[-1]
    action = get_action_index(s2.data.agent_states[agent.index].configuration.direction)
    reward = get_attacker_reward(agent, s1, s2)
    # print(reward)
    available_actions = [get_action_index(name) for name in s2.get_legal_actions(agent.index)]
    return (state_to_pic(agent, s1), state_to_feature(agent, s1), action, reward, 
            state_to_pic(agent, s2), state_to_feature(agent, s2), available_actions)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv2d = nn.Conv2d(2, 4, (3, 3))
        self.conv2d2 = nn.Conv2d(4, 8, (3, 3))

        self.layer1 = nn.Linear(89, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 5)

    def forward(self, x, f):
        x = F.relu(self.conv2d(x))
        x = F.relu(self.conv2d2(x))
        x = F.max_pool2d(x, (2,2))
        x = torch.flatten(x, x.dim()-3)
        x = torch.cat((x, f), x.dim()-1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
BATCH_SIZE = 128
GAMMA = 0.95
TAU = 0.005
LR = 0.0005
EPSILON = 0.8


def optimize_model(agent):
    if len(agent.memory) < BATCH_SIZE:
        return
    transitions = agent.memory.sample(BATCH_SIZE)
    state_batch = torch.tensor(np.array([state for (state, _, _, _, _, _, _) in transitions]), dtype=torch.float32)
    f1 = torch.tensor(np.array([f1 for (_, f1, _, _, _, _, _) in transitions]), dtype=torch.float32)
    action_batch = torch.tensor([[action] for (_, _, action, _, _, _, _) in transitions])
    reward_batch =  torch.tensor([reward for (_, _, _, reward, _, _, _) in transitions])
    next_state_batch = torch.tensor(np.array([s2 for (_, _, _, _, s2, _, _) in transitions]), dtype=torch.float32)
    f2 = torch.tensor(np.array([f2 for (_, _, _, _, _, f2, _) in transitions]), dtype=torch.float32)
    acs = [a for (_, _, _, _, _, _, a) in transitions]


    state_qvalues = agent.policy_net(state_batch, f1)
    state_action_values = state_qvalues.squeeze(1).gather(1, action_batch).squeeze(1)

    with torch.no_grad():
        next_qvalues = agent.target_net(next_state_batch, f2).squeeze(1).data
        next_state_values = []
        for i, a_index in enumerate(acs):
            next_state_values.append(max([next_qvalues[i, a] for a in a_index]))
        next_state_values = torch.tensor(np.array(next_state_values))
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()

    loss = criterion(state_action_values, expected_state_action_values)

    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(agent.policy_net.parameters(), 100)
    agent.optimizer.step()

    target_net_state_dict = agent.target_net.state_dict()
    policy_net_state_dict = agent.policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    agent.target_net.load_state_dict(target_net_state_dict)

    
ATTACK_MEMORY_PATH = os.path.join(cd, "attack_memory.pkl")
MODEL_PATH =  os.path.join(cd, "attacker_model")

# ATTACK_MEMORY_PATH = "attack_memory.pkl"
# MODEL_PATH = "attacker_model"


class Memory():
    def __init__(self, capacity, file=None):
        if file is not None:
            try:
                with open(file, "rb") as f:
                    d = pickle.load(f)
                    self.memory = d
            except Exception as e:
                self.memory = deque([], maxlen=capacity)
                print(e)
        else:
            self.memory = deque([], maxlen=capacity)

    def push(self, el):
        self.memory.append(el)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def to_list(self):
        return list(self.memory)



#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.go_home = False
        self.last_seen = None
        self.time = 0 
        if TRAINING:
            self.memory = Memory(10**5//2, ATTACK_MEMORY_PATH)
        self.target_net = DQN()
        try:
            self.target_net.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=torch.device('cpu')))
        except Exception as e:
            if not TRAINING:
                raise(Exception(e))
            print("Cannot find target net")
        if TRAINING:
            self.policy_net = DQN()
            try:
                self.policy_net.load_state_dict(torch.load(MODEL_PATH+"_policy", weights_only=True, map_location=torch.device('cpu')))
            except Exception as e:
                self.policy_net.load_state_dict(self.target_net.state_dict())
                print("Cannot find policy net")
            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        shape = (len(game_state.data.layout.walls.data),  len(game_state.data.layout.walls.data[0]))
        self.last_seen = np.full(shape, 0)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if game_state.data.layout.walls.data[i][j]:
                    self.last_seen[i,j] = 10**9
        pos = game_state.data.agent_states[self.index].configuration.pos
        self.update_seen(pos)
        CaptureAgent.register_initial_state(self, game_state)

    def update_seen(self, pos, d=5):
        for i in range(-d, d+1):
            for j in range(-d, d+1):
                if abs(i)+abs(j) > d:
                    continue
                if pos[0]+i < 0 or pos[0]+i >= self.last_seen.shape[0]:
                    continue
                if pos[1]+j < 0 or pos[1]+j >= self.last_seen.shape[1]:
                    continue
                self.last_seen[pos[0]+i,pos[1]+j] = max(self.time,  self.last_seen[pos[0]+i,pos[1]+j])

    def get_successor(self, game_state, action, index=None):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        if index is None:
            index = self.index
        successor = game_state.generate_successor(index, action)
        pos = successor.get_agent_state(index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def calculate_move(self, game_state, pos1, pos2):
        dirs = [([1,0], 'East'), ([-1, 0], 'West'), ([0, -1], 'South'), ([0, 1], 'North')]
        d = self.maze_distance(pos1, pos2)
        for (dir, name) in dirs:
            npos = (pos1[0] + dir[0], pos1[1] + dir[1])
            if game_state.data.layout.walls.data[npos[0]][npos[1]]:
                continue
            if self.maze_distance(npos, pos2) == d-1:
                return name
        return None
    
    def max_agent(self, game_state, players, id, depth, heur, prune, alpha=-10**10, beta=10**10):
        if depth == 6:
            return (heur(game_state, players), None)
        actions = game_state.get_legal_actions(players[id])
        actions = [action for action in actions if action != "Stop"]
        max_value = (-10**10, None)
        for action in actions:
            successor = self.get_successor(game_state, action, players[id])
            if prune(successor):
                continue
            val = self.min_agent(successor, players, id+1, depth+1, heur, prune, alpha, beta)
            max_value = max(max_value, (val, action))
            alpha = max(alpha, max_value[0])
            if max_value[0] > beta:
                return max_value
        return max_value

    def min_agent(self, game_state, players, id, depth, heur, prune, alpha=-10**10, beta=10**10):
        if depth == 6:
            return heur(game_state, players)
        actions = game_state.get_legal_actions(players[id])
        actions = [action for action in actions if action != "Stop"]
        min_val = 10**10
        for action in actions:
            successor = self.get_successor(game_state, action, players[id])
            if id == len(players)-1:
                val = self.max_agent(successor, players, 0, depth+1, heur, prune, alpha, beta)[0]
            else:
                val = self.min_agent(successor, players, id+1, depth+1, heur, prune, alpha, beta)
            min_val = min(min_val, val)
            beta = min(beta, min_val)
            if min_val < alpha:
                return min_val
        return min_val

    def intify(self, pos):
            return (int(pos[0]), int(pos[1]))

    def maze_distance(self, pos1, pos2):
        return self.get_maze_distance(self.intify(pos1), self.intify(pos2))

class OffensiveReflexAgent(ReflexCaptureAgent):
    def closest_food(self, game_state, pos):
        foods = self.get_food(game_state)
        closest_food = (10**9, None)
        for i, row in enumerate(foods):
            for j, el in enumerate(row):
                if not el:
                    continue
                closest_food = min(closest_food, (self.get_maze_distance(pos, (i,j)), (i,j)))
        return closest_food

    def aggressive_heur(self, game_state, players):
        pos = game_state.data.agent_states[self.index].configuration.pos
        dist = self.maze_distance(pos, self.start)
        if dist < 10:
            return -10**9
        if not game_state.data.agent_states[self.index].is_pacman:
            return 5 - (1/dist if dist > 0 else 0)
        return 1/dist
        # closest_food = self.closest_food(game_state, pos)
        # return 1/closest_food[0] + game_state.data.agent_states[self.index].num_carrying

    def should_go_home(self, game_state, pos, opps, closest_food):
        carrying = game_state.data.agent_states[self.index].num_carrying
        distances = [game_state.agent_distances[i] for i in self.get_opponents(game_state)]
        if min(distances) <= 6 and closest_food[0] > 2 and carrying > 0:
            return True
        if carrying >= 6:
            return True
        return False
    
    def prune(self, game_state):
        return False

    def avoid_heur(self, game_state, players):
        distances = [game_state.agent_distances[i] for i in self.get_opponents(game_state)]
        return min(distances)

    def choose_action(self, game_state):
        if TRAINING:
            if game_state.data.timeleft <= 10:
                with open(ATTACK_MEMORY_PATH, 'wb') as f:
                    pickle.dump(self.memory.memory, f)
                torch.save(self.target_net.state_dict(), MODEL_PATH)
                torch.save(self.policy_net.state_dict(), MODEL_PATH+"_policy")
        pos = game_state.data.agent_states[self.index].configuration.pos
        distance_home = self.maze_distance(pos, self.start)
        distance_food = self.closest_food(game_state, pos)[0]
        if distance_home/(distance_home+distance_food) > NETWORK_DISTANCE:
            if TRAINING:
                if len(self.observation_history) > 1:
                    (s1, f1, action, reward, s2, f2, acs) = get_transition(self)
                    self.memory.push((s1, f1, action, reward, s2, f2, acs))
                    optimize_model(self)
            with torch.no_grad():
                input = state_to_pic(self, game_state)
                f = state_to_feature(self, game_state)
                res = self.target_net(torch.tensor(input, dtype=torch.float32), torch.tensor(f, dtype=torch.float32))
                actions = game_state.get_legal_actions(self.index)[:-1]
                if self.red:
                    actions = [invert_action(action) for action in actions]
                ids = [get_action_index(action) for action in actions]
                values = [res[id] for id in ids]
                if random.random() < EPSILON:
                    move = ids[np.argmax(values)]
                else:
                    move = np.random.choice(ids)
                action = get_action_name(move)
                if self.red:
                    action = invert_action(action)
                return action
        else:
            return self.choose_action_minmax(game_state)
        

    def choose_action_minmax(self, game_state):
        if TRAINING:
            if game_state.data.timeleft <= 10:
                with open(ATTACK_MEMORY_PATH, 'wb') as f:
                    pickle.dump(self.memory.memory, f)
            if len(self.observation_history) > 1:
                (s1, f1, action, reward, s2, f2, acs) = get_transition(self)
                self.memory.push((s1, f1, action, reward, s2, f2, acs))
        pos = game_state.data.agent_states[self.index].configuration.pos
        pos = (int(pos[0]), int(pos[1]))
        opps = [opp for opp in self.get_opponents(game_state) if game_state.data.agent_states[opp].configuration is not None]
        opps = [opp for opp in opps if self.maze_distance(pos, game_state.data.agent_states[opp].configuration.pos) < 6]
        closest_food = self.closest_food(game_state, pos)
        if game_state.data.agent_states[self.index].num_carrying > 3:
            if len(opps) > 0:
                res = self.max_agent(game_state, [self.index] + opps, 0, 0, self.avoid_heur, self.prune)
                return res[1]
            move = self.calculate_move(game_state, pos, self.start)
            if move is None:
                move = 'Stop'
            return move
        else:
            if len(opps) > 0:
                res = self.max_agent(game_state, [self.index] + opps, 0, 0, self.aggressive_heur, self.prune)
                return res[1]
            
            move = self.calculate_move(game_state, pos, closest_food[1])
            if move is None:
                move = 'Stop'
            return move
        return 'Stop'


class DefensiveReflexAgent(ReflexCaptureAgent):
    def get_interesting_positions(self, shape):
        if not self.red:
            return ((shape[0]//2, 2*shape[0]//3), (0, shape[1]))
        else:
            return ((shape[0]//3, shape[0]//2), (0, shape[1]))
    
    def get_distance_to_agent(self, game_state, agent):
        pos = game_state.data.agent_states[self.index].configuration.pos
        enemy_pos = game_state.data.agent_states[agent].configuration.pos
        return self.maze_distance(pos, enemy_pos)

    def recently_died(self, game_state, player):
        start = game_state.data.agent_states[player].start.pos
        pos = game_state.data.agent_states[player].configuration.pos
        return self.maze_distance(start, pos) < 5

    def prune(self, game_state):
        return game_state.data.agent_states[self.index].is_pacman

    def heur(self, game_state, players):
        opps = self.get_opponents(game_state)
        carrying = sum(game_state.data.agent_states[opp].num_carrying for opp in opps)
        distance = sum(self.get_distance_to_agent(game_state, player) for player in players[1:])
        killed = sum(1 for player in players[1:] if self.recently_died(game_state, player))
        died = int(self.recently_died(game_state, self.index))
        over = int(game_state.data.agent_states[self.index].is_pacman)
        return died * -100 + over * -10 + killed * 100 + (1/distance)

    def find_missing_foods(self, last_foods, foods):
        missing = []
        for i, row in enumerate(last_foods):
            for j, el in enumerate(row):
                if el != foods[i][j]:
                    missing.append((i,j))
        return missing

    def choose_action(self, game_state):
        if len(self.observation_history) > 1:
            foods = self.get_food_you_are_defending(game_state)
            last_foods = self.get_food_you_are_defending(game_state)
            missing = self.find_missing_foods(last_foods, foods)
            for c in missing:
                self.last_seen[c[0],c[1]] = - 100 - self.time
        self.time += 1
        pos = game_state.data.agent_states[self.index].configuration.pos
        pos = self.intify(pos)
        self.update_seen(pos)
        interesting = self.get_interesting_positions(self.last_seen.shape)
        area = self.last_seen[interesting[0][0]:interesting[0][1],interesting[1][0]:interesting[1][1]]
        ind = np.unravel_index(np.argmin(area, axis=None), area.shape)
        opps = [opp for opp in self.get_opponents(game_state) if game_state.data.agent_states[opp].configuration is not None]
        opps = [opp for opp in opps if self.maze_distance(pos, game_state.data.agent_states[opp].configuration.pos) < 6]
        opps = [opp for opp in opps if game_state.data.agent_states[opp].is_pacman]
        if len(opps) > 0:
            move =  self.max_agent(game_state, [self.index]+opps, 0, 0, self.heur, self.prune)[1]
            if move is None:
                move = 'Stop'
            return move
        else:
            target = (ind[0] + interesting[0][0], ind[1])
            move = self.calculate_move(game_state, pos, target)
            if move is None:
                move = 'Stop'
            return move
