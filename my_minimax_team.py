import os
import sys

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


def create_team(first_index, second_index, is_red,
                first='HibridAgent', second='DefensiveAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


class MinimaxAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.go_home = False
        self.last_seen = None
        self.time = 0 

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
    
    def recently_died(self, game_state, player):
        start = game_state.data.agent_states[player].start.pos
        pos = game_state.data.agent_states[player].configuration.pos
        return self.maze_distance(start, pos) < 5

    def intify(self, pos):
            return (int(pos[0]), int(pos[1]))

    def maze_distance(self, pos1, pos2):
        return self.get_maze_distance(self.intify(pos1), self.intify(pos2))
    
class HibridAgent(MinimaxAgent):
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing) 

        self.offense = OffensiveAgent(index, time_for_computing)
        self.defense = DefensiveAgent(index, time_for_computing)

    def winning(self, game_state):
        if self.red:
            return game_state.data.score > 0
        else:
            return game_state.data.score < 0

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.offense.register_initial_state(game_state)
        self.defense.register_initial_state(game_state)

    def choose_action(self, game_state):
        offense_move = self.offense.choose_action(game_state)
        defense_move = self.defense.choose_action(game_state)
        if self.winning(game_state):
            return defense_move
        else:
            return offense_move
    

class OffensiveAgent(MinimaxAgent):
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
        opps = self.get_opponents(game_state)
        pos = game_state.data.agent_states[self.index].configuration.pos
        carrying = game_state.data.agent_states[self.index].num_carrying
        returned = game_state.data.agent_states[self.index].num_returned
        food_distance = self.closest_food(game_state, pos)[0]
        killed = sum(1 for player in players[1:] if self.recently_died(game_state, player))
        died = int(self.recently_died(game_state, self.index))
        enemy_scared = sum([game_state.data.agent_states[opp].scared_timer for opp in opps])
        return 1/(food_distance+1) + carrying + 10 * returned + 100 * killed - 100 * died + int(enemy_scared > 0)

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
        opps = self.get_opponents(game_state)
        pos = game_state.data.agent_states[self.index].configuration.pos
        carrying = game_state.data.agent_states[self.index].num_carrying
        returned = game_state.data.agent_states[self.index].num_returned
        home_distance = self.maze_distance(pos, self.start)
        killed = sum(1 for player in players[1:] if self.recently_died(game_state, player))
        died = int(self.recently_died(game_state, self.index))
        enemy_scared = sum([game_state.data.agent_states[opp].scared_timer for opp in opps])
        return 1/(home_distance+1) + carrying + 10 * returned + 100 * killed - 100 * died + + int(enemy_scared > 0)
    
    def choose_action(self, game_state):
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


class DefensiveAgent(MinimaxAgent):
    def get_interesting_positions(self, shape):
        if not self.red:
            return ((shape[0]//2, 3*shape[0]//4), (0, shape[1]))
        else:
            return ((shape[0]//4, shape[0]//2), (0, shape[1]))
    
    def get_distance_to_agent(self, game_state, agent):
        pos = game_state.data.agent_states[self.index].configuration.pos
        enemy_pos = game_state.data.agent_states[agent].configuration.pos
        return self.maze_distance(pos, enemy_pos)

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
        opps = [opp for opp in self.get_opponents(game_state) if game_state.data.agent_states[opp].configuration is not None]
        opps = [opp for opp in opps if game_state.data.agent_states[opp].is_pacman]
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
