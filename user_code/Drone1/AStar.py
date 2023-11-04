# =============================================================================
# Copyright 2022-2023. Codex Laboratories LLC. All rights reserved.
#
# Created By: Tyler Fedrizzi
# Created On: 9 June 2022
#
#
# Description: An implementation of A* in SWARM
# =============================================================================
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import logging

import copy

from SWARMRDS.utilities.algorithm_utils import Algorithm
from SWARMRDS.utilities.data_classes import Trajectory, MovementCommand, PosVec3
from SWARMRDS.utilities.log_utils import UserLogger


class AStar(Algorithm):
    """
    A planner that simply passes through the given commands to the
    velocity controller.

    Modified from: https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/AStar/a_star.py

    ## Inputs:
    - goal_point [list] Where the agent should head [x,y]
    - map_size [list] The size of the map from the start point to one
                      edge. This value is the offset for the map
    - resolution [float] The resolution of the occupancy map
    - agent_radisu [float] How wide the agent is
    - flight_altitude [float] What altitude the agent should fly at
    """
    def __init__(self,
                 resolution: float,
                 goal_point: list,
                 map_size: list,
                 starting_point: list,
                 agent_radius: float = 0.5,
                 flight_altitude: float = -3.0) -> None:
        super().__init__()
        self.goal_point = PosVec3(X=goal_point[0], Y=goal_point[1], Z=flight_altitude)
        self.resolution = resolution  # Meters
        self.rr = agent_radius  # Meters
        self.map_size = map_size  # Meters (X<Y)
        self.agent_speed = 3.0  # Meters / Second
        # We are in NED Coordinates so -100 is 100 meters behind the starting
        # point
        self.starting_position = starting_point  # Position you start in NED in meters
        self.motion = self.get_motion_model()
        self.flight_altitude = flight_altitude
        self.executing_trajectory = False

    def run(self, **kwargs) -> None:
        """
        Core method that is run every timestemp. Syntax for the module
        is:

        ```
        algo = Astar(goal_point)
        return_args = algo.run()
        ```

        You can extract the appropriate inputs values from Kwargs as
        well, which are determined by what you input in the 
        DefaultSimulationSettings.json file.

        In this example, the map is provided for you by the Mapping
        module and requires no work on your part to extract.
        """
        # Update the map each iteration based upon currently sensed
        # values. Plenty of flexibility here as you could stipulate
        # that one of the inputs be the next goal point that is 
        # determined by another algorithm.
        for key, item in kwargs.items():
            if key == "OccupancyMap":
                self.obstacle_map = item
                break

        if type(self.obstacle_map).__name__ == "NoneType":
            return None

        if not self.executing_trajectory:
            # Plan for the trajectory
            # We start at X=0 and Y=0 in NED coordiantes, but that is map_size[0], map_size[1] in the map
            # We also must make sure that we offset our goal point as well
            self.log.log_message("Requesting a Trajectory from the planner")                

            trajectory = self.planning(self.position.X, self.position.Y,
                                       self.goal_point.X, self.goal_point.Y)
            
            self.log.log_message("Trajectory found!")
            self.log.log_message(trajectory.displayPretty())
            self.executing_trajectory = True
            return trajectory
        else:
            time.sleep(0.1)
            # SWARM will ignore these values
            return None

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy) -> Trajectory:
        """
        A star path search
        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]
        output:
            trajectory
        """

        start_x, start_y = self.calc_real_to_array((sx, sy))
        start_node = self.Node(start_x, start_y, 0.0, -1)

        goal_x, goal_y = self.calc_real_to_array((gx, gy))
        goal_node = self.Node(goal_x, goal_y, 0.0, -1)

        open_set, closed_set = dict(), dict()
        self.log.log_message("Grid index is {}".format(self.calc_grid_index(start_node)))
        open_set[self.calc_grid_index(start_node)] = start_node
        iteration = 0
        while True:
            iteration += 1
            if len(open_set) == 0:
                print("Open set is empty..")
                break
            # print(f"Iteration {iteration}")
            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                self.log.log_message("Found goal!")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)
                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        trajectory = self.calc_final_path(goal_node, closed_set)

        return trajectory

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        trajectory = Trajectory()
        position = PosVec3()
        position.X, position.Y = self.calc_array_to_real((goal_node.x, goal_node.y))
        position.Z = self.flight_altitude
        
        command = MovementCommand(position=position, speed=self.agent_speed)
        trajectory.points.append(command)
        parent_index = goal_node.parent_index
        

        PATH = []

        while parent_index != -1:
            position = PosVec3()
            n = closed_set[parent_index]
            PATH.append([n.y, n.x])

            position.X, position.Y = self.calc_array_to_real((n.x, n.y))
            position.Z = self.flight_altitude
            command = MovementCommand(position=position, speed=self.agent_speed)
            trajectory.points.append(command)
            parent_index = n.parent_index

        # DEBUGGING
        grid = copy.deepcopy(self.obstacle_map)
        for tup in PATH:
            grid[tup[0]][tup[1]] = 2
        for r in range(len(grid)):
            string = ''
            for c in range(len(grid[0])):
                string += str(grid[r][c])
            self.log.log_message(string)

        # We go from goal to start, so reverse this so we fly start to goal
        trajectory.points.reverse()

        # calculate the headings for each point
        for i, point in enumerate(trajectory.points):
            if i == 0:
                heading = np.degrees(np.arctan2(point.position.Y - self.position.Y, point.position.X - self.position.X))
            else:
                heading = np.degrees(np.arctan2(point.position.Y - trajectory.points[i - 1].position.Y, point.position.X -  trajectory.points[i - 1].position.X))
            
            trajectory.points[i].heading = heading

        return trajectory

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_array_to_real(self, position: tuple):
        """
        Go from numpy array to real world
        input:
            position: (x, y) [grid]
        output:
            real_position: (x, y) [m]
        """
        return (int((position[0] - self.map_size[0]) * self.resolution),
                int((position[1] - self.map_size[1]) * self.resolution))

    def calc_real_to_array(self, position: tuple):
        """""
        Go from real world to numpy array
        input:
            position: (x, y) [m]
        output:
            array_position: (x, y) [grid]
        """
        return (int(self.map_size[0] + position[0] / self.resolution),
                int(self.map_size[1] + position[1] / self.resolution))

    def calc_grid_index(self, node: Node):
        return node.y * self.map_size[1] * 2 + node.x

    def verify_node(self, node: Node):
        if node.x < 0:
            return False
        elif node.x >= len(self.obstacle_map[0]):
            return False
        elif node.y < 0:
            return False
        elif node.y >= len(self.obstacle_map):
            return False

        # collision check
        # Numpy is row major so first index is the Y axis and we still need
        # to be sure to offset the points when we check the map
        if self.obstacle_map[node.y][node.x]:
            return False

        return True

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion