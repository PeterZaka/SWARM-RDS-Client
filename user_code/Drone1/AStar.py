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

import heapq
import copy

from SWARMRDS.utilities.algorithm_utils import Algorithm
from SWARMRDS.utilities.data_classes import Trajectory, MovementCommand, PosVec3
from SWARMRDS.utilities.log_utils import UserLogger


def transform(data):
    # Access the 'point_cloud' key and convert it to a numpy array
    arr = np.array(data["point_cloud"])

    # Access individual values from the 'pose' key
    pose_data = data["pose"]
    q0 = pose_data["orientation"]["w_val"]
    q1 = pose_data["orientation"]["x_val"]
    q2 = pose_data["orientation"]["y_val"]
    q3 = pose_data["orientation"]["z_val"]

    x_origin = pose_data["position"]["x_val"]
    y_origin = pose_data["position"]["y_val"]
    z_origin = pose_data["position"]["z_val"]
        
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
        
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
        
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
    transposition = np.array([x_origin, y_origin, z_origin])

        
    rot_matrix = np.array([[r00, r01, r02],
                        [r10, r11, r12],
                        [r20, r21, r22]])
        
    rotated = np.zeros((len(arr), 3))

    for i in range(len(arr)):
    
        point = np.matmul(rot_matrix, arr[i])
        point += transposition
        rotated[i] = point

    rotated[:, 2] *= -1

    data['point_cloud'] = rotated
    return data


def point_cloud_to_occupancy_map(points, grid_size, resolution):
    # pc to bin arr
    binary_grid = np.zeros(grid_size, dtype=float)
    for point in points:
      x, y = point
      grid_x = int(grid_size[0] / 2 + x / resolution)
      grid_y = int(grid_size[1] / 2 + y / resolution)
      if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1]:
        binary_grid[grid_x, grid_y] = 1

    binary_grid = binary_grid.T

    return binary_grid



class AStar(Algorithm):
    """
    A planner that simply passes through the given commands to the
    velocity controller.

    Modified from: https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/AStar/a_star.py

    ## Inputs:
    - resolution [float] The resolution of the occupancy map
    - goal_point [list] Where the agent should head [x,y]
    - map_size [list] The size of the map from the start point to one
                      edge. This value is the offset for the map
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

        self.total_points = np.empty((0, 2))
        self.grid = np.zeros((200, 200), dtype=float)
        self.path = []
        self.drone_path = []

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
            if key == "SWARMPointCloud":
                self.swarm_point_cloud = item
                if type(self.swarm_point_cloud.metadata.position).__name__ != "PosVec3":
                    self.log.log_message(f"Position type: {type(self.swarm_point_cloud.metadata.position)}")
                    self.log.log_message(f"{self.swarm_point_cloud.metadata.position}")
                    self.swarm_point_cloud = None
                    continue
                if type(self.swarm_point_cloud.metadata.orientation).__name__ != "Quaternion":
                    self.log.log_message(f"Orientation type: {type(self.swarm_point_cloud.metadata.orientation)}")
                    self.log.log_message(f"{self.swarm_point_cloud.metadata.orientation}")
                    self.swarm_point_cloud = None
                    continue

            if key == "OccupancyMap":
                self.obstacle_map = item

            if key == "AgentState":
                self.log.log_message(f"agentstate: {item}")
                self.position = item.position

        if type(self.obstacle_map).__name__ == "NoneType":
            return None

        if self.swarm_point_cloud == None:
            return None
        else:
            self.point_cloud = {
                "point_cloud" : np.reshape(self.swarm_point_cloud.points, (-1, 3)),
                "pose" : {
                    "orientation" : {
                        "w_val" : self.swarm_point_cloud.metadata.orientation.w,
                        "x_val" : self.swarm_point_cloud.metadata.orientation.x,
                        "y_val" : self.swarm_point_cloud.metadata.orientation.y,
                        "z_val" : self.swarm_point_cloud.metadata.orientation.z
                    },
                    "position" : {
                        "x_val" : self.swarm_point_cloud.metadata.position.X,
                        "y_val" : self.swarm_point_cloud.metadata.position.Y,
                        "z_val" : self.swarm_point_cloud.metadata.position.Z
                    }
                }
            }

            self.position.X = self.swarm_point_cloud.metadata.position.X
            self.position.Y = self.swarm_point_cloud.metadata.position.Y

            points = self.point_cloud['point_cloud']
            if len(points) != 0:
                self.point_cloud = transform(self.point_cloud)
                points = self.point_cloud['point_cloud']

                z_range = (0, 3)
                (z_min, z_max) = z_range
                mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
                points = points[mask, :2]

                self.total_points = np.concatenate((self.total_points, points))

                self.grid = point_cloud_to_occupancy_map(self.total_points, (200, 200), 1)


        if not self.executing_trajectory:
            # Plan for the trajectory
            # We start at X=0 and Y=0 in NED coordiantes, but that is map_size[0], map_size[1] in the map
            # We also must make sure that we offset our goal point as well

            self.executing_trajectory = True

            self.log.log_message("Requesting a Trajectory from the planner")                

            # Have to change map_size here for some reason
            self.map_size = (int(len(self.obstacle_map[0])), int(len(self.obstacle_map)))

            if len(self.path) == 0:
                self.path = self.myalgo(self.calc_real_to_array((self.position.X, self.position.Y)),
                                    self.calc_real_to_array((self.goal_point.X, self.goal_point.Y)),
                                    self.grid)
            else:
                self.path = self.fix_path(self.path, self.grid, [10, 5, 2.5, 1.1])

                self.log.log_message(f"Path: {self.path}")

                blocking = self.is_path_blocked(self.path, self.grid, 1)
                if blocking != -1:
                    self.log.log_message(f'Was blocked')
                    self.path = self.myalgo(self.calc_real_to_array((self.position.X, self.position.Y)),
                                        self.calc_real_to_array((self.goal_point.X, self.goal_point.Y)),
                                        self.grid)
                    blocking = self.is_path_blocked(self.path, self.grid, 1)
                    if blocking != -1:
                        self.log.log_message(f'Extremely blocked')
                        xsign = math.copysign(1, self.path[1][0])
                        ysign = math.copysign(1, self.path[1][1])
                        self.path[1] = (self.path[0][0] - xsign, self.path[0][1] - ysign)

            if len(self.path) == 0:
                self.log.log_message("No path found")
                self.executing_trajectory = False
                return Trajectory()

            # Print map with drone_path
            self.drone_path.append(self.calc_real_to_array((self.position.X, self.position.Y)))
            grid = copy.deepcopy(self.grid)
            for tup in self.drone_path:
                grid[tup[1]][tup[0]] = 2
            for r in range(len(grid)):
                string = ''
                for c in range(len(grid[0])):
                    string += str(grid[r][c])
                self.log.log_message(string)

            path = self.path[1:]
            trajectory = self.array_to_trajectory(path)

            self.executing_trajectory = False
            return trajectory
        else:
            time.sleep(0.1)
            # SWARM will ignore these values
            return None



    def myalgo(self, start, end, graph):
        """
        Returns the path traversing graph using a* search

        Parameters
        ----------
        start : tuple or list
            (x, y)
        end : tuple or list
            (x, y)
        graph : 2d list
            graph to traverse

        Returns
        -------
        list of (x, y)
            path from start to end if found
        """

        cost_grid = self.create_cost_grid(graph, [10, 5, 2.5, 1])
        points = self.calculate_path(start, end, graph, cost_grid)

        if len(points) == 0:
            return []

        points = self.prune_path(points)
        points = self.line_of_sight_path(points, cost_grid, 5)

        return points

    class Node:

        def __init__(self, pos, parent=None):
            self.pos = tuple(pos)
            self.parent = parent
            self.g = 0 # cost of path from n to start
            self.h = 0 # estimated cost of path from n to end
            self.f = 0 # total cost = g + h

        def __eq__(self, other):
            return self.pos[0] == other.pos[0] and self.pos[1] == other.pos[1]
        
        def __hash__(self):
            return hash(self.pos)

    
    def create_cost_grid(self, grid, costs):
        new_grid = copy.deepcopy(grid)

        check_range = len(costs)
        costs.insert(0, -1)
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 1:
                    if c == 0 or grid[r][c - 1] == 1:
                        left_check = 0
                    else:
                        left_check = -check_range

                    if c == len(grid[0]) - 1 or grid[r][c + 1] == 1:
                        right_check = 0
                    else:
                        right_check = check_range

                    if r == len(grid) - 1 or grid[r + 1][c] == 1:
                        up_check = 0
                    else:
                        up_check = check_range

                    if r == 0 or grid[r - 1][c] == 1:
                        down_check = 0
                    else:
                        down_check = -check_range

                    for dx in range(left_check, right_check + 1):
                        for dy in range(down_check, up_check + 1):
                            x, y = c + dx, r + dy
                            if 0 <= x < len(grid[0]) and 0 <= y < len(grid) and grid[y][x] != 1:
                                new_grid[y][x] = max(new_grid[y][x], costs[max(abs(dx), abs(dy))])

        return new_grid


    def calculate_path(self, start, end, graph, cost_grid=[]):
        """
        Returns the path traversing graph using a* search

        Parameters
        ----------
        start : tuple or list
            (x, y)
        end : tuple or list
            (x, y)
        graph : 2d list
            graph to traverse
        cost_grid: 2d list
            adds a penalty

        Returns
        -------
        list of (x, y)
            path from start to end if found
        """

        open_list = []
        closed_list = set()

        start_node = self.Node(start)
        index = 0
        open_list.append((start_node.f, index, start_node))
        heapq.heapify(open_list)

        while len(open_list) > 0:
            start_node = heapq.heappop(open_list)[-1]
            closed_list.add(start_node)

            for dir in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]:
                x = dir[0]
                y = dir[1]

                node = self.Node((start_node.pos[0] + x, start_node.pos[1] + y), start_node)
                if node.pos[0] < 0:
                    continue
                if node.pos[0] >= len(graph[0]):
                    continue
                if node.pos[1] < 0:
                    continue
                if node.pos[1] >= len(graph):
                    continue

                if node in closed_list:
                    continue

                if graph[node.pos[1]][node.pos[0]] == 1:
                    continue

                node.g = start_node.g + math.sqrt(math.pow(start_node.pos[0] - node.pos[0], 2) + math.pow(start_node.pos[1] - node.pos[1], 2))
                dup = any(node == n[-1] and n[-1].g <= node.g for n in open_list)
                if dup:
                    continue

                node.h = math.sqrt(math.pow(end[0] - node.pos[0], 2) + math.pow(end[1] - node.pos[1], 2))
                node.f = node.g + node.h
                if len(cost_grid) != 0:
                    node.f += cost_grid[node.pos[1]][node.pos[0]]


                if node.pos[0] == end[0] and node.pos[1] == end[1]:
                    path = []
                    while node is not None:
                        path.append(node.pos)
                        node = node.parent

                    path = path[::-1]
                    return path

                index += 1
                heapq.heappush(open_list, (node.f, index, node))

        return []
    
    def prune_path(self, path):
        new_path = [path[0]]
        for i in range(1, len(path) - 1):
            dx = path[i][0] - path[i+1][0]
            dx2 = path[i-1][0] - path[i][0]

            dy = path[i][1] - path[i+1][1]
            dy2 = path[i-1][1] - path[i][1]

            if dx == dx2 and dy == dy2:
                continue
            new_path.append(path[i])
        new_path.append(path[-1])
        return new_path
    
    def is_line_of_sight_blocked(self, start, end, grid, threshold):
        x1, y1 = start
        x2, y2 = end

        dy = y2 - y1
        dx = x2 - x1

        length = math.sqrt(dy**2 + dx**2)
        vx = dx / length
        vy = dy / length

        for t in range(0, math.floor(length) + 2):
            x = x1 + t * vx
            y = y1 + t * vy

            floor_cell = grid[math.floor(y), math.floor(x)]
            ceil_cell = grid[math.ceil(y), math.ceil(x)]

            if floor_cell >= threshold or floor_cell == 1 or ceil_cell >= threshold or ceil_cell == 1:
                return True
        
        return False
    
    def line_of_sight_path(self, path, grid, threshold):
        new_path = []

        i = 0
        while i < len(path):
            start = path[i]
            new_path.append(start)
            for j in range(len(path) - 1, i, -1):
                end = path[j]
                if not self.is_line_of_sight_blocked(start, end, grid, threshold):
                    new_path.append(end)
                    i = j
                    break

            i += 1

        return new_path

    def is_path_blocked(self, path, grid, threshold):
        for i in range(len(path) - 1):
            if self.is_line_of_sight_blocked(path[i], path[i+1], grid, threshold):
                return i
        return -1

    def reroute_path(self, path, intersection, graph, cost_graph):
        unblocking_path = self.calculate_path(path[intersection], path[intersection+1], graph, cost_graph)
        unblocking_path = unblocking_path[1:-1]
        intersection += 1
        path = path[:intersection] + unblocking_path + path[intersection:]
        return path

    def fix_path(self, path, grid, costs):
        # path[0] = pos
        path[0] = self.calc_real_to_array((self.position.X, self.position.Y))
        for j in range(len(path) - 1, 0, -1):
            end = path[j]
            if not self.is_line_of_sight_blocked(path[0], end, grid, 100):
                path = path[0:1] + path[j:]
                break

        blocking = self.is_path_blocked(path, grid, 1)
        if blocking != -1:
            if len(path) > 2:
                if blocking == 0:
                    path = path[0:1] + path[2:]
                else:
                    path = path[:blocking] + path[(blocking + 1):]
                    blocking -= 1
            data_cost = self.create_cost_grid(grid, costs)
            path = self.reroute_path(path, blocking, grid, data_cost)
            path = self.prune_path(path)
            path = self.line_of_sight_path(path, data_cost, 5)

        return path



    # Simulation helper functions

    def array_to_trajectory(self, points) -> Trajectory:
        """
        Convert list of points [grid] to a trajectory
        input:
            List of points: [(x, y), ...] [grid]
        output:
            trajectory
        """
        trajectory = Trajectory()
        position = PosVec3()

        for point in points:
            position = PosVec3()

            position.X, position.Y = self.calc_array_to_real((point[0], point[1]))
            position.Z = self.flight_altitude
            command = MovementCommand(position=position, speed=self.agent_speed)
            trajectory.points.append(command)

        # calculate the headings for each point
        for i, point in enumerate(trajectory.points):
            if i == 0:
                heading = np.degrees(np.arctan2(point.position.Y - self.position.Y, point.position.X - self.position.X))
            else:
                heading = np.degrees(np.arctan2(point.position.Y - trajectory.points[i - 1].position.Y, point.position.X -  trajectory.points[i - 1].position.X))

            trajectory.points[i].heading = heading

        return trajectory

    def calc_array_to_real(self, position: tuple):
        """
        Go from numpy array to real world
        input:
            position: (x, y) [grid]
        output:
            real_position: (x, y) [m]
        """
        return (int((position[0] - self.map_size[0] / 2) * self.resolution),
                int((position[1] - self.map_size[1] / 2) * self.resolution))

    def calc_real_to_array(self, position: tuple):
        """""
        Go from real world to numpy array
        input:
            position: (x, y) [m]
        output:
            array_position: (x, y) [grid]
        """
        return (int(self.map_size[0] / 2 + position[0] / self.resolution),
                int(self.map_size[1] / 2 + position[1] / self.resolution))

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