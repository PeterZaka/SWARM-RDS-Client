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


class MyAlgorithm(Algorithm):
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

            # Have to change map_size here for some reason
            self.map_size = (int(len(self.obstacle_map[0])), int(len(self.obstacle_map)))

            points = self.myalgo(self.calc_real_to_array((self.position.X, self.position.Y)),
                                self.calc_real_to_array((self.goal_point.X, self.goal_point.Y)),
                                self.obstacle_map)
            
            if len(points) == 0:

                # Print map
                grid = copy.deepcopy(self.obstacle_map)
                for r in range(len(grid)):
                    string = ''
                    for c in range(len(grid[0])):
                        string += str(grid[r][c])
                    self.log.log_message(string)

                self.executing_trajectory = True
                return Trajectory()

            trajectory = self.array_to_trajectory(points)

            # Print map with path
            grid = copy.deepcopy(self.obstacle_map)
            for tup in points:
                grid[tup[1]][tup[0]] = 2
            for r in range(len(grid)):
                string = ''
                for c in range(len(grid[0])):
                    string += str(grid[r][c])
                self.log.log_message(string)

            self.executing_trajectory = True
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

        points = [start, end]

        return points



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
