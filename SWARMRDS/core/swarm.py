# =============================================================================
# Copyright 2022-2023. Codex Laboratories LLC. All Rights Reserved.
#
# Created By: Tyler Fedrizzi
# Created On: 6 December 2022
#
# Description: Core class for interacting with the SWARM Simulation Framework
# =============================================================================
import json
import traceback
import datetime
import os
import glob
import ipaddress
import time
import typing
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from queue import Queue
from uuid import uuid4

from SWARMRDS.core.client import SWARMClient
from SWARMRDS.utilities.date_utils import convert_datetime_to_str
from SWARMRDS.utilities.file_utils import find_file_path, find_folder_path
from SWARMRDS.utilities.settings_utils import (
    receive_user_input,
    generate_new_user_settings_file,
)


class SWARM:
    """
    Core wrapper that contains all logic for running the SWARM platform.

    ### Arguements:
    - ip_address [str] The API address to connect if you using SWARM in
                       the cloud or remote setting.
    """

    def __init__(
        self, ip_address: str = "127.0.0.1", debug=False, response_queue: Queue = None,
        user_file_path: str = None
    ) -> None:
        self.client = SWARMClient(
            ip_address=ip_address, debug=debug, response_queue=response_queue, user_file_path=user_file_path
        )
        self.ip_address = ip_address
        self._file_path = None
        if user_file_path is not None:
            self._file_path = user_file_path
        self.generate_submission_tracking()
        self.map_name = ""
        self.debug = debug
        self._response_queue = response_queue
        self._has_trajectory = False

    def regenerate_connection(self) -> None:
        """
        Regenerate the connection since we kill it after the simulation
        is finished.
        """
        self.client = SWARMClient(
            ip_address=self.ip_address, response_queue=self._response_queue
        )
        self.client.connect()

    # =========================================================================
    #                       Helper Functions
    # =========================================================================

    def _get_supported_environments(self, working_path: str = os.getcwd()) -> dict:
        """
        Get the SupportedEnvironments.json file that exists in the
        settings folder. This describes all valid options.

        ### Inputs:
        - None

        ### Returns:
        - A dictonary containing the supported environments.
        """
        try:
            if self._file_path is not None:
                file_path = self._file_path + "/settings/SupportedEnvironments.json"
            else:
                file_path = find_file_path("SupportedEnvironments.json", "settings")
            with open(file_path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            raise AssertionError(
                "Error!\n"
                + "The file titled SupportedEnvironments.json was not found in the settings folder\n"
                + "If you do not see this file, please run example in the examples folder titled 'retrieve_environment_info.py\n"
                + "which will download this file from the server to provide the most up to date information."
            )

    def _get_supported_scenarios(self, working_path: str = os.getcwd()) -> dict:
        """
        Get the SupportedScenarios.json file that exists in the
        settings folder. This describes all valid options.

        ### Inputs:
        - None

        ### Returns:
        - A dictonary containing the supported scenarios.
        """
        try:
            if self._file_path is not None:
                file_path = self._file_path + "/settings/SupportedScenarios.json"
            else:
                file_path = find_file_path("SupportedScenarios.json", "settings")
            with open(file_path, "r") as file:
                return json.load(file)["Scenarios"]
        except FileNotFoundError:
            raise AssertionError(
                "Error!\n"
                + "The file titled SupportedScenarios.json was not found in the settings folder\n"
                + "If you do not see this file, please run example in the examples folder titled 'retrieve_environment_info.py\n"
                + "which will download this file from the server to provide the most up to date information."
            )

    def setup_simulation(
        self,
        map_name: str = None,
        settings_file_name: str = "settings/DefaultSimulationSettings.json",
    ) -> None:
        """
        Run the setup process for a Simulation, which includes choosing
        what maps to run.

        ### Inputs:
        - map_name [str] The name of the map to load the simulation
        - view_map [bool] Whether you want to view the map
        - settings_file_name [str] The name of the settings file to use
        Default is settings/DefaultSimulationSettings.json

        ### Outputs:
        - None
        """
        input_options = ["Yes", "No"]
        prompt = "Would you like to create a New Settings file?\n(Please input the number for you choice!)\n(Choose No to use settings/DefaultSimulationSettings.json)\n"
        for i, option in enumerate(input_options):
            prompt += "\t{}. {}\n".format(i + 1, option)
        prompt += "Your Choice: "
        user_input = receive_user_input(int, prompt, input_options, isList=True)
        bool_options = [option == "Yes" for option in input_options]
        if bool_options[user_input]:
            new_settings, settings_file_name = generate_new_user_settings_file()

        print("Reading map name from {}".format(settings_file_name))
        settings_map_name = self.read_map_name_from_settings(settings_file_name)
        if map_name != settings_map_name:
            self.set_environment_name(settings_file_name, map_name)

        valid = self.validate_environment_name(map_name)

        if not valid:
            print("Environment name is invalid!")
            raise AssertionError("Environment name {} is invalid!".format(map_name))

        input_options = ["Yes", "No"]
        prompt = "Would you like to view a map of the enviroment?\n(Please input the number for you choice!)\n"
        for i, option in enumerate(input_options):
            prompt += "\t{}. {}\n".format(i + 1, option)
        prompt += "Your Choice: "
        user_input = receive_user_input(int, prompt, input_options, isList=True)
        bool_options = [option == "Yes" for option in input_options]
        if bool_options[user_input]:
            # TODO Check if the map file exists and call the appropriate
            # function to get the Map from the Server.
            self.map_name = map_name
            # Run the GUI application to pull up the map
            with open(settings_file_name, "r") as file:
                settings = json.load(file)

            # We only use Trajectoris in DataCollection and other scenarios
            if settings["Scenario"]["Name"] == "DataCollection":
                self.display_map_image_with_trajectories(
                    settings["Scenario"]["Options"]["LevelNames"],
                    settings["Environment"]["Name"],
                )
            else:
                self.display_map_image(
                    settings["Scenario"]["Options"]["LevelNames"],
                    settings["Environment"]["Name"],
                )

    def read_map_name_from_settings(self, settings_file_name: str) -> str:
        """
        Read the map name given from the settings file and validate
        that this is supported before moving on.

        ### Inputs:
        - None

        ### Outputs:
        - A map name as a string
        """
        try:
            with open("{}".format(settings_file_name), "r") as file:
                settings = json.load(file)
            return settings["Environment"]["Name"]
        except Exception:
            traceback.print_exc()
            return ""

    def set_environment_name(self, settings_file_name: str, map_name: str) -> None:
        """
        Set the environment name in the settings file.
        """
        with open("{}".format(settings_file_name), "r") as file:
            settings = json.load(file)
            
        settings["Environment"]["Name"] = map_name

        with open("{}".format(settings_file_name), "w") as file:
            json.dump(settings, file, indent=4)

        return

    def load_map_metadata(self, level_name: str, env_name: str) -> None:
        """
        Load the metadata for the map to help with the coordinates
        """
        with open("maps/{}_metadata_{}.json".format(env_name, level_name), "r") as file:
            metadata = json.load(file)

        return metadata

    def display_map_image_with_coordinates_v1(self) -> None:
        """
        Display the map of the selected environment with the corrected
        coordinate frame. This will show up in MatPlotLib correctly
        so that a user can decide where to send the agent.

        NOTE Map name is "ENVIRONMENT_NAME".png
        TODO Add levels to this as we advance the environment
        """
        img = plt.imread("maps/{}.png".format(self.map_name))
        self.load_map_metadata()
        plt.imshow(img)
        fig = plt.gcf()
        for ax in fig.axes:
            scale = 1e1  # Map is in centimeters
            # Shift the axes of the map by half the bounds of the image, then
            # scale, then reverse both axes to make sense for NED coordinates
            ticks_x = ticker.FuncFormatter(
                lambda x, pos: "{0:g}".format(
                    1 * (x - (self.map_metadata["ImageSize"][0] / 2)) / scale
                )
            )
            ax.xaxis.set_major_formatter(ticks_x)
            ax.yaxis.set_major_formatter(ticks_x)
        plt.xlabel("X Coordinate (meters)")
        plt.ylabel("Y Coordinate (meters)")
        plt.show()

    def display_map_image(
        self, level_names: list, env_name: str, maps_dir: str = "maps"
    ) -> None:
        """
        Display a Map Image of the Environment with no trajectory

        ### Inputs:
        - level_names [list] The levels to display
        - env_name [str] The name of the environment to run
        - maps_dir [str] The folder where the map system exists
        """
        for level_name in level_names:
            metadata = self.load_map_metadata(level_name, env_name)
            img = plt.imread("maps/{}_{}.png".format(env_name, level_name))
            plt.imshow(img)
            fig = plt.gcf()

            for ax in fig.axes:
                # ax.axis('off')
                # ax.margins(0,0)
                # ax.xaxis.set_major_locator(plt.NullLocator())
                # ax.yaxis.set_major_locator(plt.NullLocator())
                # scale = 1.925
                ticks_x = ticker.FuncFormatter(
                    lambda x, pos: "{0:g}".format(
                        1 * (x - metadata["Offset"][0]) / metadata["Scale"]
                    )
                )
                ticks_y = ticker.FuncFormatter(
                    lambda x, pos: "{0:g}".format(
                        1 * (x - metadata["Offset"][1]) / metadata["Scale"]
                    )
                )
                ax.xaxis.set_major_formatter(ticks_x)
                ax.yaxis.set_major_formatter(ticks_y)
            plt.plot(
                metadata["Offset"][0] + metadata["Origin"][0],
                metadata["Offset"][1] + metadata["Origin"][1],
                marker=".",
                color="white",
                label="Origin",
            )
            plt.plot(
                metadata["Offset"][0] + metadata["Origin"][0] + 7,
                metadata["Offset"][1] + metadata["Origin"][1],
                marker=">",
                color="blue",
                label="Origin",
            )
            plt.text(
                metadata["Offset"][0] + metadata["Origin"][0] - 10,
                metadata["Offset"][1] + metadata["Origin"][1] - 8,
                "Origin",
                color="white",
            )
            plt.text(
                metadata["Offset"][0] + metadata["Origin"][0] + 30,
                metadata["Offset"][1] + metadata["Origin"][1],
                "Front",
                color="white",
            )
            plt.xlabel("X Coordinate (meters)")
            plt.ylabel("Y Coordinate (meters")
            plt.show()

    def display_map_image_with_trajectories(
        self, level_names: list, env_name: str, maps_dir: str = "maps"
    ) -> None:
        """
        Display the map image with the trajectory overlayed.
        """

        for level_name in level_names:
            img = plt.imread("{}/{}_{}.png".format(maps_dir, env_name, level_name))

            map_metadata = self.load_map_metadata(level_name, env_name)
            plt.imshow(img)
            fig = plt.gcf()
            capture_size = map_metadata["CaptureSize"]
            grid_marks = [
                measure * (1 / map_metadata["CaptureIncrement"])
                for measure in capture_size
            ]
            origin_offset = map_metadata["Origin"]

            for ax in fig.axes:
                scale = 1e1  # Map is in centimeters
                # Shift the axes of the map by half the bounds of the image, then
                # scale, then reverse both axes to make sense for NED coordinates
                ticks_x = ticker.FuncFormatter(
                    lambda x, pos: "{0:g}".format(
                        1 * (x - (map_metadata["ImageSize"][0] / 2)) / scale
                    )
                )
                ax.xaxis.set_major_formatter(ticks_x)
                ax.yaxis.set_major_formatter(ticks_x)

            origin = [
                grid_marks[0] / 2 + origin_offset[0],
                grid_marks[1] / 2 + origin_offset[1],
            ]
            plt.plot(origin[0], origin[1], marker=".", color="red", label="1")

            # load the waypoints from the json file
            with open("settings/DefaultTrajectory.json", "r") as file:
                trajectory = json.load(file)
            # level = map_name[(map_name.index("_") + 1):]
            trajectory = trajectory["Trajectory"][level_name]

            # Loop through the waypoints and plot the points
            trajectorylist = [origin]
            n = 0
            scale = 11.5
            for waypoint in trajectory:
                plt.plot(
                    origin[0] + (waypoint["X"] * scale),
                    origin[1] + (waypoint["Y"] * scale),
                    marker=".",
                    color="blue",
                    label=n,
                )
                trajectorylist.append(
                    [
                        origin[0] + (waypoint["X"] * scale),
                        origin[1] + (waypoint["Y"] * scale),
                    ]
                )
                n += 1
            # Loop through the points and draw a line connecting the current point to the next point
            for i in range(len(trajectorylist) - 1):
                x = [trajectorylist[i][0], trajectorylist[i + 1][0]]
                y = [trajectorylist[i][1], trajectorylist[i + 1][1]]
                plt.plot(x, y, color="green", linewidth=2)

            plt.xlabel("X Coordinate (meters)")
            plt.ylabel("Y Coordinate (meters)")

            # Save the trajectory map to the data assets folder
            # TODO: Define the location of the UE project content folder
            print("Saving map with trajecotry to /{}".format(maps_dir))
            plt.savefig("{}/{}_{}_Display.png".format(maps_dir, env_name, level_name))

            plt.show()
            plt.close()

    def generate_submission_tracking(self) -> None:
        """
        Generate the necessary files for submission tracking and
        history, so that users can track what settings have been
        changed over time.

        ### Inputs:
        - None

        ### Outputs:
        - None
        """
        print("Checking if submission history exsits")
        try:
            if self._file_path is not None:
                folder_path = self._file_path + "/settings"
            else:
                folder_path = find_folder_path("settings")
            os.chdir(folder_path)
        except FileNotFoundError:
            print("Settings folder not found!")
            print("Creating settings folder")
            if self._file_path is not None:
                os.mkdir(self._file_path + "/settings")
                folder_path = self._file_path + "/settings"
                os.chdir(folder_path)
            else:
                # We are running this from the root directory of the Client
                # repo and don't need to worry about paths
                os.mkdir("settings")
                folder_path = find_folder_path("settings")
                os.chdir(folder_path)
        # We should now be in file_path + "/settings"
        files = glob.glob("*.json")
        if "SubmissionHistory.json" not in files:
            print("Generating new submission history")
            sub_histroy = {"History": list()}
            if self._file_path is not None:
                file_path = self._file_path + "/settings/SubmissionHistory.json"
            else:
                file_path = find_folder_path("settings") + "/SubmissionHistory.json"
            with open(file_path, "a") as file:
                json.dump(sub_histroy, file)
        if "SubmissionList.json" not in files:
            if self._file_path is not None:
                file_path = self._file_path + "/settings/SubmissionList.json"
            else:
                file_path = find_folder_path("settings") + "/SubmissionList.json"
            print("Generating new submission list")
            sub_list = {"Submissions": {}}
            with open(file_path, "a") as file:
                json.dump(sub_list, file)
        os.chdir("..")
        print("File path is now {}".format(os.getcwd()))
        if self._file_path is not None:
            print("User File path is now {}".format(self._file_path))
        print("Submission history has been successfully setup!")
        return

    def _get_scenarios_with_trajectories(self) -> list:
        """
        Access the SupportedScenarios.json file and determine which
        environments have trajectories by reading the member named
        "TrajectoryRequired".

        ### Inputs:
        - None

        ### Outputs:
        - A list of environments that have trajectories
        """
        scenarios = self._get_supported_scenarios()
        scenarios_with_trajectories = list()
        for scenario_name, details in scenarios.items():
            if details["TrajectoryRequired"]:
                scenarios_with_trajectories.append(scenario_name)
        return scenarios_with_trajectories

    def _set_using_trajectory(self, settings_file_name: str) -> None:
        """
        Read in the Settings file and extract the Scenario name,
        determining if we are using a trajectory or not.

        ### Inputs:
        - settings_file_name [str] The name of the settings file

        ### Outputs:
        - None
        """
        if self._file_path is not None:
            file_path = self._file_path + "/settings/" + settings_file_name
        else:
            file_path = find_file_path(settings_file_name, "settings")
        with open(file_path, "r") as file:
            settings = json.load(file)

        if settings["Scenario"]["Name"] in self._get_scenarios_with_trajectories():
            self._has_trajectory = True
        else:
            self._has_trajectory = False

    def build_simulation(
        self,
        map_name: str,
        custom_name: str = None,
        settings_file_name: str = "DefaultSimulationSettings.json",
        trajectory_file_name: str = "DefaultTrajectory.json",
        submission_list_location: str = "settings",
    ) -> str:
        """
        Build a Simulation Package using the setup from the previous
        step. This creates a new unique name and id to reference all of
        the simulation steps, including the ability to load this data
        into the SWARM Web Portal for viewing.

        ### Inputs:
        - custom_name [str] A custom name that can be set instead of a
                            random UUID.

        ### Outputs:
        - The simulation name as a string
        """
        valid = self.validate_environment_name(map_name)

        if not valid:
            print("Environment name is invalid!")
            raise AssertionError("Environment name {} is invalid!".format(map_name))

        print("Reading map name from {}".format(settings_file_name))
        if self._file_path is not None:
            file_name = self._file_path + "/" + submission_list_location  + "/" + settings_file_name
        else:
            file_name = submission_list_location  + "/" + settings_file_name
        settings_map_name = self.read_map_name_from_settings(file_name)
        if map_name != settings_map_name:
            self.set_environment_name(file_name, map_name)

        if custom_name:
            sim_name = custom_name
        else:
            sim_name = uuid4()
            sim_name = sim_name.__str__()

        self._set_using_trajectory(settings_file_name)

        settings, trajectory = self.generate_simulation_package(
            submission_list_location + "/" + settings_file_name,
            submission_list_location + "/" + trajectory_file_name,
            sim_name,
        )

        if self._file_path is not None:
            file_path = self._file_path + "/" + submission_list_location + "/SubmissionList.json"
        else:
            file_path = find_file_path("SubmissionList.json", submission_list_location)

        with open(file_path, "r") as file:
            submission_list = json.load(file)

        print(submission_list["Submissions"].keys())
        print("Building submission package for {}".format(sim_name))
        if sim_name in submission_list["Submissions"].keys():
            submission_list["Submissions"][sim_name]["Settings"] = settings
            submission_list["Submissions"][sim_name]["Trajectory"] = trajectory
            submission_list["Submissions"][sim_name]["Number Of Runs"] += 1
        else:
            submission_list["Submissions"][sim_name] = {
                "Completed": False,
                "Settings": settings,
                "Trajectory": trajectory,
                "Created": convert_datetime_to_str(datetime.datetime.now()),
                "Submitted": True,
                "Number Of Runs": 1,
            }

        with open(file_path, "w") as file:
            json.dump(submission_list, file)

        return sim_name

    def generate_simulation_package(
        self, settings_file_name: str, trajectory_file_name: str, sim_name: str
    ) -> tuple:
        """
        Generate the Submission package by reading hte settings and the
        Trajectory file (assumign a Trajectory is required)

        ### Inputs:
        - settings_file_name [str] The name of the settings file
        - trajectory_file_name [str] The name of the trajectory file
        - sim_name [str] The name of the simulation

        ### Outputs:
        - The settings and trajectory as JSON strings
        """
        if self._file_path is not None:
            settings_file_name = self._file_path + "/" + settings_file_name
            trajectory_file_name = self._file_path + "/" + trajectory_file_name

        settings = self.read_json_file(settings_file_name)
        settings = self.add_simulation_name_to_settings(
            sim_name, settings, settings_file_name
        )
        settings = self._update_all_sensor_settings(settings, settings_file_name)
        if self._has_trajectory:
            trajectory = self.read_json_file(trajectory_file_name)
        else:
            trajectory = None
        return settings, trajectory

    def add_simulation_name_to_settings(
        self, sim_name: str, settings: str, settings_file_name: str
    ) -> str:
        """
        Update the name of the simulation name in the settings file.
        """
        settings = json.loads(settings)
        settings["SimulationName"] = sim_name
        with open("{}".format(settings_file_name), "w") as file:
            json.dump(settings, file, indent=4)
        settings = json.dumps(settings)
        return settings

    def _update_all_sensor_settings(
        self, settings: dict, settings_file_name: str
    ) -> str:
        """
        Update and fill in all Sensor Settings that aren't provided,
        so we show the user what is possible but don't need to
        validate the settings just yet.

        ### Inputs:
        - settings [dict] The User settings file

        ### Outputs:
        - The updated settings files as a JSON string
        """
        valid_sensor_info = self._retrieve_valid_sensor_info()

        settings = json.loads(settings)

        # If they have provided agents
        if "Agents" in settings.keys():
            for agent_name, agent_info in settings["Agents"].items():
                # If they attached sensors to an agent
                if "Sensors" in agent_info.keys():
                    for sensor_type_name, sensors in agent_info["Sensors"].items():
                        # If the sensor is a GPS or IMU device
                        if sensor_type_name == "GPS" or sensor_type_name == "IMU":
                            for sensor_name, sensor_info in sensors.items():
                                # Go through each device, determining if the User included these parameters
                                for param_name, param_info in valid_sensor_info[
                                    sensor_type_name
                                ]["Parameters"].items():
                                    if param_name not in sensor_info.keys():
                                        settings["Agents"][agent_name]["Sensors"][
                                            sensor_type_name
                                        ][sensor_name][param_name] = param_info[
                                            "Default"
                                        ]

        with open("{}".format(settings_file_name), "w") as file:
            json.dump(settings, file, indent=4)
        settings = json.dumps(settings)
        return settings

    def read_json_file(self, file_name: str) -> str:
        """
        Read a JSON file and then return the string form of the file so
        that we can send this to the server.

        ### Inputs:
        - file_name [str] The file to read from

        ### Outputs:
        - The contents of the JSON file as a JSON string
        """
        with open(file_name, "r", encoding="utf-8") as file:
            code = json.load(file)

        return json.dumps(code)

    def retrieve_sim_package(self, sim_name: str, folder: str = "settings") -> tuple:
        """
        Retrieve the settings files as JSON strings to send to the
        server.

        ### Inputs:
        - sim_name [str] The unique simulation name

        ### Outputs:
        - The settings, trajectory and vehicle settings to send to the
          server
        """
        if self._file_path is not None:
            file_path = self._file_path + "/" + folder + "/" + "SubmissionList.json"
        else:
            file_path = find_file_path("SubmissionList.json", folder)
        with open(file_path, "r") as file:
            submission_list = json.load(file)

        try:
            settings = submission_list["Submissions"][sim_name]["Settings"]
            trajectory = submission_list["Submissions"][sim_name]["Trajectory"]
        except KeyError:
            traceback.print_exc()
            print(
                "{} doesn't exist in the submission list! Please try a different name!".format(
                    sim_name
                )
            )
            return {}, {}

        return settings, trajectory

    def update_submission_list(self, message: dict, sim_name: str = None, folder: str = "settings") -> None:
        """
        Once we have received a finish message, update the submission
        list to reflect this fact.

        ### Inputs:
        - message [dict] - The message received from the client

        Message should have the following format:
        {
            "Sim_name": self.settings["SimulationName"],
            "Status": "NotCompleted",
            "Minutes": 5
            "Seconds": 12
        }

        ### Outputs:
        - None
        """
        if self._file_path is not None:
            list_file_path = self._file_path + "/" + folder + "/SubmissionList.json"
        else:
            list_file_path = find_file_path("SubmissionList.json", folder)
        with open(list_file_path, "r") as file:
            sub_list = json.load(file)

        if self._file_path is not None:
            history_file_path = self._file_path + "/" + folder + "/SubmissionHistory.json"
        else:
            history_file_path = find_file_path("SubmissionHistory.json", folder)

        with open(history_file_path, "r") as file:
            history = json.load(file)

        if sim_name is None:
            try:
                sim_name = message["Sim_name"]
            except KeyError:
                print("Error! No simulation name was provided!")
                return
        try:
            submission = sub_list["Submissions"][sim_name]
            if message["Status"] == "Completed":
                submission["Completed"] = True
                print("Simulation {} has been completed!".format(sim_name))
                print(
                    "Completion Time: {} Mins {} Seconds".format(
                        message["Minutes"], message["Seconds"]
                    )
                )
            history["History"].append(submission)
            with open(list_file_path, "w") as file:
                json.dump(sub_list, file)
            with open(history_file_path, "w") as file:
                json.dump(history, file)
        except KeyError:
            traceback.print_exc()
            print("This simulation was not apart of the Submission List")
        except Exception:
            traceback.print_exc()

    def access_environment_list(self) -> list:
        """
        Retrieve the environment list to determine what environments
        are supported.
        """
        try:
            files = os.listdir("settings")
            assert "SupportedEnvironments.json" in files
            with open("settings/SupportedEnvironments.json", "r") as file:
                envs = json.load(file)
            return envs["Environments"]
        except AssertionError:
            print("The file settings/SupportedEnvironments.json does not exist!")
            return False
        except Exception:
            traceback.print_exc()
            return False

    # =========================================================================
    #                       Validation Functions
    # =========================================================================

    def validate_environment_name(
        self, map_name: str, folder: str = "settings"
    ) -> bool:
        """
        Given a User-defined string, validate that the string contains
        a supported map name of an environment the SWARM Core system
        actually supports.

        ### Inputs:
        - map_name [str] The map name to run

        ### Outputs:
        - A boolean describing whether the map name is valid
        """
        try:
            assert isinstance(map_name, str)
            if self._file_path is not None:
                file_path = self._file_path + "/" + folder + "/SupportedEnvironments.json"
            else:
                file_path = find_file_path("SupportedEnvironments.json", folder)
            with open(file_path, "r") as file:
                envs = json.load(file)

            if map_name in envs["Environments"].keys():
                return True
            else:
                return False
        except AssertionError:
            print("The provided map name is not a string!")
            return False
        except Exception:
            traceback.print_exc()
            return False

    def validate_scenario_name(
        self, scenario_name: str, folder: str = "settings"
    ) -> bool:
        try:
            assert isinstance(scenario_name, str)
            if self._file_path is not None:
                file_path = self._file_path + "/" + folder + "/SupportedScenarios.json"
            else:
                file_path = find_file_path("SupportedScenarios.json", folder)
            with open(file_path, "r") as file:
                scenarios = json.load(file)
            scenario_name = "".join(scenario_name.split(" "))
            if scenario_name in scenarios["Scenarios"].keys():
                return True
            else:
                print("Error! {} is not a supported scenario".format(scenario_name))
                return False
        except AssertionError:
            print("The provided scenario name is not a string!")
            return False
        except Exception:
            traceback.print_exc()
            return False

    def validate_settings_file(self, settings_file: dict) -> bool:
        """
        Validate the settings file provided by the user before sending
        this to the SWARM Core Server. Provide information about
        acceptable values for each parameter field.

        ### Inputs:
        - settings_file [dict] The JSON file as a dictionary containing
                               the settings for the SWARM Core system

        ### Outputs:
        - A boolean indicating whether the file is valid or not
        """
        key_components = [
            "ID",
            "RunLength",
            "SimulationName",
            "Scenario",
            "Environment",
            "Agents",
        ]
        if "Data" in settings_file.keys():
            key_components.append("Data")
        try:
            # First, validate that all required sections are present
            try:
                assert list(settings_file.keys()).sort() == key_components.sort()
            except AssertionError:
                statement = (
                    "The required components are not containined in this file.\n"
                )
                statement += "Required Components: {}\n".format(key_components)
                statement += "What you submitted: {}\n".format(
                    list(settings_file.keys())
                )
                print("The required components are not containined in this file.")
                print("Required Components: {}".format(key_components))
                print("What you submitted: {}".format(list(settings_file.keys())))
                return False, statement

            # Next, validate each section
            for key, options in settings_file.items():
                print("\nValidating section {}".format(key))
                if key == "ID":
                    try:
                        assert isinstance(options, int)
                    except AssertionError:
                        statement = "The required ID is not an integer. Please regenerate this file."

                        print(statement)
                        return False, statement
                elif key == "RunLength":
                    try:
                        assert isinstance(options, int) or isinstance(options, float)
                        if not (options >= 10.0 and options <= 9999):
                            raise ValueError()
                    except AssertionError:
                        statement = "The required Run Length is not an integer or float. Please fix this!"
                        print(
                            statement
                        )
                        return False, statement
                    except ValueError:
                        statement = "The Run Length specified is not a valid value.\n"
                        statement += "Valid values are 10.0 to 9999.0 seconds\n"
                        statement += "Please change this value in your settings file"
                        print(statement)

                        return False, statement
                elif key == "Scenario":
                    valid_keys = ["Name", "Options"]
                    try:
                        if valid_keys != list(options.keys()):
                            raise AssertionError(
                                "Error! Please ensure you have the following sections: {}".format(
                                    valid_keys
                                )
                            )
                        if not self.validate_scenario_name(options["Name"]):
                            raise AssertionError(
                                "Scenario with name {} is not supported!".format(
                                    options["Name"]
                                )
                            )
                        if "LevelNames" in options["Options"]:
                            for level_name in options["Options"]["LevelNames"]:
                                if not self.validate_level_is_supported(
                                    settings_file["Environment"]["Name"], level_name
                                ):
                                    raise AssertionError(
                                        "Level {} is not supported for environment with name {}!".format(
                                            level_name,
                                            settings_file["Environment"]["Name"],
                                        )
                                    )
                        if "MultiLevel" in options["Options"]:
                            if not isinstance(options["Options"]["MultiLevel"], bool):
                                raise AssertionError(
                                    "Error! The field MultiLevel must be a boolean value!"
                                )
                        if "GoalPoint" in options["Options"]:
                            if len(list(options["Options"]["GoalPoint"].keys())) != len(
                                list(settings_file["Agents"].keys())
                            ):
                                raise AssertionError(
                                    "\n\nError!\nYou must have the same number of goal points as agents!\nYou defined {} agents and {} goal points".format(
                                        len(
                                            list(options["Options"]["GoalPoint"].keys())
                                        ),
                                        len(list(settings_file["Agents"].keys())),
                                    )
                                )
                            for agent_name, goal in options["Options"][
                                "GoalPoint"
                            ].items():
                                if type(goal).__name__ != "dict" or [
                                    "X",
                                    "Y",
                                    "Z",
                                ] != list(goal.keys()):
                                    raise AssertionError(
                                        "\n\nError for agent {}!\n You must define a dictonary with the keys 'X', 'Y' and 'Z' in NED coordaintes\n Your input was of type {}".format(
                                            agent_name, type(goal).__name__
                                        )
                                    )
                                for coord, coord_value in goal.items():
                                    if type(coord_value).__name__ != "float":
                                        raise AssertionError(
                                            "\n\nError!\n Coordinate {} must be of type float\n Your input was of type {}".format(
                                                coord, type(coord_value).__name__
                                            )
                                        )
                                    if coord_value < -999.0 or coord_value > 999.0:
                                        raise AssertionError(
                                            "\n\nError!\n Coordinate {} must be between -999.0 and 999.0 meters!\n Your input was {}".format(
                                                coord, coord_value
                                            )
                                        )
                    except AssertionError as error:
                        print(error)
                        return False, str(error)
                elif key == "Environment":
                    valid_keys = ["Name", "StreamVideo", "StartingLevelName", "Options"]
                    try:
                        if not "Name" in list(options.keys()):
                            raise AssertionError(
                                "You must provide a Name that is in the environment"
                            )
                        if not self.validate_environment_name(options["Name"]):
                            raise AssertionError(
                                "Environment name {} is not valid!".format(
                                    options["Name"]
                                )
                            )
                        if not isinstance(options["StreamVideo"], bool):
                            raise AssertionError(
                                "Error! Please ensure the StreamVideo section is a boolean value"
                            )
                        if not isinstance(options["StartingLevelName"], str):
                            raise AssertionError(
                                "Error! Please ensure the Level name is a string value"
                            )
                        if (
                            options["StartingLevelName"]
                            not in settings_file["Scenario"]["Options"]["LevelNames"]
                        ):
                            raise AssertionError(
                                "Error! You must choose a level name that is in the Scenario Levels selected!\nValid options are: {}".format(
                                    settings_file["Scenario"]["Options"]["LevelNames"]
                                )
                            )
                        if not self.validate_level_is_supported(
                            options["Name"], options["StartingLevelName"]
                        ):
                            raise AssertionError(
                                "Level selected is not supported by this environment"
                            )
                        self.validate_environment_options(options)
                    except AssertionError as error:
                        print(error)
                        return False, str(error)
                elif key == "Data":
                    valid_options = ["Images", "VehicleState", "Video"]
                    valid_image_types = ["PNG"]
                    valid_video_types = ["MP4"]
                    try:
                        for data_type in options.keys():
                            print("Validating Data Type: {}".format(data_type))
                            if not data_type in valid_options:
                                raise AssertionError(
                                    "Error! {} is not a valid type of data to collect".format(
                                        data_type
                                    )
                                )
                            if data_type == "Images":
                                if not options["Images"]["Format"] in valid_image_types:
                                    raise AssertionError(
                                        "Error!\n{} is not a valid image format to save collected images".format(
                                            data_type
                                        )
                                    )
                                if not "ImagesPerSecond" in list(
                                    options[data_type].keys()
                                ):
                                    raise AssertionError(
                                        "\nError!\nYou must include the parameter 'ImagesPerSecond' in the Images options!\n"
                                    )
                                if not isinstance(
                                    options[data_type]["ImagesPerSecond"], int
                                ):
                                    raise AssertionError(
                                        "\nError!\nImagesPerSecond must be a integer value!!\n"
                                    )
                                if (
                                    options[data_type]["ImagesPerSecond"] > 20
                                    or options[data_type]["ImagesPerSecond"] < 1
                                ):
                                    raise AssertionError(
                                        "\nError!\nImagesPerSecond must be within the range of 1 to 20 images!\n"
                                    )
                            elif data_type == "Video":
                                if "Format" not in options["Video"].keys():
                                    raise AssertionError(
                                        "Error!\n'Format' is a required key in this section! Video Format options are {}".format(
                                            valid_video_types
                                        )
                                    )
                                if "VideoName" not in options["Video"].keys():
                                    raise AssertionError(
                                        "\nError!\n'VideoName' is a required key in this section!\nThis must be a string!\n\n"
                                    )
                                if "CameraName" not in options["Video"].keys():
                                    raise AssertionError(
                                        "\nError!\n'CameraName' is a required key in this section!\nThis must be a string!\n\n"
                                    )
                                if not options["Video"]["Format"] in valid_video_types:
                                    raise AssertionError(
                                        "Error!\n{} is not a valid video format!".format(
                                            options["Video"]["Format"]
                                        )
                                    )
                                if not isinstance(options["Video"]["VideoName"], str):
                                    raise AssertionError(
                                        "Error!\nVideo name must be a string! \n"
                                    )
                                if not isinstance(options["Video"]["CameraName"], str):
                                    raise AssertionError(
                                        "Error!\n\n Data Options error!\n Camera name must be a string! \n"
                                    )
                                camera_names = list()
                                for agent_name, agent_options in settings_file[
                                    "Agents"
                                ].items():
                                    if "Cameras" not in agent_options["Sensors"].keys():
                                        raise AssertionError(
                                            "Error!\n\nYou must add a Cameras section to your settings to record video!"
                                        )
                                    for camera_name in agent_options["Sensors"][
                                        "Cameras"
                                    ].keys():
                                        if camera_name not in camera_names:
                                            camera_names.append(camera_name)
                                if options["Video"]["CameraName"] not in camera_names:
                                    raise AssertionError(
                                        "Error!\n\nNo Agent in your settings has a camera with name: {}\nPlease correct this by adding a Camera with that name!!".format(
                                            options["Video"]["CameraName"]
                                        )
                                    )
                    except AssertionError as error:
                        print(error)

                        return False, str(error)
                elif key == "Agents":
                    # Test that they have an agent
                    try:
                        numb_agents = len(list(options.keys()))
                        if numb_agents < 1 or numb_agents > 5:
                            raise AssertionError(
                                "\n\nError!\nYou must have at least one agent but can only have up to 5!"
                            )
                    except Exception as error:
                        print(error)
                        # This is a critical error that should not allow them to continue
                        return False
                    # Now, iterate through each agent and check that the modules are there
                    for agent, agent_options in options.items():
                        print("Validating {}".format(agent))
                        # Test that the appropriate sections exist
                        valid_agent_sections = [
                            "Vehicle",
                            "AutoPilot",
                            "Sensors",
                            "Controller",
                            "SoftwareModules",
                            "StartingPosition",
                            "VehicleOptions",
                            "VehiclePhysicsProfile"
                        ]
                        valid_agent_sections.sort()
                        try:
                            user_agent_options = list(agent_options.keys())
                            user_agent_options.sort()
                            if not user_agent_options == valid_agent_sections:
                                raise AssertionError(
                                    "Sections for {} are not valid!\nYour Options: {}\nValid Options: {}".format(
                                        agent, user_agent_options, valid_agent_sections
                                    )
                                )
                        except Exception as error:
                            traceback.print_exc()
                            return False
                        for section_name, section in agent_options.items():
                            if section_name == "Vehicle":
                                if not section == "Multirotor":
                                    raise AssertionError(
                                        "Vehicle parameter must be Multirotor. Support for different vehicle types coming soon!"
                                    )
                            elif section_name == "VehiclePhysicsProfile":
                                self._validate_vehicle_physics_profile(section, agent_options["Vehicle"])
                            elif section_name == "VehicleOptions":
                                if not isinstance(section, dict):
                                    raise AssertionError(
                                        "Error!\n Section VehicleOptions must be a dictionary!\nYour input was of type: {}".format(
                                            type(section).__name__
                                        )
                                    )
                                valid_options = [
                                    "RunROSNode",
                                    "UseLocalPX4",
                                    "PlanningCoordinateFrame",
                                    "LocalHostIP",
                                ]
                                for option_key, option_value in section.items():
                                    if option_key not in valid_options:
                                        raise AssertionError(
                                            "Error!\nOption {} for Vehicle Options for agent {} is not in the list of valid options!".format(
                                                option_key, agent
                                            )
                                        )
                                    if option_key == "RunROSNode":
                                        if not isinstance(section[option_key], bool):
                                            raise AssertionError(
                                                "Error!\nOption {} must be of type bool.\nYour input was of type {}".format(
                                                    option_key,
                                                    type(option_value).__name__,
                                                )
                                            )
                                        if (
                                            "PlanningCoordinateFrame"
                                            not in section.keys()
                                        ):
                                            raise AssertionError(
                                                "Error!\nOption 'PlanningCoordinateFrame' must be included if you are running ROS. Options are NED and ENU."
                                            )
                                    if option_key == "PlanningCoordinateFrame":
                                        if not isinstance(section[option_key], str):
                                            raise AssertionError(
                                                "Error!\nOption {} must be of type string.\nYour input was of type {}".format(
                                                    option_key,
                                                    type(option_value).__name__,
                                                )
                                            )
                                        if section[option_key] not in ["NED", "ENU"]:
                                            raise AssertionError(
                                                "Error for agent {}!\nOption 'PlanningCoordinateFrame' can only be NED and ENU.".format(
                                                    agent
                                                )
                                            )
                                    if option_key == "UseLocalPX4":
                                        if not isinstance(section[option_key], bool):
                                            raise AssertionError(
                                                "Error!\nOption {} must be of type bool.\nYour input was of type {}".format(
                                                    option_key,
                                                    type(option_value).__name__,
                                                )
                                            )
                                    if option_key == "LocalHostIP":
                                        if not isinstance(option_value, str):
                                            raise AssertionError(
                                                "Error!\n\nLoclHostIP must be of type str. \nYour input was of type {}".format(
                                                    type(option_value).__name__
                                                )
                                            )
                                        try:
                                            ipaddress.ip_address(option_value)
                                            print(
                                                "Provided IP address has been validated!"
                                            )
                                        except ValueError:
                                            raise AssertionError(
                                                "Error!\n\nThe provided IP address is not a valid IPV4 address!\nYour input was: {}".format(
                                                    option_value
                                                )
                                            )
                            elif section_name == "StartingPosition":
                                if (
                                    "X" not in section.keys()
                                    or "Y" not in section.keys()
                                    or "Z" not in section.keys()
                                ):
                                    raise AssertionError(
                                        "\n\n Error for Agent {}!\nYou must have X, Y and Z as keys to a dictionary for the StartingPosition!\nYour Input: {}".format(
                                            agent, section.keys()
                                        )
                                    )
                                for key, pos in section.items():
                                    if type(pos).__name__ != "float":
                                        raise AssertionError(
                                            "\n\n Error for Agent {}!\n{} must be of type float!\nYour Input: {}".format(
                                                agent, key, pos
                                            )
                                        )
                                    if pos > 999.0 or pos < -999.0:
                                        raise AssertionError(
                                            "\n\n Error for Agent {}!\n{} must be of type within -999.0 and 999.0 meters!\nYour Input: {}".format(
                                                agent, key, pos
                                            )
                                        )
                            elif section_name == "AutoPilot":
                                if not section == "SWARM" and not section == "PX4":
                                    raise AssertionError(
                                        "Autopilot parameter must be SWARM or PX4. Support for different autopilots, including PX4/Ardupilot coming in February 2023!"
                                    )
                            elif section_name == "Sensors":
                                self.validate_sensors(
                                    section,
                                    settings_file["Agents"][agent]["AutoPilot"],
                                    agent_name=agent,
                                )
                                # Update the broader file
                                settings_file["Agents"][agent][section_name] = section
                            elif section_name == "Controller":
                                valid_controller_sections = ["Name", "Gains"]
                                for (
                                    controller_section_name,
                                    controller_setting,
                                ) in section.items():
                                    if (
                                        controller_section_name
                                        not in valid_controller_sections
                                    ):
                                        raise AssertionError(
                                            "Controller for {} has an invalid field!\nYour input: {}\nValid Inputs: {}".format(
                                                agent,
                                                controller_section_name,
                                                valid_controller_sections,
                                            )
                                        )
                                    if controller_section_name == "Name":
                                        if not controller_setting == "PID":
                                            raise AssertionError(
                                                "Controller for {} type is is invalid. Name must be PID. Support for different controllers will be available in March 2023!".format(
                                                    agent
                                                )
                                            )
                                    elif controller_section_name == "Gains":
                                        for (
                                            gain_key,
                                            gain,
                                        ) in controller_setting.items():
                                            if not isinstance(gain, float):
                                                raise AssertionError(
                                                    "Gain {} for controller for {} must be a float value!".format(
                                                        gain_key, agent
                                                    )
                                                )
                                            if gain < 0.0 or gain > 20.0:
                                                raise AssertionError(
                                                    "Gain {} for controller for {} must be between 0.0 and 20.0!".format(
                                                        gain_key, agent
                                                    )
                                                )
                            elif section_name == "SoftwareModules":
                                self.validate_software_modules(
                                    section,
                                    agent,
                                    settings_file["Agents"][agent]["Sensors"],
                                )
            return True, "Success! Your settings file has been validated!"
        except Exception as error:
            traceback.print_exc()
            print(error)
            statement = error
            return False, str(error)

    def _validate_vehicle_physics_profile(self, vehicle_physics_profile: str, vehicle_type: str):
        """
        Validate that the provided vehicle physics profile has been
        defined in the VehicleProfiles.json file in the vehicle_profiles
        folder.

        ### Inputs:
        ----
        `vehicle_physics_profile` : str\n
        The vehicle physics profile to validate.

        `vehicle_type` : str\n
        The vehicle type to validate.

        ### Returns:
        ----
        `valid` : bool\n
        """
        if not isinstance(vehicle_physics_profile, str):
            raise AssertionError(
                "Error!\n\nThe vehicle physics profile must be of type str.\nYour input was of type {}".format(
                    type(vehicle_physics_profile).__name__
                )
            )
        vehicle_profiles = self._retrieve_valid_vehicle_physics_profiles(vehicle_type)
        if vehicle_physics_profile not in vehicle_profiles.keys():
            raise AssertionError(
                "Error!\n\nThe vehicle physics profile {} is not a valid vehicle physics profile.\nValid vehicle physics profiles are: {}".format(
                    vehicle_physics_profile, vehicle_profiles.keys()
                )
            )
    
        # Check that the related json file for the vehicle physics profile exists
        if self._file_path is not None:
            file_path = self._file_path + "/vehicle_profiles/{}".format(vehicle_profiles[vehicle_physics_profile])
        else:
            file_path = find_file_path(vehicle_profiles[vehicle_physics_profile], "vehicle_profiles")
        
        if not os.path.exists(file_path):
            raise AssertionError(
                "Error!\n\nThe vehicle physics profile {} is not a valid vehicle physics profile.\nValid vehicle physics profiles are: {}".format(
                    vehicle_physics_profile, vehicle_profiles.keys()
                )
            )

    def _retrieve_valid_vehicle_physics_profiles(self, vehicle_type: str) -> dict:
        """
        Retrieve the valid physics profiles for the provided vehicle
        type.
        """
        if self._file_path is not None:
            file_path = self._file_path + "/vehicle_profiles/VehicleProfiles.json"
        else:
            file_path = find_file_path("VehicleProfiles.json", "vehicle_profiles")
        
        with open(file_path, "r") as file:
            json_file = json.load(file)
        
        if vehicle_type not in json_file.keys():
            raise AssertionError(
                "Error!\n\nThe vehicle type {} is not a valid vehicle type.\nValid vehicle types are: {}".format(
                    vehicle_type, json_file.keys()
                )
            )
        
        return json_file[vehicle_type]

    def _retrieve_valid_sensor_info(self) -> dict:
        """
        Retrieve the SupportedSensors JSON file
        """
        try:
            if self._file_path is not None:
                file_path = self._file_path + "/" + "SWARMRDS/core/SupportedSensors.json"
            else:
                file_path = find_file_path(
                    "SupportedSensors.json", "SWARMRDS/core"
                )
            with open(file_path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            raise AssertionError(
                "Error!\nThe file 'SupportedSensors.json' was not found in the core folder of this client! Please make sure this file exists!"
            )

    def _validate_sensor_setting(
        self,
        sensor_name: str,
        sensor_setting_key: str,
        valid_sections: dict,
        sensor_setting: None,
        sensor_type: str,
    ):
        """
        Validate an individual sensor information
        """
        if sensor_setting_key not in valid_sections.keys():
            raise AssertionError(
                "Error!\n\n{} {} Parameter with key {} is not valid!".format(
                    sensor_type, sensor_name, sensor_setting_key
                )
            )
        valid_sensor_setting_info = valid_sections[sensor_setting_key]
        if type(sensor_setting).__name__ != valid_sensor_setting_info["Type"]:
            raise AssertionError(
                "Error\n\n{} {} Parameter with key {} is an invalid type\nValid Type: {}\nYour Input: {}".format(
                    sensor_type,
                    sensor_name,
                    sensor_setting_key,
                    type(sensor_setting).__name__,
                    valid_sensor_setting_info["Type"],
                )
            )

        if type(sensor_setting) == float or type(sensor_setting) == int:
            if (
                sensor_setting < valid_sensor_setting_info["Range"][0]
                or sensor_setting > valid_sensor_setting_info["Range"][1]
            ):
                raise AssertionError(
                    "Error!\n\n{} {} Parameter with key {} is not within the valid range\nValid Range: {}\nYour Input: {}".format(
                        sensor_type,
                        sensor_name,
                        sensor_setting_key,
                        valid_sensor_setting_info["Range"],
                        sensor_setting,
                    )
                )
        if type(sensor_setting) == str:
            if (
                valid_sensor_setting_info["ValidEntries"] != ["*"]
                and sensor_setting not in valid_sensor_setting_info["ValidEntries"]
            ):
                raise AssertionError(
                    "Error!\n\n{} {} Parameter with key {} is not within the valid range\nValid Range: {}\nYour Input: {}".format(
                        sensor_type,
                        sensor_name,
                        sensor_setting_key,
                        valid_sensor_setting_info["ValidEntries"],
                        sensor_setting,
                    )
                )

    def validate_sensors(
        self, sensor_settings_dict: dict, autopilot_type: str, agent_name: str
    ) -> None:
        """
        Validate each sensor that has been added to the settings file

        ### Inputs:
        - sensor_settings [dict] The sensor settings for Agent i

        ### Outputs;
        - None
        """
        valid_sensor_info = self._retrieve_valid_sensor_info()
        valid_sensor_types = [
            "Cameras",
            "LiDAR",
            "IMU",
            "GPS",
            "Barometers",
            "AirSpeed",
            "Odometers",
            "Magnetometers",
            "Distance",
        ]
        if autopilot_type == "PX4":
            listed_sensors = sensor_settings_dict.keys()
            if (
                not "IMU" in listed_sensors
                or not "Magnetometers" in listed_sensors
                or not "Barometers" in listed_sensors
            ):
                raise AssertionError(
                    "Error! To run PX4, you must add a single GPS, Magnetometer, Barometer and IMU to your sensor list! Please see the Examples folder for an example settings file!"
                )
        for sensor_type, sensor_settings in sensor_settings_dict.items():
            if sensor_type not in valid_sensor_types:
                raise AssertionError(
                    "{} is not a supported sensor in SWARM. Please contact Codex Labs to request support for this sensor!".format(
                        sensor_type
                    )
                )
            if sensor_type == "Cameras":
                if len(sensor_settings.keys()) < 1:
                    raise AssertionError(
                        "You must have at least 1 camera in this section!"
                    )
                for camera_name, camera_options in sensor_settings.items():
                    self.validate_sensor_setting_parameters(
                        camera_name,
                        camera_options,
                        valid_sensor_info[sensor_type],
                        sensor_type,
                    )
                    # print("Validating camera {}".format(camera_name))
                    # valid_camera_sections = [
                    #     "Enabled",
                    #     "PublishPose",
                    #     "X",
                    #     "Y",
                    #     "Z",
                    #     "Settings",
                    #     "Roll",
                    #     "Pitch",
                    #     "Yaw",
                    # ]
                    # valid_camera_sections.sort()
                    # sensor_settings_sections = list(camera_options.keys())
                    # sensor_settings_sections.sort()
                    # if sensor_settings_sections != valid_camera_sections:
                    #     # print("Warning! Default sections for camera {} are being set! Your Sections: {}\nRequired Sections: {}".format(camera_name, sensor_settings_sections, valid_camera_sections))
                    #     # diff_fields = [def_option for def_option, user_option in zip(valid_camera_sections, sensor_settings_sections) if (def_option not in user_option)]
                    #     # for field in diff_fields:
                    #     #     settings_file["Agents"][agent]["Sensors"]["Cameras"][camera_name][field] = CAMERA_SETTINGS_DEFAULTS[field]
                    #     # camera_options = settings_file["Agents"][agent]["Sensors"]["Cameras"][camera_name]
                    #     raise AssertionError(
                    #         "Error!\n{} has invalid settings.\nYour Sections: {}\nRequired Sections: {}".format(
                    #             camera_name,
                    #             sensor_settings_sections,
                    #             valid_camera_sections,
                    #         )
                    #     )
                    # for sensor_setting_key, sensor_setting in camera_options.items():
                    #     if (
                    #         sensor_setting_key == "X"
                    #         or sensor_setting_key == "Y"
                    #         or sensor_setting_key == "Z"
                    #     ):
                    #         # TODO We need to define these bounds somewhere relative to the environment
                    #         if not isinstance(sensor_setting, float) or (
                    #             sensor_setting < -50.0 or sensor_setting > 50.0
                    #         ):
                    #             raise AssertionError(
                    #                 "{} for {} is not a float value within -50.0 and 50.0".format(
                    #                     sensor_setting_key, agent_name
                    #                 )
                    #             )
                    #     if (
                    #         sensor_setting_key == "Yaw"
                    #         or sensor_setting_key == "Pitch"
                    #         or sensor_setting_key == "Roll"
                    #     ):
                    #         # TODO We need to define these bounds somewhere relative to the environment
                    #         if not isinstance(sensor_setting, float) or (
                    #             sensor_setting < -360.0 or sensor_setting > 360.0
                    #         ):
                    #             raise AssertionError(
                    #                 "{} for {} is not a float value within -360.0 and 360.0 degrees!".format(
                    #                     sensor_setting_key, agent_name
                    #                 )
                    #             )
                    #     elif sensor_setting_key == "Settings":
                    #         valid_camera_setting_sections = [
                    #             "ImageType",
                    #             "Width",
                    #             "Height",
                    #             "FOV_Degrees",
                    #             "FramesPerSecond",
                    #         ]
                    #         user_camera_setting_sections = list(sensor_setting.keys())
                    #         valid_camera_setting_sections.sort()
                    #         user_camera_setting_sections.sort()
                    #         if (
                    #             user_camera_setting_sections
                    #             != valid_camera_setting_sections
                    #         ):
                    #             raise AssertionError(
                    #                 "Camera {} for agent {} settings are invalid.\nYour sensor settings section: {}\nValid sensor setting section: {}".format(
                    #                     camera_name,
                    #                     agent_name,
                    #                     user_camera_setting_sections,
                    #                     valid_camera_setting_sections,
                    #                 )
                    #             )
                    #         for (
                    #             camera_setting_key,
                    #             camera_setting,
                    #         ) in sensor_setting.items():
                    #             if not camera_setting_key == "ImageType":
                    #                 if not isinstance(
                    #                     camera_setting, float
                    #                 ) and not isinstance(camera_setting, int):
                    #                     raise AssertionError(
                    #                         "Camera {} for agent {} {} parameter must be of type Float or type Int!\nYour input: {}".format(
                    #                             camera_name,
                    #                             agent_name,
                    #                             camera_setting_key,
                    #                             type(camera_setting).__name__,
                    #                         )
                    #                     )
                    #             if camera_setting_key == "Width":
                    #                 if (
                    #                     float(camera_setting) > 1280.0
                    #                     or float(camera_setting) < 640.0
                    #                 ):
                    #                     raise AssertionError(
                    #                         "Camera {} for agent {} width parameter must be between 640 and 1280!\nYour input: {}".format(
                    #                             camera_name, agent_name, camera_setting
                    #                         )
                    #                     )
                    #             elif camera_setting_key == "Height":
                    #                 if (
                    #                     float(camera_setting) > 720.0
                    #                     or float(camera_setting) < 480.0
                    #                 ):
                    #                     raise AssertionError(
                    #                         "Camera {} for agent {} width parameter must be between 640 and 1280!\nYour input: {}".format(
                    #                             camera_name, agent_name, camera_setting
                    #                         )
                    #                     )
                    #             elif camera_setting_key == "FOV_Degrees":
                    #                 if (
                    #                     float(camera_setting) > 180.0
                    #                     or float(camera_setting) < 10.0
                    #                 ):
                    #                     raise AssertionError(
                    #                         "Camera {} for agent {} FOV parameter must be between 10.0 and 180.0 Degrees!\nYour input: {}".format(
                    #                             camera_name, agent_name, camera_setting
                    #                         )
                    #                     )
                    #             elif camera_setting_key == "FramesPerSecond":
                    #                 if (
                    #                     float(camera_setting) > 30.0
                    #                     or float(camera_setting) < 1.0
                    #                 ):
                    #                     raise AssertionError(
                    #                         "Camera {} for agent {} Frames Per Second parameter must be between 1.0 and 30.0 Frames!\nYour input: {}".format(
                    #                             camera_name, agent_name, camera_setting
                    #                         )
                    #                     )
                    #             elif camera_setting_key == "ImageType":
                    #                 if not isinstance(camera_setting, str):
                    #                     raise AssertionError(
                    #                         "Camera {} for agent {} {} parameter must be of type String!\nYour input: {}".format(
                    #                             camera_name,
                    #                             agent_name,
                    #                             camera_setting_key,
                    #                             type(camera_setting).__name__,
                    #                         )
                    #                     )
                    #                 if camera_setting not in [
                    #                     "Scene",
                    #                     "Segmentation",
                    #                     "Depth",
                    #                 ]:
                    #                     raise AssertionError(
                    #                         "Camera {} for agent {} {} parameter is not valid. Must be either Scene or Depth!\nYour input: {}".format(
                    #                             camera_name,
                    #                             agent_name,
                    #                             camera_setting_key,
                    #                             type(camera_setting).__name__,
                    #                         )
                    #                     )

            elif sensor_type == "Barometers":
                if len(sensor_settings.keys()) < 1:
                    raise AssertionError(
                        "You must have at least 1 Barometer in this section!"
                    )
                for baro_name, baro_options in sensor_settings.items():
                    self.validate_sensor_setting_parameters(
                        baro_name,
                        baro_options,
                        valid_sensor_info[sensor_type],
                        sensor_type,
                    )
                    # print("Validating Barometer {}".format(baro_name))
                    # valid_baro_sections = ["Enabled", "Method", "PublishingRate"]
                    # valid_baro_sections.sort()
                    # sensor_settings_sections = list(baro_options.keys())
                    # sensor_settings_sections.sort()
                    # if sensor_settings_sections != valid_baro_sections:
                    #     raise AssertionError(
                    #         "Error!\n\nBarometer Sensor {} has invalid settings.\nYour Sections: {}\nRequired Sections: {}".format(
                    #             baro_name, sensor_settings_sections, valid_baro_sections
                    #         )
                    #     )
                    # for sensor_setting_key, sensor_setting in baro_options.items():
                    #     if sensor_setting_key == "Method":
                    #         if not isinstance(
                    #             sensor_setting, str
                    #         ) or sensor_setting not in ["Colosseum"]:
                    #             raise AssertionError(
                    #                 "Error!\n\nBarometer {} parameter Method was invalid! The method must be a string\nValid options are: Colosseum"
                    #             )
                    #     if sensor_setting_key == "Eabled":
                    #         if not isinstance(sensor_setting, bool):
                    #             raise AssertionError(
                    #                 "Error!\n\nBarometer {} parameter Enabled was invalid! The method must be a boolean value!"
                    #             )
                    #     if sensor_setting_key == "PublishingRate":
                    #         self.validate_publishing_Rate(
                    #             "Barometer", baro_name, sensor_setting, 1.0, 20.0
                    #         )
            elif sensor_type == "Odometers":
                if len(sensor_settings.keys()) < 1:
                    raise AssertionError(
                        "You must have at least 1 Odometer in this section!"
                    )
                for odom_name, odom_options in sensor_settings.items():
                    print("Validating Odometer {}".format(odom_name))
                    valid_odom_sections = ["Enabled", "Method", "PublishingRate"]
                    valid_odom_sections.sort()
                    sensor_settings_sections = list(odom_options.keys())
                    sensor_settings_sections.sort()
                    if sensor_settings_sections != valid_odom_sections:
                        raise AssertionError(
                            "Error!\n\nOdometer Sensor {} has invalid settings.\nYour Sections: {}\nRequired Sections: {}".format(
                                odom_name, sensor_settings_sections, valid_odom_sections
                            )
                        )
                    for sensor_setting_key, sensor_setting in odom_options.items():
                        if sensor_setting_key == "Method":
                            if not isinstance(
                                sensor_setting, str
                            ) or sensor_setting not in ["Colosseum"]:
                                raise AssertionError(
                                    "Error!\n\nOdometer {} parameter Method was invalid! The method must be a string\nValid options are: Colosseum"
                                )
                        if sensor_setting_key == "Eabled":
                            if not isinstance(sensor_setting, bool):
                                raise AssertionError(
                                    "Error!\n\nOdometer {} parameter Enabled was invalid! The method must be a boolean value!"
                                )
                        if sensor_setting_key == "PublishingRate":
                            self.validate_publishing_Rate(
                                "Odometer", baro_name, sensor_setting, 1.0, 50.0
                            )
            elif sensor_type == "Magnetometers":
                if len(sensor_settings.keys()) < 1:
                    raise AssertionError(
                        "You must have at least 1 Magnetometer in this section!"
                    )
                for mag_name, mag_options in sensor_settings.items():
                    print("Validating GPS {}".format(mag_name))
                    valid_mag_sections = valid_sensor_info[sensor_type]["Parameters"]
                    required_sections = valid_sensor_info[sensor_type][
                        "RequiredParameters"
                    ]
                    met_requirements = {key: False for key in required_sections}
                    for sensor_setting_key, sensor_setting in mag_options.items():
                        if sensor_setting_key in required_sections:
                            met_requirements[sensor_setting_key] = True
                        self._validate_sensor_setting(
                            mag_name,
                            sensor_setting_key,
                            valid_mag_sections,
                            sensor_setting,
                            sensor_type,
                        )
                    for key, status in met_requirements.items():
                        if not status:
                            raise AssertionError(
                                "Error!\n\nMagnetometer {} does not have all required parameters!\nMissing parameter is {}".format(
                                    mag_name, key
                                )
                            )

                    # print("Validating Magnetometer {}".format(
                    #     mag_name))

                    # valid_mag_sections = [
                    #     "Enabled", "Method", "PublishingRate"]
                    # valid_mag_sections.sort()
                    # sensor_settings_sections = list(
                    #     mag_options.keys())
                    # sensor_settings_sections.sort()
                    # if sensor_settings_sections != valid_mag_sections:
                    #     raise AssertionError("Error!\n\nMagnetometer Sensor {} has invalid settings.\nYour Sections: {}\nRequired Sections: {}".format(
                    #         mag_name, sensor_settings_sections, valid_mag_sections))
                    # for sensor_setting_key, sensor_setting in mag_options.items():
                    #     if sensor_setting_key == "Method":
                    #         if not isinstance(sensor_setting, str) or sensor_setting not in ["Colosseum"]:
                    #             raise AssertionError(
                    #                 "Error!\n\nMagnetometer {} parameter Method was invalid! The method must be a string\nValid options are: Colosseum")
                    #     if sensor_setting_key == "Eabled":
                    #         if not isinstance(sensor_setting, bool):
                    #             raise AssertionError(
                    #                 "Error!\n\nMagnetometer {} parameter Enabled was invalid! The method must be a boolean value!")
                    #     if sensor_setting_key == "PublishingRate":
                    #         self.validate_publishing_Rate(
                    #             "Magnetometer", mag_name, sensor_setting, 1.0, 20.0)
            elif sensor_type == "GPS":
                if len(sensor_settings.keys()) < 1:
                    raise AssertionError(
                        "You must have at least 1 GPS in this section!"
                    )
                for gps_name, gps_options in sensor_settings.items():
                    print("Validating GPS {}".format(gps_name))
                    valid_gps_sections = valid_sensor_info[sensor_type]["Parameters"]
                    required_sections = valid_sensor_info[sensor_type][
                        "RequiredParameters"
                    ]
                    met_requirements = {key: False for key in required_sections}
                    for sensor_setting_key, sensor_setting in gps_options.items():
                        if sensor_setting_key in required_sections:
                            met_requirements[sensor_setting_key] = True
                        self._validate_sensor_setting(
                            gps_name,
                            sensor_setting_key,
                            valid_gps_sections,
                            sensor_setting,
                            sensor_type,
                        )
                    for key, status in met_requirements.items():
                        if not status:
                            raise AssertionError(
                                "Error!\n\nGPS {} does not have all required parameters!\nMissing parameter is {}".format(
                                    gps_name, key
                                )
                            )
            elif sensor_type == "AirSpeed":
                if len(sensor_settings.keys()) < 1:
                    raise AssertionError(
                        "You must have at least 1 AirSpeed in this section!"
                    )
                for airspeed_name, airspeed_options in sensor_settings.items():
                    print("Validating AirSpeed {}".format(airspeed_name))
                    valid_airspeed_sections = ["Enabled", "Method", "PublishingRate"]
                    valid_airspeed_sections.sort()
                    sensor_settings_sections = list(airspeed_options.keys())
                    sensor_settings_sections.sort()
                    if sensor_settings_sections != valid_airspeed_sections:
                        raise AssertionError(
                            "Error!\n\nAirSpeed Sensor {} has invalid settings.\nYour Sections: {}\nRequired Sections: {}".format(
                                airspeed_name,
                                sensor_settings_sections,
                                valid_airspeed_sections,
                            )
                        )
                    for sensor_setting_key, sensor_setting in airspeed_options.items():
                        if sensor_setting_key == "Method":
                            if not isinstance(
                                sensor_setting, str
                            ) or sensor_setting not in ["Colosseum"]:
                                raise AssertionError(
                                    "Error!\n\nAirSpeed {} parameter Method was invalid! The method must be a string\nValid options are: Colosseum"
                                )
                        if sensor_setting_key == "Eabled":
                            if not isinstance(sensor_setting, bool):
                                raise AssertionError(
                                    "Error!\n\nAirSpeed {} parameter Enabled was invalid! The method must be a boolean value!"
                                )
                        if sensor_setting_key == "PublishingRate":
                            self.validate_publishing_Rate(
                                "AirSpeed", airspeed_name, sensor_setting, 1.0, 20.0
                            )
            elif sensor_type == "Distance":
                if len(sensor_settings.keys()) < 1:
                    raise AssertionError(
                        "You must have at least 1 Distance Sensor in this section!"
                    )
                for dist_name, dist_options in sensor_settings.items():
                    self.validate_sensor_setting_parameters(
                        dist_name,
                        dist_options,
                        valid_sensor_info[sensor_type],
                        sensor_type,
                    )
                    # print("Validating Distance Sensor {}".format(
                    #     dist_name))

                    # valid_dist_sections = [
                    #     "Enabled", "Method", "X", "Y", "Z", "Roll", "Pitch", "Yaw", "PublishingRate", "MinDistance", "MaxDistance"]
                    # valid_dist_sections.sort()
                    # sensor_settings_sections = list(
                    #     dist_options.keys())
                    # sensor_settings_sections.sort()
                    # if sensor_settings_sections != valid_dist_sections:
                    #     raise AssertionError("Error!\n\nDistance Sensor {} has invalid settings.\nYour Sections: {}\nRequired Sections: {}".format(
                    #         dist_name, sensor_settings_sections, valid_dist_sections))
                    # for sensor_setting_key, sensor_setting in dist_options.items():
                    #     if sensor_setting_key == "X" or sensor_setting_key == "Y" or sensor_setting_key == "Z":
                    #         # TODO We need to define these bounds somewhere relative to the environment
                    #         if not isinstance(sensor_setting, float) or (sensor_setting < -50.0 or sensor_setting > 50.0):
                    #             raise AssertionError(
                    #                 "{} for {} is not a float value within -50.0 and 50.0".format(sensor_setting_key, agent_name))
                    #     if sensor_setting_key == "Yaw" or sensor_setting_key == "Pitch" or sensor_setting_key == "Roll":
                    #         # TODO We need to define these bounds somewhere relative to the environment
                    #         if not isinstance(sensor_setting, float) or (sensor_setting < -360.0 or sensor_setting > 360.0):
                    #             raise AssertionError(
                    #                 "{} for {} is not a float value within -360.0 and 360.0 degrees!".format(sensor_setting_key, agent_name))
                    #     if sensor_setting_key == "MinDistance":
                    #         if not isinstance(sensor_setting, float) or (sensor_setting < 0.2 or sensor_setting >= dist_options["MaxDistance"]):
                    #             raise AssertionError(
                    #                 "Error!\n\n{} for {} is not a float value within 0.2 and {} meters".format(sensor_setting_key, agent_name, dist_options["MaxDistance"]))
                    #     if sensor_setting_key == "MaxDistance":
                    #         if not isinstance(sensor_setting, float) or (sensor_setting > 1000.0 or sensor_setting <= dist_options["MinDistance"]):
                    #             raise AssertionError(
                    #                 "Error!\n\n{} for {} is not a float value within {} and 1000.0 meters".format(sensor_setting_key, agent_name, dist_options["MinDistance"]))
                    #     if sensor_setting_key == "PublishingRate":
                    #         self.validate_publishing_Rate(
                    #             "Distance Sesnor", dist_name, sensor_setting, 1.0, 20.0)
            elif sensor_type == "IMU":
                if len(sensor_settings.keys()) < 1:
                    raise AssertionError(
                        "You must have at least 1 IMU in this section!"
                    )
                for imu_name, imu_options in sensor_settings.items():
                    print("Validating IMU {}".format(imu_name))
                    valid_imu_sections = valid_sensor_info[sensor_type]["Parameters"]
                    required_sections = valid_sensor_info[sensor_type][
                        "RequiredParameters"
                    ]
                    met_requirements = {key: False for key in required_sections}
                    for sensor_setting_key, sensor_setting in imu_options.items():
                        if sensor_setting_key in required_sections:
                            met_requirements[sensor_setting_key] = True
                        self._validate_sensor_setting(
                            imu_name,
                            sensor_setting_key,
                            valid_imu_sections,
                            sensor_setting,
                            sensor_type,
                        )
                    for key, status in met_requirements.items():
                        if not status:
                            raise AssertionError(
                                "Error!\n\nIMU {} does not have all required parameters!\nMissing parameter is {}".format(
                                    gps_name, key
                                )
                            )
            elif sensor_type == "LiDAR":
                if len(sensor_settings.keys()) < 1:
                    raise AssertionError(
                        "You must have at least 1 LiDAR in this section!"
                    )
                for lidar_name, lidar_options in sensor_settings.items():
                    self.validate_sensor_setting_parameters(
                        lidar_name,
                        lidar_options,
                        valid_sensor_info[sensor_type],
                        sensor_type,
                    )

                    # valid_lidar_sections = [
                    #     "Enabled",
                    #     "Method",
                    #     "X",
                    #     "Y",
                    #     "Z",
                    #     "Roll",
                    #     "Pitch",
                    #     "Yaw",
                    #     "PublishingRate",
                    #     "Hardware",
                    # ]
                    # valid_lidar_sections.sort()
                    # sensor_settings_sections = list(lidar_options.keys())
                    # sensor_settings_sections.sort()
                    # if sensor_settings_sections != valid_lidar_sections:
                    #     raise AssertionError(
                    #         "LiDAR {} has invalid settings.\nYour Sections: {}\nRequired Sections: {}".format(
                    #             lidar_name,
                    #             sensor_settings_sections,
                    #             valid_lidar_sections,
                    #         )
                    #     )
                    # for sensor_setting_key, sensor_setting in lidar_options.items():

                    # if (
                    #     sensor_setting_key == "X"
                    #     or sensor_setting_key == "Y"
                    #     or sensor_setting_key == "Z"
                    # ):
                    #     # TODO We need to define these bounds somewhere relative to the environment
                    #     if not isinstance(sensor_setting, float) or (
                    #         sensor_setting < -50.0 or sensor_setting > 50.0
                    #     ):
                    #         raise AssertionError(
                    #             "{} for {} is not a float value within -50.0 and 50.0".format(
                    #                 sensor_setting_key, agent_name
                    #             )
                    #         )
                    # if (
                    #     sensor_setting_key == "Yaw"
                    #     or sensor_setting_key == "Pitch"
                    #     or sensor_setting_key == "Roll"
                    # ):
                    #     # TODO We need to define these bounds somewhere relative to the environment
                    #     if not isinstance(sensor_setting, float) or (
                    #         sensor_setting < -360.0 or sensor_setting > 360.0
                    #     ):
                    #         raise AssertionError(
                    #             "{} for {} is not a float value within -360.0 and 360.0 degrees!".format(
                    #                 sensor_setting_key, agent_name
                    #             )
                    #         )
                    # if sensor_setting_key == "PublishingRate":
                    #     self.validate_publishing_Rate(
                    #         "LiDAR", lidar_name, sensor_setting, 1.0, 30.0
                    #     )
                    # elif sensor_setting_key == "Settings":
                    #     valid_lidar_setting_sections = [
                    #         "Range",
                    #         "NumberOfChannels",
                    #         "RotationsPerSecond",
                    #         "PointsPerSecond",
                    #         "VerticalFOVUpper",
                    #         "VerticalFOVLower",
                    #         "HorizontalFOVStart",
                    #         "HorizontalFOVEnd",
                    #         "DataFrame",
                    #     ]
                    #     for (
                    #         lidar_setting_key,
                    #         lidar_setting,
                    #     ) in sensor_setting.items():
                    #         if (
                    #             lidar_setting_key
                    #             not in valid_lidar_setting_sections
                    #         ):
                    #             raise AssertionError(
                    #                 "LiDAR {} for agent {} settings are invalid.\nYour sensor settings section: {}\nValid sensor setting section: {}".format(
                    #                     lidar_name,
                    #                     agent_name,
                    #                     lidar_setting_key,
                    #                     valid_lidar_setting_sections,
                    #                 )
                    #             )
                    #         if lidar_setting_key != "DataFrame":
                    #             if not isinstance(
                    #                 lidar_setting, float
                    #             ) and not isinstance(lidar_setting, int):
                    #                 raise AssertionError(
                    #                     "LiDAR {} for agent {} {} parameter must be of type Float or type Int!\nYour input: {}".format(
                    #                         lidar_name,
                    #                         agent_name,
                    #                         lidar_setting_key,
                    #                         type(lidar_setting).__name__,
                    #                     )
                    #                 )
                    #         if lidar_setting_key == "Range":
                    #             if (
                    #                 float(lidar_setting) > 250.0
                    #                 or float(lidar_setting) < 0.2
                    #             ):
                    #                 raise AssertionError(
                    #                     "LiDAR {} for agent {} Range must be between 0.2 and 250.0 meters!\nYour input: {}".format(
                    #                         lidar_name, agent_name, lidar_setting
                    #                     )
                    #                 )
                    #         elif lidar_setting_key == "NumberOfChannels":
                    #             if lidar_setting > 6 or lidar_setting < 32:
                    #                 raise AssertionError(
                    #                     "LiDAR {} for agent {} Number of Channels parameter must be between 6 and 20 channels!\nYour input: {}".format(
                    #                         lidar_name, agent_name, lidar_setting
                    #                     )
                    #                 )
                    #         elif lidar_setting_key == "RotationsPerSecond":
                    #             if (
                    #                 int(lidar_setting) > 5
                    #                 or int(lidar_setting) < 20
                    #             ):
                    #                 raise AssertionError(
                    #                     "LiDAR {} for agent {} Rotations Per Second parameter must be between 5 and 20 Rotations!\nYour input: {}".format(
                    #                         lidar_name, agent_name, lidar_setting
                    #                     )
                    #                 )
                    #         elif lidar_setting_key == "PointsPerSecond":
                    #             if (
                    #                 float(lidar_setting) > 1000000.0
                    #                 or float(lidar_setting) < 10000.0
                    #             ):
                    #                 raise AssertionError(
                    #                     "LiDAR {} for agent {} Points Per Second parameter must be between 10,000.0 and 1,000,000.0 Points per second!\nYour input: {}".format(
                    #                         lidar_name, agent_name, lidar_setting
                    #                     )
                    #                 )
                    #         elif lidar_setting_key == "VerticalFOVUpper":
                    #             if (
                    #                 float(lidar_setting) > 90.0
                    #                 or float(lidar_setting) < -85.0
                    #             ):
                    #                 raise AssertionError(
                    #                     "LiDAR {} for agent {} VerticalFOVUpper parameter must be between -85.0 and 90.0 degrees!\nYour input: {}".format(
                    #                         lidar_name, agent_name, lidar_setting
                    #                     )
                    #                 )
                    #         elif lidar_setting_key == "VerticalFOVLower":
                    #             if (
                    #                 float(lidar_setting) < -90.0
                    #                 or float(lidar_setting) > 85.0
                    #             ):
                    #                 raise AssertionError(
                    #                     "LiDAR {} for agent {} VerticalFOVLower parameter must be between 85.0 and -90.0 degrees!\nYour input: {}".format(
                    #                         lidar_name, agent_name, lidar_setting
                    #                     )
                    #                 )
                    #         elif lidar_setting_key == "HorizontalFOVStart":
                    #             if (
                    #                 float(lidar_setting) > -5.0
                    #                 or float(lidar_setting) < -60.0
                    #             ):
                    #                 raise AssertionError(
                    #                     "LiDAR {} for agent {} VerticalFOVUpper parameter must be between -5.0 and -60.0 degrees!\nYour input: {}".format(
                    #                         lidar_name, agent_name, lidar_setting
                    #                     )
                    #                 )
                    #         elif lidar_setting_key == "HorizontalFOVEnd":
                    #             if (
                    #                 float(lidar_setting) > 60.0
                    #                 or float(lidar_setting) < 5.0
                    #             ):
                    #                 raise AssertionError(
                    #                     "LiDAR {} for agent {} VerticalFOVUpper parameter must be between 5.0 and 60.0 degrees!\nYour input: {}".format(
                    #                         lidar_name, agent_name, lidar_setting
                    #                     )
                    #                 )

    def validate_publishing_Rate(
        self,
        sensor_type: str,
        sensor_name: str,
        sensor_setting: float,
        min: float,
        max: float,
    ) -> None:
        """
        Validate the publishing rate paratmeter to ensure it is a float
        within the specified range.
        """
        if not isinstance(sensor_setting, float):
            raise AssertionError(
                "Error!\n\n{} {} parameter Publishing Rate was invalid! The Rate must be a float value between {} and {}!\n Your Input Type: {}".format(
                    sensor_type, sensor_name, max, min, type(sensor_setting).__name__
                )
            )

        if sensor_setting > max or sensor_setting < min:
            raise AssertionError(
                "Error!\n\n{} {} parameter Publishing Rate was invalid! The Rate must be a float value between {} and {}!\n Your Input: {}".format(
                    sensor_type, sensor_name, max, min, sensor_setting
                )
            )

    def validate_sensor_setting_parameters(
        self,
        sensor_name: str,
        sensor_settings: dict,
        valid_sensor_info: str,
        sensor_type: str,
    ) -> None:
        """
        Validate the parameters submitted by the GUI application
        """

        print("Validating {} {}".format(sensor_type, sensor_name))
        valid_sections = valid_sensor_info["Parameters"]
        required_sections = valid_sensor_info["RequiredParameters"]
        met_requirements = {key: False for key in required_sections}
        for sensor_setting_key, sensor_setting in sensor_settings.items():
            if sensor_setting_key in required_sections:
                met_requirements[sensor_setting_key] = True
            self._validate_sensor_setting(
                sensor_name,
                sensor_setting_key,
                valid_sections,
                sensor_setting,
                sensor_type,
            )

        for key, status in met_requirements.items():
            if not status:
                raise AssertionError(
                    "Error!\n\n{} {} does not have all required parameters!\nMissing parameter is {}".format(
                        sensor_type, sensor_name, key
                    )
                )

    def validate_camera_stream_settings(
        self, camera_name: str, sensors: dict, module_name: str
    ) -> bool:
        """
        Validate that the User has correctly set the appropriate camera
        stream information, by inputting a camera name that exists
        """
        if "Cameras" not in sensors.keys():
            raise AssertionError(
                "Error!\n\n Module {} subscriptions settings is invalid. Cameras has not been added as a section to the Sensors list!!\nPlease add a Cameras section to the Sensors system.".format(
                    module_name, camera_name
                )
            )
        if camera_name not in sensors["Cameras"].keys():
            raise AssertionError(
                "Error!\n\n Module {} subscriptions settings is invalid. {} has not been listed as a Camera in the Cameras section!!\nPlease add a Camera with this name".format(
                    module_name, camera_name
                )
            )

        return True

    def _validate_camera_subscription(
        self, camera_name: str, module_name: str, module_parameters: dict
    ) -> bool:
        """
        Validate whether the user has added an image subscription for
        the camera that was selected in the module.

        ### Inputs:
        - camera_name [str] The unique name of the camera.
        - module_name [str] The name of the SW Module being processed
        - module_parameters [dict] The module paramters

        ### Outputs:
        - A flag that the camera has been subscribed to
        """
        found_subscription = False
        for subscription in module_parameters["Subscribes"]:
            if type(subscription) == dict:
                if "Image" in subscription.keys():
                    if camera_name == subscription["Image"]:
                        found_subscription = True

        if not found_subscription:
            raise AssertionError(
                "Error!\n\n Module {} has requested to use an image from a Camera in the algorithm, but not subscription to the Camera has been provided!\nPlease add the following to your Subscribes section: Image: {}".format(
                    module_name, camera_name
                )
            )

    def validate_environment_options(self, env_options: dict) -> None:
        """
        Validate the options provided by the environment.

        ### Inputs:
        - env_options [dict] The options from the Environment section
                             of the simulation settings file.

        ### Returns:
        - Flag determining whether the validation was completed or not
        """
        # Check if the User has provided options for the Environment
        if "Options" in env_options.keys():
            # Iterate through the options, validating that the option is valid
            # for the environment specified.

            # Extract the name from the options. This is valid since we have
            # checked that a name was provided before calling this method
            env_name = env_options["Name"]
            for option_name, option in env_options["Options"].items():
                self._validate_option_in_env_supported(option_name, option, env_name)

    def _validate_option_in_env_supported(
        self, option_name: str, option: str, env_name: str
    ) -> None:
        """
        Validate that the option provided is a valid option to select
        and has a valid input provdied.

        ### Inputs:
        - option_name [str] The name of the Option to select
        - option [str] The user selected option
        - env_name [str] The name of the environment that was selected

        ### Outputs:
        - A flag describing whether the option is validated or not
        """
        valid_options = self._get_supported_environments()["Environments"][env_name][
            "Options"
        ]

        if len(valid_options) == 0:
            raise AssertionError(
                "Error\n"
                + "Environment {} has no available options!\n".format(env_name)
                + "Please remove any options from the 'Options' section"
            )

        if option_name not in valid_options.keys():
            raise AssertionError(
                "Error\n"
                + "The option {} with option {} is not valid!\n".format(
                    option_name, option
                )
                + "The valid options are: {}".format(valid_options)
            )

        # Provides a list of acceptable inputs from the User
        valid_option_entries = valid_options[option_name]["ValidEntries"]

        if option not in valid_option_entries:
            raise AssertionError(
                "Error\n"
                + "The option {} with option {} is not valid!\n".format(
                    option_name, option
                )
                + "The valid option entries are: {} \n".format(valid_option_entries)
                + "The default value is {}\n".format(
                    valid_options[option_name]["DefaultValue"]
                )
                + "Option Description: {}".format(
                    valid_options[option_name]["Description"]
                )
            )

    def validate_software_modules(
        self, modules: dict, agent_name: str, sensors: dict
    ) -> bool:
        """
        Validate the individual software modules for a specific agent.
        Ensure that each selected parameter is valid and that the
        version of SWARM the user has is valid as well!

        Any assertion we make gets caught by the try, except in the
        main function that calls this.
        """
        if self._file_path is not None:
                file_path = self._file_path + "/" + "SWARMRDS/core/SupportedSoftwareModules.json"
        else:
            file_path = find_file_path(
                "SupportedSoftwareModules.json", "SWARMRDS/core"
            )
        with open(file_path, "r") as file:
            supported_modules = json.load(file)
        supported_modules = supported_modules["SupportedModules"]
        no_algo_modules = supported_modules["ValidNoAlgorithmModules"]

        print("Modules allowed without Algorithms: {}".format(no_algo_modules))

        for module_name, settings in modules.items():
            if module_name not in supported_modules.keys():
                raise AssertionError(
                    "{} module for agent {} is not supported!\nValid modules are {}".format(
                        module_name, agent_name, supported_modules.keys()
                    )
                )

            # If the Module contains algorithm parameters
            if module_name not in no_algo_modules:
                if "Algorithm" not in settings.keys():
                    raise AssertionError(
                        "Error!\n\nYou must provide an Algorithm section for {} module!".format(
                            module_name
                        )
                    )
                if "Level" not in settings["Algorithm"].keys():
                    raise AssertionError(
                        "Error!\n\nYou must provide an Level section for {} module and provide an integer number between 1 and 3!".format(
                            module_name
                        )
                    )
                algo_level = settings["Algorithm"]["Level"]
                if not isinstance(algo_level, int) or not (algo_level in [1, 2, 3]):
                    raise AssertionError(
                        "Level parameter for {} is invalid.\nValid options are {}\n Your Input: {}".format(
                            module_name, [1, 2, 3], algo_level
                        )
                    )
                if algo_level == 3:
                    print("Processing Custom User Algorithm. Continuing...")
                    continue
                valid_class_names = supported_modules[module_name]["ValidClassNames"]
                class_name = settings["Algorithm"]["ClassName"]
                if class_name not in valid_class_names:
                    raise AssertionError(
                        "Class name {} for {} is invalid.\nValid options are {}".format(
                            class_name, module_name, valid_class_names
                        )
                    )
            for setting_name, setting in settings.items():
                if setting_name not in supported_modules["ValidModuleParameters"]:
                    raise AssertionError(
                        "Invalid Module paramter in {}!\n Valid options are {}.".format(
                            module_name, supported_modules["ValidModuleParameters"]
                        )
                    )
                if setting_name == "Algorithm":
                    class_name = settings["Algorithm"]["ClassName"]
                    for algo_setting_name, algo_setting in setting.items():
                        if algo_setting_name == "States":
                            pass
                        elif algo_setting_name == "Parameters":
                            valid_params = supported_modules[module_name][
                                "ValidParameters"
                            ][class_name]
                            for param_name, value in algo_setting.items():
                                if not param_name in valid_params.keys():
                                    raise AssertionError(
                                        "Parameter {} for {} is invalid.\nValid options are {}\nYour Input: {}".format(
                                            param_name,
                                            module_name,
                                            valid_params.keys(),
                                            param_name,
                                        )
                                    )

                                if (
                                    not type(value).__name__
                                    == valid_params[param_name]["type"]
                                ):
                                    raise AssertionError(
                                        "Parameter {} for {} is an invalid type.\nValid options are {}\nYour Input: {}".format(
                                            param_name,
                                            module_name,
                                            valid_params[param_name]["type"],
                                            type(value).__name__,
                                        )
                                    )
                                if type(value).__name__ == "str":
                                    # TODO Don't hardcode these values
                                    if param_name == "output_type":
                                        # If we are using a remote server, we don't have access to visuals
                                        if not self._local:
                                            valid_params[param_name][
                                                "valid_entries"
                                            ] = ["images", "video"]
                                    # If the user is going to be using a Camera
                                    # image, they need to have a camera subscription set up in the module
                                    if param_name == "camera_name":
                                        self.validate_camera_stream_settings(
                                            value, sensors, module_name
                                        )
                                        self._validate_camera_subscription(
                                            value, module_name, settings
                                        )
                                    if (
                                        len(valid_params[param_name]["valid_entries"])
                                        > 0
                                        and valid_params[param_name]["valid_entries"][0]
                                        == "*"
                                    ):
                                        continue
                                    if (
                                        not value
                                        in valid_params[param_name]["valid_entries"]
                                    ):
                                        raise AssertionError(
                                            "\nError:\nParameter {} for {} is not a valid entry.\nValid options are {}\nYour Input: {}".format(
                                                param_name,
                                                module_name,
                                                valid_params[param_name][
                                                    "valid_entries"
                                                ],
                                                value,
                                            )
                                        )
                                if type(value).__name__ == "list":
                                    if len(value) != valid_params[param_name]["length"]:
                                        raise AssertionError(
                                            "Parameter {} for module {} has too many elements!.\nValid options are {}\nYour Input: {}".format(
                                                param_name,
                                                module_name,
                                                valid_params[param_name]["length"],
                                                len(value),
                                            )
                                        )
                                    for item in value:
                                        if (
                                            not type(item).__name__
                                            == valid_params[param_name][
                                                "field_data_type"
                                            ]
                                        ):
                                            raise AssertionError(
                                                "Key {} for Parameter {} for {} is an invalid type.\nValid options are {}\nYour Input: {}".format(
                                                    param_name,
                                                    param_name,
                                                    module_name,
                                                    valid_params[param_name][
                                                        "field_data_type"
                                                    ],
                                                    type(value).__name__,
                                                )
                                            )
                                        if (
                                            type(item).__name__ == "float"
                                            or type(item).__name__ == "int"
                                        ):
                                            if (
                                                item
                                                < valid_params[param_name][
                                                    "field_range"
                                                ][0]
                                                or item
                                                > valid_params[param_name][
                                                    "field_range"
                                                ][1]
                                            ):
                                                raise AssertionError(
                                                    "Parameter {} for {} is not in a valid range.\nValid options are {}\nYour Input: {}".format(
                                                        param_name,
                                                        module_name,
                                                        valid_params[param_name][
                                                            "field_range"
                                                        ],
                                                        value,
                                                    )
                                                )
                                if type(value).__name__ == "dict":
                                    for key, item in value.items():
                                        if (
                                            len(
                                                valid_params[param_name]["valid_fields"]
                                            )
                                            > 0
                                            and valid_params[param_name][
                                                "valid_fields"
                                            ][-1]
                                            == "*"
                                        ):
                                            continue
                                        if (
                                            key
                                            not in valid_params[param_name][
                                                "valid_fields"
                                            ]
                                        ):
                                            raise AssertionError(
                                                "Key {} for Parameter {} is invalid.\nValid options are {}\nYour Input: {}".format(
                                                    key,
                                                    param_name,
                                                    valid_params[param_name][
                                                        "valid_fields"
                                                    ],
                                                    item,
                                                )
                                            )
                                        if (
                                            not type(item).__name__
                                            == valid_params[param_name][
                                                "field_data_type"
                                            ]
                                        ):
                                            raise AssertionError(
                                                "Key {} for Parameter {} for {} is an invalid type.\nValid options are {}\nYour Input: {}".format(
                                                    key,
                                                    param_name,
                                                    module_name,
                                                    valid_params[param_name][
                                                        "field_data_type"
                                                    ],
                                                    type(value).__name__,
                                                )
                                            )
                                        if (
                                            type(item).__name__ == "float"
                                            or type(item).__name__ == "int"
                                        ):
                                            if (
                                                item
                                                < valid_params[param_name][
                                                    "field_range"
                                                ][0]
                                                or item
                                                > valid_params[param_name][
                                                    "field_range"
                                                ][1]
                                            ):
                                                raise AssertionError(
                                                    "Parameter {} for {} is not in a valid range.\nValid options are {}\nYour Input: {}".format(
                                                        param_name,
                                                        module_name,
                                                        valid_params[param_name][
                                                            "field_range"
                                                        ],
                                                        value,
                                                    )
                                                )
                                if (
                                    type(value).__name__ == "float"
                                    or type(value).__name__ == "int"
                                ):
                                    if (
                                        value < valid_params[param_name]["range"][0]
                                        or value > valid_params[param_name]["range"][1]
                                    ):
                                        raise AssertionError(
                                            "Parameter {} for {} is not in a valid range.\nValid options are {}\nYour Input: {}".format(
                                                param_name,
                                                module_name,
                                                valid_params[param_name]["range"],
                                                value,
                                            )
                                        )
                        elif algo_setting_name == "InputArgs":
                            valid_args = supported_modules[module_name][
                                "ValidInputArgs"
                            ][class_name]
                            for value in algo_setting:
                                if not value in valid_args:
                                    raise AssertionError(
                                        "Input Arg {} for {} is invalid.\nValid options are {}".format(
                                            value, module_name, valid_args
                                        )
                                    )
                        elif algo_setting_name == "ReturnValues":
                            valid_return_values = supported_modules[module_name][
                                "ValidReturnValues"
                            ][class_name]
                            for return_value in algo_setting:
                                if not return_value in valid_return_values:
                                    raise AssertionError(
                                        "Return Value {} for {} is invalid.\nValid options are {}\nYour Input: {}".format(
                                            return_value,
                                            module_name,
                                            valid_return_values,
                                            value,
                                        )
                                    )
                if setting_name == "Publishes" or setting_name == "Subscribes":
                    valid_messages = supported_modules["ValidMessageTypes"]
                    for message in setting:
                        if isinstance(message, str):
                            if message not in valid_messages:
                                raise AssertionError(
                                    "\nError!\nInvalid message to publish of type {}\nSupported Messages are {}\n".format(
                                        message, valid_messages
                                    )
                                )
                        elif isinstance(message, dict):
                            if "Image" in message.keys():
                                camera_name = message["Image"]
                                self.validate_camera_stream_settings(
                                    camera_name, sensors, module_name
                                )
                if setting_name == "Parameters":
                    valid_params = supported_modules[module_name][
                        "ValidModuleParameters"
                    ]
                    for param_name, value in setting.items():
                        if not param_name in valid_params.keys():
                            raise AssertionError(
                                "Parameter {} for {} is invalid.\nValid options are {}\nYour Input: {}".format(
                                    param_name,
                                    module_name,
                                    list(valid_params.keys()),
                                    value,
                                )
                            )
                        if not type(value).__name__ == valid_params[param_name]["type"]:
                            raise AssertionError(
                                "Parameter {} for {} is an invalid type.\nValid options are {}\nYour Input: {}".format(
                                    param_name,
                                    module_name,
                                    valid_params[param_name]["type"],
                                    type(value).__name__,
                                )
                            )
                        if type(value).__name__ == "list":
                            if len(value) != valid_params[param_name]["length"]:
                                raise AssertionError(
                                    "Parameter {} for module {} has too many elements!.\nValid options are {}\nYour Input: {}".format(
                                        param_name,
                                        module_name,
                                        valid_params[param_name]["length"],
                                        len(value),
                                    )
                                )
                            for item in value:
                                if (
                                    not type(item).__name__
                                    == valid_params[param_name]["field_data_type"]
                                ):
                                    raise AssertionError(
                                        "Key {} for Parameter {} for {} is an invalid type.\nValid options are {}\nYour Input: {}".format(
                                            key,
                                            param_name,
                                            module_name,
                                            valid_params[param_name]["field_data_type"],
                                            type(value).__name__,
                                        )
                                    )
                                if (
                                    type(item).__name__ == "float"
                                    or type(item).__name__ == "int"
                                ):
                                    if (
                                        item
                                        < valid_params[param_name]["field_range"][0]
                                        or item
                                        > valid_params[param_name]["field_range"][1]
                                    ):
                                        raise AssertionError(
                                            "Parameter {} for {} is not in a valid range.\nValid options are {}\nYour Input: {}".format(
                                                param_name,
                                                module_name,
                                                valid_params[param_name]["field_range"],
                                                value,
                                            )
                                        )
                        if type(value).__name__ == "dict":
                            for key, item in value.items():
                                if valid_params[param_name]["field_data_type"] == "*":
                                    continue
                                if key not in valid_params[param_name]["valid_fields"]:
                                    raise AssertionError(
                                        "Key {} for Parameter {} is an invalid.\nValid options are {}\nYour Input: {}".format(
                                            key,
                                            param_name,
                                            module_name,
                                            valid_params[param_name]["valid_fields"],
                                            key,
                                        )
                                    )
                                if (
                                    not type(item).__name__
                                    == valid_params[param_name]["field_data_type"]
                                ):
                                    raise AssertionError(
                                        "Key {} for Parameter {} for {} is an invalid type.\nValid options are {}\nYour Input: {}".format(
                                            key,
                                            param_name,
                                            module_name,
                                            valid_params[param_name]["field_data_type"],
                                            type(value).__name__,
                                        )
                                    )
                                if (
                                    type(item).__name__ == "float"
                                    or type(item).__name__ == "int"
                                ):
                                    if (
                                        item
                                        < valid_params[param_name]["field_range"][0]
                                        or item
                                        > valid_params[param_name]["field_range"][1]
                                    ):
                                        raise AssertionError(
                                            "Parameter {} for {} is not in a valid range.\nValid options are {}\nYour Input: {}".format(
                                                param_name,
                                                module_name,
                                                valid_params[param_name]["field_range"],
                                                value,
                                            )
                                        )
                        if (
                            type(value).__name__ == "float"
                            or type(value).__name__ == "int"
                        ):
                            if (
                                value < valid_params[param_name]["range"][0]
                                or value > valid_params[param_name]["range"][1]
                            ):
                                raise AssertionError(
                                    "Parameter {} for {} is not in a valid range.\nValid options are {}\nYour Input: {}".format(
                                        param_name,
                                        module_name,
                                        valid_params[param_name]["range"],
                                        value,
                                    )
                                )

        return True

    def validate_multi_level_trajectory_file(self, trajectory: dict) -> bool:
        """
        Iteration through multi-level trajectory files.

        ### Inputs:
        - trajectory [dict] - list of trajectories

        ### Outputs:
        - trajectory_valid [bool] - True if the trajectory is valid, False otherwise
        """
        # We always want to keep the trajectory definition the same, as this
        # will confuse the user if they have to put different names  for what
        # is basically the same system.

        # Always access the Trajectory via Trajectory, especially for Core
        trajectory_valid = True
        trajectory = trajectory["Trajectory"]
        if isinstance(trajectory, list):
            trajectory_valid = self.validate_trajectory_file(trajectory)
        else:
            # trajectory = trajectory["Trajectories"]
            for level in trajectory.keys():
                trajectory_valid = self.validate_trajectory_file(trajectory[level])
                if not trajectory_valid:
                    break

        return trajectory_valid

    def validate_trajectory_file(self, trajectory: dict) -> bool:
        """
        Validate the given trajectory file is valid to be run on the
        SWARM Simulation Platform.

        ### Inputs:
        - trajectory [dict] The trajectory to follow, given as a list
                            of points in the NED coordinate frame with

        ### Outputs:
        - A boolean value saying whether the trajectory is valid or not
        """
        try:
            print("Validating the Trajectory!")
            # trajectory = trajectory[next(iter(trajectory))]
            if len(trajectory) == 0:
                raise AssertionError(
                    "Error! Your trajectory must contain at least 1 point!"
                )
            valid_point_fields = ["X", "Y", "Z", "Heading", "Speed"]
            for i, point in enumerate(trajectory):
                print(
                    "Validing point {} of {} of the trajectory!".format(
                        i, len(trajectory)
                    )
                )
                if not isinstance(point, dict):
                    raise AssertionError(
                        "Error! The point definition must be a dictionary!\nYour Input: {}".format(
                            type(point).__name__
                        )
                    )
                for field_name, value in point.items():
                    if not isinstance(value, float):
                        raise AssertionError(
                            "Type {} for field name {} is an invalid type! You must input a float value!".format(
                                type(value).__name__, field_name
                            )
                        )
                    if field_name not in valid_point_fields:
                        raise AssertionError(
                            "The provided field name for the point is invalid!"
                        )
                    if field_name == "X" or field_name == "Y":
                        if value > 1000.0 or value < -1000.0:
                            raise AssertionError(
                                "{} value is invalid. Valid range is [-1000.0, 1000.0].".format(
                                    field_name
                                )
                            )
                    elif field_name == "Z":
                        if value > 0.5:
                            print(
                                "WARNING! You have input a Z value that is greater the 0.5, which is below the starting point of the agent (ie. in the ground). Giving these values should only be done if you know the agent will not hit the ground and a negative value should be given for 'positive' altitude. "
                            )
                        if value > 1000.0 or value < -1000.0:
                            raise AssertionError(
                                "{} value is invalid. Valid range is [-1000.0, 1000.0].".format(
                                    field_name
                                )
                            )
                    elif field_name == "Heading":
                        if value < -360.0 or value > 360.0:
                            print(
                                "WARNING! You input a heading value greater then 360 or less than -360. This will be truncated to a proper value!"
                            )
                    elif field_name == "Speed":
                        if value < 0.0 or value > 20.0:
                            raise AssertionError(
                                "Speed value is invalid! Valid range is 0.0 to 20.0 meters per second!"
                            )
            return True
        except AssertionError as error:
            print(error)
            return False
        except Exception:
            traceback.print_exc()
            return False

    def validate_level_is_supported(self, env_name: str, level_name: str) -> bool:
        """
        Validate that the level selected in the environment is valid.
        We always validate the environment name before we validate
        the level name, so we can say envs[env_name]["Levels"] easily
        enough.

        ### Inputs:
        - env_name [str] The name of the environment to run
        - level_name [str] The name of the level inside of the environment to run

        ### Outputs:
        - A boolean determining if this level is supported or not
        """
        if self._file_path is not None:
            file_path = self._file_path + "/settings/SupportedEnvironments.json"
        else:
            file_path = find_file_path("SupportedEnvironments.json", "settings")
        with open(file_path, "r") as file:
            envs = json.load(file)
        try:
            levels = envs["Environments"][env_name]["Levels"]
            assert level_name in levels
            return True
        except AssertionError:
            print(
                "Error! Level {} does not exist!\nSupported Levels are: {}".format(
                    level_name, levels
                )
            )
        except Exception:
            traceback.print_exc()
            return False

    def _determine_local_simulation(self, ip_address: str) -> None:
        """
        Determine if we are accessing the SWARM System locally or from
        a remote server. If we aren't on a local version, specific
        functionality will not be turned on.

        ### Inputs:
        - ip_address [str] The IPv4 address of the server

        ### Outputs:
        - Sets the local flag
        """
        self._local = True
        if ip_address != "127.0.0.1":
            print(
                "Utilizing a remote SWARM Server. Local functionality has been turned off!"
            )
            self._local = False

    # =========================================================================
    #                       Core Message Functions
    # =========================================================================

    def run_simulation(
        self, map_name: str, sim_name: str, ip_address: str, folder: str = "settings"
    ) -> bool:
        """
        Run a Simulation, waiting for the server to tell us when it has
        been completed.

        TODO This should be an Async function that allows the user
        to do other things.

        ### Inputs:
        - map_name [str] The map to run
        - sim_name [str] The name of the sim to run

        ### Outputs:
        - Returns an indicator of whether the simulation has been
          completed
        """
        try:
            # Only connect once we are ready to send a command
            self._determine_local_simulation(ip_address)

            settings, trajectory = self.retrieve_sim_package(sim_name, folder=folder)

            if not isinstance(settings, str):
                assert AssertionError(
                    "Error!\n\n Settings file has been provided in the wrong data format!"
                )

            settings_valid = self.validate_settings_file(json.loads(settings))
            if not settings_valid:
                statement = "Simulation Run Failed.\nReason: Settings file invalid! Please see the settings folder!"
                if self._response_queue is not None:
                    self._response_queue.put({"Command": "RunSimulation", "Message": statement})
                print(statement)
                return False

            # TODO Set up a list of scenarios that require a trajectory
            if self._has_trajectory:
                trajectory_valid = self.validate_multi_level_trajectory_file(
                    json.loads(trajectory)
                )

                if not trajectory_valid:
                    statement = "Simulation Run Failed.\nReason: Trajectory file invalid! Please see the settings folder!"
                    if self._response_queue is not None:
                        self._response_queue.put({"Command": "RunSimulation", "Message": statement})
                    print(statement)
                    return False
                else:
                    print("Trajectory Valid!")
                    if self._response_queue is not None:
                        self._response_queue.put({"Command": "RunSimulation", "Message": "Trajectory Valid!"})

            user_code = self.client.load_user_code(json.loads(settings))

            # Always add the IP address so the server knows if we are
            # local or not
            vehicle_profiles = self._generate_vehicle_profile_list(settings)

            # Always add the IP address so the server knows if we are
            # local or not
            message = {
                "Command": "Run Simulation",
                "Settings": settings,
                "Trajectory": trajectory,
                "UserCode": user_code,
                "Sim_name": sim_name,
                "Map_name": map_name,
                "IPAddress": ip_address,
                "VehicleProfiles": vehicle_profiles
            }

            connected = self.client.connect()

            if not connected:
                statement = "Simulation Run Failed.\nReason: Could not connect to the server!"
                if self._response_queue is not None:
                    self._response_queue.put({"Command": "RunSimulation", "Message": statement})
                print(statement)
                return False
            # Give the client time to establish the connection
            time.sleep(2.0)
            rcvd_msg = self.client.send_simulation_execution_package(message)
            if "Error" in rcvd_msg.keys():
                print("There was an error!")
                print(rcvd_msg["Error"])
                return False
            else:
                self.update_submission_list(rcvd_msg, sim_name=sim_name, folder=folder)
                if self._response_queue is not None:
                    if "Status" in rcvd_msg.keys():
                        self._response_queue.put({"Command": "RunSimulation", "Message": rcvd_msg["Status"]})
                return rcvd_msg["Status"] == "Completed" or rcvd_msg["Status"] == "Client ended simulation!"
        except AssertionError:
            print("Simulation could not be completed!")
            return False
        except Exception:
            traceback.print_exc()

    def _generate_vehicle_profile_list(self, settings: str) -> dict:
        """
        Generate the VehicleProfiles structure to pass the appropriate
        vehicle profiles to the server.

        ### Inputs:
        - settings [str] The settings file

        ### Outputs:
        - Returns the VehicleProfiles structure
        """
        settings = json.loads(settings)
        supported_profiles = self._get_supported_vehicle_profiles()
        vehicle_profiles = {"Vehicles": {}, "Profiles": {}}
        
        for agent_name, agent_info in settings["Agents"].items():
            vehicle_type = agent_info["Vehicle"]
            vehicle_profiles["Vehicles"][agent_name] = agent_info["VehiclePhysicsProfile"]
            if agent_info["VehiclePhysicsProfile"] not in vehicle_profiles["Profiles"].keys():
                # Add the profile name, then read in the Vehicle Profile and load it as a dict
                vehicle_profile = self._load_vehicle_profile(supported_profiles[vehicle_type][agent_info["VehiclePhysicsProfile"]])
                if vehicle_profile is None:
                    if self._response_queue is not None:
                        self._response_queue.put({"Command": "RunSimulation", "Message": "Could not load the vehicle profile with name {}!".format(agent_info["VehiclePhysicsProfile"])})
                    print("Could not load the vehicle profile!")
                    return None
                vehicle_profiles["Profiles"][agent_info["VehiclePhysicsProfile"]] = vehicle_profile
        
        print("DEBUG Vehicle Profiles to send are: {}".format(json.dumps(vehicle_profiles, indent=4)))
        return vehicle_profiles

    def _load_vehicle_profile(self, file_name: str) -> dict:
        """
        Load the vehicle profile based upon the file name

        ### Inputs:
        - file_name [str] The name of the vehicle profile to load

        ### Outputs:
        - Returns the vehicle profile as a dict
        """
        if self._file_path is not None:
            file_path = self._file_path + "/vehicle_profiles/" + file_name
        else:
            file_path = find_file_path(file_name, "vehicle_profiles")
        
        try:
            with open(file_path, "r") as f:
                vehicle_profile = json.load(f)
        
            return vehicle_profile
        except FileNotFoundError:
            print(f"Could not find {file_name} in the vehicle_profiles folder!")
            return None

    def _get_supported_vehicle_profiles(self) -> list:
        """
        Get the supported vehicle profiles.

        ### Outputs:
        - Returns a list of supported vehicle profiles
        """
        if self._file_path is not None:
            file_path = self._file_path + "/vehicle_profiles/VehicleProfiles.json"
        else:
            file_path = find_file_path("VehicleProfiles.json", "vehicle_profiles")
        
        with open(file_path, "r") as f:
            vehicle_profiles = json.load(f)
        
        return vehicle_profiles

    def run_view_only_simulation(
        self, map_name: str, sim_name: str, folder: str = "settings"
    ) -> bool:
        """
        Run a Simulation, waiting for the server to tell us when it has
        been completed.

        TODO This should be an Async function that allows the user
        to do other things.

        ### Inputs:
        - map_name [str] The map to run
        - sim_name [str] The name of the sim to run

        ### Outputs:
        - Returns an indicator of whether the simulation has been
          completed
        """
        try:
            # Only connect once we are ready to send a command

            print("Running simulation {}".format(sim_name))

            settings, trajectory = self.retrieve_sim_package(sim_name, folder=folder)
            settings_valid = self.validate_settings_file(json.loads(settings))
            if not settings_valid:
                print(
                    "Simulation Run Failed.\nReason: Settings file invalid! Please see the settings folder!"
                )

            self.client.connect()
            message = {
                "Command": "View Level",
                "Settings": settings,
                "Trajectory": trajectory,
                "Sim_name": sim_name,
                "Map_name": map_name,
            }
            completed = self.client.send_simulation_execution_package(message)
            if completed is None:
                return False
            if isinstance(completed, dict):
                if "Error" in completed.keys():
                    print("There was an error!")
                    print(completed["Error"])
                    return False
                else:
                    self.update_submission_list(completed, folder=folder)
                    assert completed["Status"] == "Completed"
                    return completed["Status"] == "Completed"
            if "Error" in completed.keys():
                print("There was an error!")
                print(completed["Error"])
                return False
            else:
                self.update_submission_list(completed, folder=folder)
                assert completed["Status"] == "Completed"
                return completed["Status"] == "Completed"
        except AssertionError:
            print("Simulation could not be completed!")
            return False
        except Exception:
            traceback.print_exc()

    def extract_data(self, sim_name: str) -> bool:
        """
        Run a Simulation, waiting for the server to tell us when it has
        been completed. Ensure that we send an access key to validate
        that we can get access to the system.

        TODO This should be an Async function that allows the user
        to do other things.

        ### Inputs:
        - sim_name [str] The name of the sim to run

        ### Outputs:
        - Returns an indicator of whether the extraction has been
          completed
        """
        try:
            statement = "Running Data Extraction for Simulation {}".format(sim_name)
            print(statement)
            if self._response_queue is not None:
                self._response_queue.put({"Command": "RunSimulation", "Message": statement})

            # TODO Add regenerate logic
            if not self.client.connected:
                self.regenerate_connection()

            message = {"Command": "Extract Data", "SimName": sim_name}
            if self.debug:
                print(f"DEBUG: Sending {message}")
            completed = self.client.send_data_extraction_message(message)
            print(type(completed))
            if completed is None:
                return False
            if isinstance(completed, bool):
                return completed
            if "Error" in completed.keys():
                print("There was an error!")
                print(completed["Error"])
                return False
            else:
                return True
        except AssertionError:
            print("Simulation could not be completed!")
            return False
        except KeyboardInterrupt:
            print("Shutting down connection!")
            # Give time for the Server to stope the Simulation
            time.sleep(4)
            self.client.close()
        except Exception:
            traceback.print_exc()

    def retreive_supported_environments(self) -> bool:
        """
        Retrieve the supported environments from the SWARM Core
        and provide those as a JSON file stored in the `settings` folder
        called `SupportedEnvironments.json`.

        ### Inputs:
        - None

        ### Outputs:
        - None
        """
        try:
            print("Requesting the supported environemtns from SWARM Core")

            time.sleep(1)
            # TODO Add regenerate logic
            if not self.client.connected:
                self.regenerate_connection()

            message = {
                "Command": "Supported Environments",
            }
            # Returns the body of the message containing the list
            # of supported environments
            environments = self.client.send_supported_envs_message(message)
            if isinstance(environments, bool):
                return environments
            elif "Error" in environments.keys():
                print("There was an error!")
                print(environments["Error"])
                return False
            else:
                print("Supported Environments with SWARM Container:")
                for env_name in environments["SupportedEnvironments"]:
                    print("\t{}".format(env_name))
                if self._file_path is not None:
                    file_path = self._file_path + "/settings/SupportedEnvironments.json"
                else:
                    file_path = find_file_path("SupportedEnvironments.json", "settings")
                with open(file_path, "w") as file:
                    json.dump(
                        {"Environments": environments["SupportedEnvironments"]}, file
                    )
                return True
        except AssertionError:
            print("Simulation could not be completed!")
            return False
        except Exception:
            traceback.print_exc()

    def retreive_environment_information(self, env_name: str) -> bool:
        """
        Retrieve the map and metadata from the SWARM Core
        instance. This will reach out to the server and download
        a map_data.tar.gz file containing the map information of the
        environment provided.

        ### Usage:
        - Create a sim manager with
          `sim_manager = SWARM()`
        - Call this method and wait to receive the required files

        ### Inputs:
        - env_name [str] The name of the environment

        ### Outputs:
        - A boolean telling you whether this worked or not
        """
        try:
            print(
                "Requesting the map information for {} environment from SWARM Core".format(
                    env_name
                )
            )

            time.sleep(1)
            # TODO Add regenerate logic
            if not self.client.connected:
                self.regenerate_connection()

            message = {
                "Command": "Environment Information",
                "EnvironmentName": env_name,
            }
            if self.debug:
                print(f"DEBUG: Sending {message}")
            completed = self.client.send_env_information_message(message)
            if self.debug:
                print(f"DEBUG: Received {completed}")
            if isinstance(completed, bool):
                return completed
            if "Error" in completed.keys():
                print("There was an error!")
                print(completed["Error"])
                return False
            else:
                return True
        except AssertionError:
            print("Simulation could not be completed!")
            return False
        except Exception:
            traceback.print_exc()

    def validate_user_code(
        self, settings_file: str = "settings/DefaultSimulationSettings.json"
    ) -> None:
        """
        Given a settings file, determine if any custom code has been
        indicated by the User and validate said code if that is
        the case.
        """
        try:
            with open(settings_file, "r") as file:
                settings = json.load(file)

            custom_code = False
            for agent_name, agent_info in settings["Agents"].items():
                for module_name, module in agent_info["SoftwareModules"].items():
                    if module["Algorithm"]["Level"] == 3:
                        custom_code = True
            print("DEBUG Code is custom: {}".format(custom_code))
            if custom_code:
                completed = self.client.send_user_code_for_validation({}, settings)
        except AssertionError:
            print("Simulation could not be completed!")
            return False
        except Exception:
            traceback.print_exc()
