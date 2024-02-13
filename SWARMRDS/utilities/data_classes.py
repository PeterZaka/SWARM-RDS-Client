# =============================================================================
# Copyright 2022-2023. Codex Laboratories LLC. All rights reserved.
#
# Created By: Tyler Fedrizzi
# Created On: 9 June 2022
#
#
# Description: AirSim interface file for the SWARM platform
# =============================================================================
import copy
import json
import math
import numpy as np

from dataclasses import dataclass, field
from enum import Enum, unique
from math import sqrt

MAX_MESSAGE_SIZE = 220  # Bytes


@dataclass
class PosVec3:
    """
    Position vector containing the X, Y and Z coordinates as defined
    in an Earth-Center, Earth Facing global coordinate system.

    Units are Meters.

    Members are:
    - X: X coordinate of the ECEF coordinate system
    - Y: Y coordinate of the ECEF coordinate system
    - Z: Z coordinate of the ECEF coordinate system
    """
    X: float = 0.0
    Y: float = 0.0
    Z: float = 0.0
    frame: str = "body"
    NED: bool = True
    ENU: bool = False

    def toNumpyArray(self):
        return np.array([
           self.X,
           self.Y,
           self.Z 
        ])

    def toENU(self):
        temp_x = copy.deepcopy(self.X)
        self.X = self.Y
        self.Y = temp_x
        self.Z = -self.Z
        self.NED = False
        self.ENU = True

    def toNED(self):
        temp_x = copy.deepcopy(self.X)
        self.X = self.Y
        self.Y = temp_x
        self.Z = -self.Z
        self.NED = True
        self.ENU = False

    def displayPretty(self):
        return {"X": self.X,
                "Y": self.Y,
                "Z": self.Z,
                "Frame": self.frame,
                "ENU": self.ENU,
                "NED": self.NED}

    def toList(self):
        return [self.X, self.Y, self.Z]


@dataclass
class GPSPosVec3:
    """
    Position vector containing the Latitude, Longitude and Altitude
    coordinates as defined in an elliptical spheriod Earth model.

    Units are Meters.

    Members are:
    - Lat: Latitude of the current vehicle in degrees
    - Lon: Longitude of the current vehicle in degrees
    - Alt: Altitude of the current vehicle in meter
    """
    Lat: float = 0.0
    Lon: float = 0.0
    Alt: float = 0.0

    def toRadians(self):
        self.Lat = math.radians(self.Lat)
        self.Lon = math.radians(self.Lon)

    def LLAToECEF(self):
        pass

    def ECEFtoLLA(self):
        pass

    def displayPretty(self):
        return {"Latitude": self.Lat,
                "Longitude": self.Lon,
                "Altitude": self.Alt}

@dataclass
class VelVec3:
    """
    Velocity vector containing the X, Y and Z velocities as measured
    relative to the body frame of the vehicle.

    Units are Meters / Second

    Members are:
    - vx - x-component of the velocity vector
    - vy - y-component of the velocity vector
    - vz - z-component of the velocity vector
    """
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    frame: str = "body"
    NED: bool = True
    ENU: bool = False

    def toNumpyArray(self):
        return np.array([
           self.vx,
           self.vy,
           self.vz 
        ])

    def toENU(self):
        temp_x = copy.deepcopy(self.vx)
        self.vx = self.vy
        self.vy = temp_x
        self.vz = -self.vz
        self.NED = False
        self.ENU = True

    def toNED(self):
        temp_x = copy.deepcopy(self.vx)
        self.vx = self.vy
        self.vy = temp_x
        self.vz = -self.vz
        self.NED = True
        self.ENU = False

    def displayPretty(self) -> dict:
        return {"vx": self.vx,
                "vy": self.vy,
                "vz": self.vz,
                "Frame": self.frame,
                "ENU": self.ENU,
                "NED": self.NED}


@dataclass
class AccVec3:
    """
    Velocity vector containing the X, Y and Z velocities as measured
    relative to the body frame of the vehicle.

    Units are Meters / Second

    Members are:
    - ax - x-component of the acceleration vector
    - ay - y-component of the acceleration vector
    - az - z-component of the acceleration vector
    """
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    frame: str = "body"
    NED: bool = True
    ENU: bool = False

    def toNumpyArray(self):
        return np.array([
           self.ax,
           self.ay,
           self.az 
        ])

    def toENU(self):
        temp_x = copy.deepcopy(self.ax)
        self.ax = self.ay
        self.ay = temp_x
        self.az = -self.az
        self.NED = False
        self.ENU = True

    def toNED(self):
        temp_x = copy.deepcopy(self.ax)
        self.ax = self.ay
        self.ay = temp_x
        self.az = -self.az
        self.NED = True
        self.ENU = False

    def displayPretty(self) -> dict:
        return {"vx": self.ax,
                "vy": self.ay,
                "vz": self.az,
                "Frame": self.frame,
                "ENU": self.ENU,
                "NED": self.NED}


@dataclass
class ECEF():
    """
    A position in Earth Centered Earth Facing coordinates. This position
    can easily be converted to a specific coordinate frame that will
    allow for easier versions of planning.
    """
    X: float = float()
    Y: float = float()
    Z: float = float()

    def displayPretty(self):
        return {
            "X": self.X,
            "Y": self.Y,
            "Z": self.Z
        }

@dataclass
class AccelVec4:
    """
    Acceleration vector containing the acceleration to apply along the
    roll, pitch and yaw angle rates, with a throttle position available
    to provide the required thrust.

    Units are meters / second / second.

    Members:
    - roll - angle rate around the x (front) axis
    - pitch - angle rate around the y (side) axis
    - yaw - angle rate around the z (up) axis
    - throttle - ratio to apply around each axis
    """
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    throttle: float = 0.0  # Must be between 0 and 1
    NED: bool = True
    ENU: bool = False

    def toNumpyArray(self):
        return np.array([
           self.roll,
           self.pitch,
           self.yaw 
        ])

    def displayPretty(self) -> dict:
        return {"roll": self.roll,
                "pitch": self.pitch,
                "yaw": self.yaw,
                "throttle": self.throttle}


@dataclass
class Quaternion:
    """
    Rotation quaternion that represents the orientation of a body
    in free space.

    Units are Radians

    Members are:
    - w - relates the rotational magnitude of an orientation
    - x - similar to x coordinate in traditional coordinate system
    - y - similar to y coordinate in traditional coordinate system
    - z - similar to z coordinate in traditional coordinate system

    Reference:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    NED: bool = True
    ENU: bool = False

    def length(self) -> float:
        return sqrt(pow(self.w, 2)
                    + pow(self.x, 2)
                    + pow(self.y, 2)
                    + pow(self.z, 2))

    def toENU(self) -> None:
        x = copy.deepcopy(self.x)
        self.x = copy.deepcopy(self.y)
        self.y = x
        self.z = -self.z
        self.NED = False
        self.ENU = True

    def toNED(self) -> None:
        x = copy.deepcopy(self.x)
        self.x = copy.deepcopy(self.y)
        self.y = x
        self.z = -self.z
        self.NED = True
        self.ENU = False
    
    def unitize(self) -> None:
        """
        Ensures that any quaternion is normalized to a unit quaternion,
        or more generally, to a quaternion that has a length of one.
        This is important when describing the orientation of a physical
        body by a quaternion.

        Divdes each comnponent of the quaternion by the length of the
        quaternion, return each component as a float.
        """
        length = self.length()
        self.x = self.x / length
        self.y = self.y / length
        self.z = self.z / length
        self.w = self.w / length

    def displayPretty(self) -> dict:
        return {"x": self.x,
                "y": self.y,
                "z": self.z,
                "w": self.w}


@dataclass
class Attitude:
    """
    The current attitude of the aircraft in the NED coordinate frame,
    which corresponds to the Roll, Pitch and Yaw in the X, Y and Z axes
    respectively. This is critical to understand that this is only valid
    in the NED framework.

    ## Memebers:
    - roll [float] the angle of the vehicle around the Y axis
    - pitch [float] the angle of the vehicle around the X axis
    - yaw [float] the angle of the vehicle around the Z axis
    """
    roll: float = float()  # radians
    pitch: float = float()  # radians
    yaw: float = float()  # radians

    def displayPretty(self) -> dict:
        return {"roll": self.roll,
                "pitch": self.pitch,
                "yaw": self.yaw}


@dataclass
class AgentState:
    """
    Compact representation of the agents current state, with a similar
    format as the ROS message odemtry.

    Members are:
    - position [X, Y, Z] in NED coordiante frame
    - gps_position [Lat, Lon, Alt] in GPS coordinate frame
    - linear_velocity [vx, vy, vz] in NED coordinate frame
    - angular_velocity [vx, vy, vz] in NED coordiante frame
    - linear_acceleration [ax, ay, az] in NED coordinate frame
    - angular_acceleration [ax, ay, az] in NED coordainte frame
    - orientation [w, x, y, z] in NED coordiante frame
    - heading [float] The current heading of the vehicle in the NED
                      coordinate frame.
    """
    position: PosVec3 = field(default_factory=PosVec3)
    gps_position: GPSPosVec3 = field(default_factory=GPSPosVec3)
    linear_velocity: VelVec3 = field(default_factory=VelVec3)
    angular_velocity: VelVec3 = field(default_factory=VelVec3)
    linear_acceleration: AccVec3 = field(default_factory=AccVec3)
    angular_acceleration: AccVec3 = field(default_factory=AccVec3)
    orientation: Quaternion = field(default_factory=Quaternion)
    attitude: Attitude = field(default_factory=Attitude)
    heading: float = float()

@dataclass
class Orientation:
    """
    Current orientation of the vehicle, as given by a set of angles in 
    the NED coordinate frame (Roll, Pitch, Yaw) and a quaternion

    Members:
    - roll - angle rate around the x (front/back) axis
    - pitch - angle rate around the y (side) axis
    - yaw - angle rate around the z (up/down) axis
    - quat [Quaternion] - Rotatation device
    """
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    quat: Quaternion = field(default_factory=Quaternion)


@dataclass
class MovementCommand():
    """
    AirSim requires a Movement command, which contains a number of
    different properties. In general, this movement command will be the
    next X, Y and Z positions of the UAV in the local NED coordiante
    frame, along with the orientation that the drone should be headed
    in, the speed at which the UAV should approach that target and the
    UAV ID of the message.

    ## Inputs:
    - position [PosVec3] X,Y,Z NED position in the local coordinate
                         frame. Initialzied as X=0, Y=0, Z=0,
                         Frame=Local
    - heading [float] Yaw angle of drone, with inital heading being 0.0
                      in Degrees
    - speed [float] Speed to travel in the direction of travel in meters
                    per second
    """
    position: PosVec3 = field(default_factory=PosVec3)
    velocity: VelVec3 = field(default_factory=VelVec3)
    acceleration: AccelVec4 = field(default_factory=AccelVec4)
    heading: float = 0.0 # degrees
    speed: float = 5.0 # meters / second
    priority: int = 1
    move_by: str = "position"

    def displayPretty(self) -> str:
        display_dict = dict()
        display_dict["position"] = self.position.displayPretty()
        display_dict["velocity"] = self.velocity.displayPretty()
        display_dict["acceleration"] = self.acceleration.displayPretty()
        display_dict["heading"] = self.heading
        display_dict["speed"] = self.speed
        display_dict["move_by"] = self.move_by
        return display_dict


@dataclass
class Trajectory():
    """
    A multi-point path that should be followed by the agent. Can be
    generated from a Python Dictionary or List or loaded directly via
    Pickle.

    ## Members:
    - points list[PosVec3]
    - speed [float] The speed to travel in meters
    - start_time [float] The time in seconds since the Linux epoch
    - end_time [float] The time in seconds since the Linux epoch when
                       the trajectory should end
    """
    points: list = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    def convert_dict_to_traj(self, traj_dict: dict) -> None:
        """
        Given a list of waypoints as a set of dictionaries with the
        following layout:
        {
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0
        }
        create a valid trajectory to be given to the algorithm.
        """
        for point in traj_dict:
            self.points.append(PosVec3(X=point["X"], Y=point["Y"], Z=point["Z"]))

    def convert_list_to_traj_with_heading_and_speed(self, traj: list) -> None:
        """
        Given a list of waypoints as a set of dictionaries with the
        following layout:
        {
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0
        }
        create a valid trajectory to be given to the algorithm.
        """
        for point in traj:
            self.points.append((PosVec3(X=point["X"], Y=point["Y"], Z=point["Z"]), point["Heading"], point["Speed"]))
    
    def displayPretty(self) -> dict:
        traj_points = list()
        for point in self.points:
            traj_points.append(point.displayPretty())
        
        return traj_points


@dataclass
class Detection():
    """
    We generate a number of detections, whether through the AirSim
    API or through a computer vision algorithm.

    ## Inputs:
    - position [PosVec3] X,Y,Z NED position in the local coordinate
                         frame. Initialzied as X=0, Y=0, Z=0,
                         Frame=Local
    - heading [float] Yaw angle of drone, with inital heading being 0.0
                      in Degrees
    - speed [float] Speed to travel in the direction of travel in meters
                    per second
    """
    position: PosVec3 = field(default_factory=PosVec3)
    gps_position: GPSPosVec3 = field(default_factory=GPSPosVec3)
    label: str = ""
    timestamp: str = ""


@dataclass
class Message():
    """
    A generic Message container that enables a user to Send a specific
    set of information to another Agent in the Simulation.
    """
    body: dict = field(default_factory=dict)
    message_buffer: list = field(default_factory=list)
    timestamp: str = ""
    recipient: str = ""

    def length(self) -> int:
        """
        Calculate the length of the body in JSON converted,
        utf-8 serialized bytes.
        """
        return len(json.dumps(self.body).encode('utf-8'))

    def multi_part(self) -> list:
        """
        Convert the Message into a set of messages, where each message
        is a set of sub-bytes of the overall message.

        **Note** This method is run on each message, so we know
        """
        if self.length() > MAX_MESSAGE_SIZE:
             
            numb_buffers = int(np.ceil(self.length() / MAX_MESSAGE_SIZE))
            for i in range(numb_buffers):
                self.message_buffer.append(Message(body=json.dumps(self.body)[i * MAX_MESSAGE_SIZE:(i + 1) * MAX_MESSAGE_SIZE],
                                              timestamp=self.timestamp,
                                              recipient=self.recipient))
            return self.message_buffer
        return None


@dataclass
class MetaData():
    """
    Image metadata that describes the location, orientation and other
    features of the camera.

    ## Members:
    - position [PosVec3] The agent position when the image was taken
    - orientation [Quaternion] The orientation of the camera when
                               the iamge was taken
    """
    position: PosVec3 = field(default_factory=PosVec3)
    orientation: Quaternion = field(default_factory=Quaternion)
    timestamp: int = 0  # Seconds since the Linux epoch



@dataclass
class ImageMetadata(MetaData):
    """
    Image metadata that describes the location, orientation and other
    features of the camera.

    ## Members:
    - image_height [int] The height of the image in pixels
    - image_width [int] The width of the image in pixels
    - image_type [int] The type of image that was taken
    """
    image_height: int = 480
    image_width: int = 640
    image_type: int = 0  # TODO: Add Image Type
    # Need to include something like the calibration parameters and 
    # distortion parameters


@dataclass
class PointCloudMetaData(MetaData):
    """
    Point Cloud metadata that describes the location, orientation and
    other features of the camera.

    ## Members:
    - horizontal_fov list(float, float) The horizontal field of view of the camera
                             in degrees
    - vertical_fov list(float, float) The (Left and Right bounds) of the
                     vertical field of view of the camera in degrees
    - min_range [float] The minimum range of the sensor in meters
    - max_range [float] The maximum range of the sensor in meters
    """    
    horizontal_fov: list = field(default_factory=list)
    vertical_fov: list = field(default_factory=list)
    min_range: float = 0.0
    max_range: float = 0.0


@dataclass
class SWARMImage():
    """
    A generic Image container that can be transferred across the SWARM
    ecosystem. Has a base C++ class implementation and extension
    methods for OpenCV, Pillow and more.

    ## Members:
    - data [list] The raw bytes of information
    - timestamp [time] The raw time since Linux Epoch in nanoseconds
    - metadata [dict] The information contained within the image, such
                      as position of the camera.
    """
    data: list = field(default_factory=list)
    timestamp: float = 0.0
    metadata: MetaData = field(default_factory=MetaData)


@dataclass
class SWARMPointCloud:
    """
    Wrapper class for a point cloud, which is a set of points in 3D
    space that can be used to generate a 3D representation of the
    environment.

    ### Attributes:
    - points [list] A list of points in 3D space. Each point is tuple of
                    (x, y, z) coordinates (in that order)
    - metadata [Metadata] The metdata of the Point Cloud as defined in
                            the Metadata class
    """
    points: list = field(default_factory=list)
    metadata: PointCloudMetaData = field(default_factory=PointCloudMetaData)
