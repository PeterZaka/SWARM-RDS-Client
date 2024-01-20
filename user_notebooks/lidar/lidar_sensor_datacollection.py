import airsim

# Create an AirSim client
client = airsim.MultirotorClient()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json
client.confirmConnection()

# Set the desired vehicle name (e.g., "Drone1" or "Car1")
vehicle_name = "Drone1"
client.enableApiControl(True)

# Arm the vehicle
client.armDisarm(True)

# Take off to 10 meters
client.takeoffAsync(10).join()
print("Vehicle armed and flying")

# move to the left 3 meters
client.moveToPositionAsync(0, 0, -3, 5).join()

# Enable the camera on the drone (replace '0' with the camera ID if you have multiple cameras)
# client.simEnableLidar("0", True)

# Capture a photo from the drone's camera
response = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])

# Save the captured image to a file
if response:
    airsim.write_file("captured_image.png", response[0].image_data_uint8)
    print("Image saved as 'captured_image.png'")
else:
    print("Failed to capture an image.")

# Define a function to capture LiDAR data and group points by XYZ values
def group_lidar_data_by_xyz(sensor_name):
    lidar_data = client.getLidarData(sensor_name)
    if not lidar_data:
        print("No LiDAR data available")
        return []

    # Extract X, Y, Z coordinates from the point cloud
    point_cloud = lidar_data.point_cloud
    grouped_points = []

    for i in range(0, len(point_cloud), 3):
        x = point_cloud[i]
        y = point_cloud[i + 1]
        z = point_cloud[i + 2] * -1  # Invert Z axis (left-handed coordinate system)
        grouped_points.append([x, y, z])

    return grouped_points, lidar_data


# Define a function to plot grouped LiDAR data
def plot_grouped_lidar_data(grouped_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_coords, y_coords, z_coords = zip(*grouped_data)  # Unpack the coordinates

    ax.scatter(x_coords, y_coords, z_coords, marker='.')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

# Capture LiDAR data and group points by XYZ values
sensor_name = "LidarSensor1"  # Replace with your actual LiDAR sensor name
grouped_data, lidar_data = group_lidar_data_by_xyz(sensor_name)
print(grouped_data)
# Print the grouped data
print("Grouped LiDAR data:")

plot_grouped_lidar_data(grouped_data)



# Function to replace 'point_cloud' with 'grouped_data' in LiDAR data and save it to a JSON file
def replace_and_save_lidar_data(filename, lidar_data, grouped_data):
    # Create a dictionary from the 'lidar_data'
    lidar_dict = {'point_cloud': grouped_data}

    # Extract and convert the serializable parts of 'lidar_data'
    serializable_data = {
        'pose': {
            'orientation': {
                'w_val': lidar_data.pose.orientation.w_val,
                'x_val': lidar_data.pose.orientation.x_val,
                'y_val': lidar_data.pose.orientation.y_val,
                'z_val': lidar_data.pose.orientation.z_val
            },
            'position': {
                'x_val': lidar_data.pose.position.x_val,
                'y_val': lidar_data.pose.position.y_val,
                'z_val': lidar_data.pose.position.z_val
            }
        },
        'segmentation': lidar_data.segmentation,
        'time_stamp': lidar_data.time_stamp
    }

    # Add the serializable data to the dictionary
    lidar_dict.update(serializable_data)

    # Save the modified LiDAR data to the JSON file
    with open(filename, 'w') as json_file:
        json.dump(lidar_dict, json_file, indent=4)
        print(f"LiDAR data saved to {filename}")

# Assuming you have already obtained 'lidar_data' and 'grouped_data' from your code

# Define the filename for the JSON file
json_filename = 'lidar_data_sample_boxed_env.json'

# Call the function to replace 'point_cloud' and save the modified LiDAR data
replace_and_save_lidar_data(json_filename, lidar_data, grouped_data)

print(f'LiDAR data with grouped data saved to {json_filename}')




# Land the drone
print("Landing sequence initiated...")
client.landAsync()
print("Landing sequence completed.")

# Disarm the drone
client.armDisarm(False)

# Close the connection to the AirSim client
client.reset()