# Example Data for SWARM RDS
Provided in this folder are examples of actual objects that are passed to different
modules in the system. Each `.pickle` file contains a different object. The objects
are stored in "raw" format, so you can unpickle and view the type. A description of 
each file is listed below.
  
## Usage of Visualizer
To use the visualizer, run the following command:
```bash
python3 visualizer.py <path_to_pickle_file>
```
  
## File Descriptions
    - `occupancy.pickle`: A 200 x 200 meter occupancy grid with a grid size of 1 meter. Be aware that the agent starts at the middle of this map!
    - `point_cloud.pickle`: A point cloud of the environment. The point cloud is a list of points, where each point is a tuple of (x, y, z) coordinates.
    - `raw_point_cloud.pickle`: A point cloud of the environment. The point cloud is a list of points, where each point is a tuple of (x, y, z) coordinates. This point cloud is not filtered, so it contains all points.
    - `search_traj.pickle`: A list of points that the agent should visit. The agent should start at the first point in the list and visit each point in order.
