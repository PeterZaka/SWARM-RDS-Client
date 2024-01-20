import json
import numpy as np
from PIL import Image
from coordinate_gen import ground_truth


def write_array_into_image(drone_pos,pixels, filename):

    h = len(pixels)
    w = len(pixels[0])
    img = Image.new('RGB', (w, h))
    output_pixels = img.load()
    for y, row in enumerate(pixels):
        for x, color in enumerate(row):
            
            if(pixels[y][x] == 1):
                output_pixels[x, y] = (0, 252,0)
            elif(y== round(drone_pos[1])  and x== round(drone_pos[0])):
                output_pixels[x, y] = (0,252 ,0)
            else:
                output_pixels[x, y] = (0, 0,0)

    img.save(filename)


class lidar_data_manager():
    def __init__(self,fp):
        self.timestamps = []
        self.index = 0
        # x = ground_truth("lidar_data_series.json","result.json")
        x = ground_truth(fp,"result.json")
        file = open("result.json")
        data = json.load(file)
        for i in data:
            self.timestamps.append(i)
        self.data = x.new_data

    def get_lidar_data(self):
        self.index += 1
        return self.data[self.timestamps[self.index - 1]]["point_cloud"]
    

class map_gen():
    
    def __init__(self,map_length,map_width,z_max,z_min,resolution,lidar_file):
        """""
        Go from real world to numpy array
        input:
            map_length: length of the map
            map_width: length of the map
            z_max: upper z range from which the map will get updated
            z_min: lower z range from which the map will get updated
            resolution: resolution of the map
            lidar_file: file with lidar information
        output:
            self.map : contains the binary occupancy map
        """
        self.map = []
        self.z_max = z_max
        self.z_min = z_min
        self.map_length = map_length
        self.map_width =  map_width
        self.map_size = [map_length,map_width]
        self.resolution = resolution
        for i in range(self.map_length):
            self.map.append([])
            for j in range(self.map_width):
                self.map[i].append(0)
        self.lidar_process = lidar_data_manager(lidar_file)
        # print(self.map)

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
    
    def update_map(self):

        point_cloud = self.lidar_process.get_lidar_data()
        arr = np.array(point_cloud)
        mask = (arr[:, 2] >= self.z_min) & (arr[:, 2] <= self.z_max)
        point_cloud = arr[mask, :2]
        for point in point_cloud:
            x, y = point
            x,y = self.calc_real_to_array((x,y))
            grid_x = x
            grid_y = y
            if 0 <= grid_x < self.map_length and 0 <= grid_y < self.map_width:
                # print(grid_x,grid_y)
                self.map[grid_x][grid_y] = 1

# call map_gen to create a map object

x = map_gen(200,200,-1,-2,0.5,"lidar_data_series.json")
x.update_map()
x.update_map()

write_array_into_image([0,0],x.map,"testfile.jpg")

# x = lidar_data_manager()

# x.convert_data()
