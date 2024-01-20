import json
import ast

from matplotlib.font_manager import json_dump

class Quaternion():
   def __init__(self, qx: float, qy: float, qz: float, qw: float):
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw


class ground_truth():
    def __init__(self, fp, fp1, file_type='json'):

        self.world_frame = Quaternion(0,0,0,1)
        file = open(fp)
        if file_type == 'json':
            self.data = json.load(file)
        else:
            mylist = ast.literal_eval(file.read())
            self.data = {mylist[i+1]['time_stamp'] : mylist[i] | mylist[i+1] for i in range(0, len(mylist), 2)}
        self.new_data = self.data
        self.calculate_coordiante()
        self.write_file(fp1)


    def calculate_coordiante(self):
        for i in self.data:
            drone_frame = Quaternion(self.data[i]["pose"]["orientation"]["x_val"],self.data[i]["pose"]["orientation"]["y_val"],self.data[i]["pose"]["orientation"]["z_val"],self.data[i]["pose"]["orientation"]["w_val"])
            matrix = self.Quat2Mat(self.rotDiff(drone_frame,self.world_frame))
            for ind,j in enumerate(self.data[i]["point_cloud"]):
                x = self.coordTransform(matrix,j)
                self.new_data[i]["point_cloud"][ind] = [x[0] + self.data[i]["pose"]["position"]["x_val"],x[1] + self.data[i]["pose"]["position"]["y_val"],( x[2] - self.data[i]["pose"]["position"]["z_val"])]
    def rotAdd(self, q1: Quaternion, q2: Quaternion) -> Quaternion:
        w1 = q1.qw
        w2 = q2.qw
        x1 = q1.qx
        x2 = q2.qx
        y1 = q1.qy
        y2 = q2.qy
        z1 = q1.qz
        z2 = q2.qz

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

        return Quaternion(x,y,z,w)

    def rotDiff(self,q1: Quaternion,q2: Quaternion) -> Quaternion:
        conjugate = Quaternion(q2.qx*-1,q2.qy*-1,q2.qz*-1,q2.qw)
        return self.rotAdd(q1,conjugate)


    def Quat2Mat(self, q: Quaternion):
        m00 = 1 - 2 * q.qy**2 - 2 * q.qz**2
        m01 = 2 * q.qx * q.qy - 2 * q.qz * q.qw
        m02 = 2 * q.qx * q.qz + 2 * q.qy * q.qw
        m10 = 2 * q.qx * q.qy + 2 * q.qz * q.qw
        m11 = 1 - 2 * q.qx**2 - 2 * q.qz**2
        m12 = 2 * q.qy * q.qz - 2 * q.qx * q.qw
        m20 = 2 * q.qx * q.qz - 2 * q.qy * q.qw
        m21 = 2 * q.qy * q.qz + 2 * q.qx * q.qw
        m22 = 1 - 2 * q.qx**2 - 2 * q.qy**2
        result = [[m00,m01,m02],[m10,m11,m12],[m20,m21,m22]]

        return result
    
    def coordTransform(self, M, A):

        APrime = []
        # i = 0
        # for component in A:
        #     APrime.append(component * M[i][0] + component * M[i][1] + component * M[i][2])
        #     i += 1
        APrime.append(A[0] * M[0][0] + A[1] * M[0][1] + A[2] * M[0][2])
        APrime.append(A[0] * M[1][0] + A[1] * M[1][1] + A[2] * M[1][2])
        APrime.append(A[0] * M[2][0] + A[1] * M[2][1] + A[2] * M[2][2])
        return APrime
    
    def write_file(self,fp):
        json_dump(self.new_data,fp)

# x = ground_truth("ground_only_lidar_data_2023-10.json","result2.json")
# x = ground_truth("lidar_data_boxed_env.json", "result.json", 'list')


