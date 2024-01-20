import graph3d
import math

def voxelToArray(filename : str) -> list[list[list[int]]]:
    filePointer = open(filename, "r")

    header = filePointer.readline()

    # Check for header
    if header.find("#binvox") == -1:
        print("Error in format")
        return []
    
    # Check for dimensions
    dimensions = filePointer.readline()
    dimensions = dimensions.split(" ")

    if len(dimensions) != 4 :
        print("Dimension length Error")
        return []

    dimensions = [int(d) for d in dimensions[1:]]

    if not (dimensions[0] == dimensions[1] == dimensions[2]):
        print("Dimension Error")
        return []
    
    dimension = dimensions[1]
    
    # Check for translate
    translate = filePointer.readline()
    translate = translate.split(" ")
    if len(translate) != 4:
        print("Translate Error")
        return []

    # Check for scale
    scale = filePointer.readline()
    scale = scale.split(" ")
    if len(scale) != 2:
        print("Scale Error")
        return []

    # Check for data
    data = filePointer.readline()
    if data != "data\n":
        print("Data Error")
        return []
    
    voxelArray = [[[0 for i in range(dimension)] for j in range(dimension)] for k in range(dimension)] #empty 3D array
    indexCount = 0

    #Reopen file in binary mode
    index = filePointer.tell()
    filePointer.close()

    filePointer = open(filename, "rb")
    filePointer.seek(index)

    while True:
    
        # Read data
        firstByte = filePointer.read(1)
        secondByte = filePointer.read(1)

        if not firstByte or not secondByte:
            break

        count = int.from_bytes(secondByte, byteorder='big')
        boolean = int.from_bytes(firstByte, byteorder='big')
        assert boolean == 1 or boolean == 0

        for i in range(count):
            y = indexCount % dimension
            x = math.floor(indexCount / dimension) % dimension
            z = math.floor(indexCount / (dimension * dimension)) % dimension
            voxelArray[y][x][z] = boolean
            indexCount += 1
            
    return voxelArray

a= voxelToArray("chair.binvox")
    
graph3d.graph(a)