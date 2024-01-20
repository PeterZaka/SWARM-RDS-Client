import matplotlib.pyplot

def graph(grid, start, end, pathList):

    # Separate each point into x,y,z (col, row, depth) and color
    col = []
    row = []
    depth = []
    categories = []

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            for k in range(len(grid[i][j])):
                if (i, j, k) == start:
                    col.append(j)
                    row.append(i)
                    depth.append(k)
                    categories.append(3)
                elif (i, j, k) == end:
                    col.append(j)
                    row.append(i)
                    depth.append(k)
                    categories.append(2)
                elif (i, j, k) in pathList:
                    col.append(j)
                    row.append(i)
                    depth.append(k)
                    categories.append(0)
                elif grid[i][j][k] == 1:
                    col.append(j)
                    row.append(i)
                    depth.append(k)
                    categories.append(1)

    row = [-r for r in row] # To have the graph oriented 'correctly'

    # Define custom colors for each category
    category_colors = {0: 'yellow', 1: 'gray', 2: 'green', 3: 'blue', 4: 'white'}  

    # Map categories to colors
    colors = [category_colors[category] for category in categories]

    # Create a 3D scatter plot
    figure = matplotlib.pyplot.figure()
    axes = figure.add_subplot(111, projection='3d')

    # Scatter plot with discrete colors
    scatter = axes.scatter(col, depth, row, c=colors, s = 1000)

    # Set labels for the axes
    axes.set_xlabel('Column')
    axes.set_ylabel('Depth')
    axes.set_zlabel('Row * -1')

    # Show the plot
    matplotlib.pyplot.show()






