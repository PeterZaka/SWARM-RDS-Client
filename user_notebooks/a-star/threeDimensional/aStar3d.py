import queue
import math

class Node:

    def __init__(self, position : tuple[int, int, int], parent : 'Node' = None) -> None:
        self.position = position
        self.parent = parent # The last node to update this node's distance value (With the shortest path)

        self.distance = 0 # The correct distance from the start position to this Node
        self.heuristic = 0 # An estimate of the distance from the goal to this node

    def __eq__(self, other: object) -> bool: # Are the positions the same
        return (self.position[0] == other.position[0] and
                self.position[1] == other.position[1] and
                self.position[2] == other.position[2])
    
    def __hash__(self) -> int: # Allows usage in a set
        return hash(self.position)
    
    def __lt__(self, other: object) -> bool: # Allows usage in a Priority queue
        return ((self.distance + self.heuristic) < (other.distance + other.heuristic))

    

def aStar3D(start : tuple[int, int, int], end : tuple[int, int, int], grid : list[list[list[int]]], diagonal : bool = False) -> list[tuple[int, int, int]]:
    """
    # A* 3d
    AStar Algorithm for use in 3d Space. Returns a shortest path between start and end based on the grid.

    ## Parameters:
    - Start: A coordinate of the start position (Row, Column, Depth)
    - End: A coordinate of the end position
    - Grid: A 3d List of integers where (0) represents open space and (1) represents occupied space. Indexed from rows, columns, then depth.


    ## Return:
    - List: A list of 3d coordinates (tuples of ints) from start to end that represents a shortest path.
    """

    #Checks whether the node exists on the graph
    def isValid(node : Node) -> bool:
        row = node.position[0]
        col = node.position[1]
        depth = node.position[2]

        if (row < 0 or row > len(grid) - 1):
            return False
        
        if (col < 0 or col > len(grid[0]) - 1):
            return False
        
        if (depth < 0 or depth > len(grid[0][0]) - 1):
            return False
        
        return True


    unvisitedNodes = queue.PriorityQueue() # The 'open set' of nodes to be visited next. Format is (key, node). The queue returns nodes in order.
    visitedNodes = set() # The 'closed set' of nodes that have been visited

    unvisitedNodes.put((0, Node(start))) # Add the first node to visit to the queue

    endNode = Node(end) # A node to test against to see if the end has been reached

    listNode = None # This will contain the end node (if found) and is used to start construction of the path efficiently

    # Loop until you reach the end (success) or there are no more nodes to visit (fail)
    while (endNode not in visitedNodes and not unvisitedNodes.empty()):

        currentNode : Node = unvisitedNodes.get()[1] # Get the node out of the (key, node) pair
 
        if (not diagonal):
            directions = [
                (-1, 0, 0), #Up
                (1, 0, 0), #Down
                (0, -1, 0), #West
                (0, 1, 0), #East
                (0, 0, -1), #South
                (0, 0, 1), #North
            ]
        else:
            directions = [
                (-1, 0, 0),   # Up
                (1, 0, 0),    # Down
                (0, -1, 0),   # West
                (0, 1, 0),    # East
                (0, 0, -1),   # South
                (0, 0, 1),    # North
                (-1, -1, 0),  # Up-West
                (-1, 1, 0),   # Up-East
                (1, -1, 0),   # Down-West
                (1, 1, 0),    # Down-East
                (-1, 0, -1),  # Up-South
                (-1, 0, 1),   # Up-North
                (1, 0, -1),   # Down-South
                (1, 0, 1),    # Down-North
                (0, -1, -1),  # West-South
                (0, -1, 1),   # West-North
                (0, 1, -1),   # East-South
                (0, 1, 1),    # East-North
                (-1, -1, -1), # Up-West-South
                (-1, -1, 1),  # Up-West-North
                (-1, 1, -1),  # Up-East-South
                (-1, 1, 1),   # Up-East-North
                (1, -1, -1),  # Down-West-South
                (1, -1, 1),   # Down-West-North
                (1, 1, -1),   # Down-East-South
                (1, 1, 1)     # Down-East-North
            ]

        # Check each neighbor of the node. Update its distance value and add it to Unvisited Nodes.
        for direction in directions:
            # Coordinates of the current node
            currentRow = currentNode.position[0]
            currentColumn = currentNode.position[1]
            currentDepth = currentNode.position[2]

            # Coordinates of this neighbor
            neighborRow = currentRow + direction[0]
            neighborColumn = currentColumn + direction[1]
            neighborDepth = currentDepth + direction[2]

            neighborNode = Node((neighborRow, neighborColumn, neighborDepth), currentNode)

            # Is the node in the grid
            if (not isValid(neighborNode)):
                continue

            # Is the node open space
            if (grid[neighborRow][neighborColumn][neighborDepth] == 1):
                continue

            # Is the node yet to be checked
            if (neighborNode in visitedNodes):
                continue

            neighborNode.distance = currentNode.distance + math.sqrt((neighborRow - currentRow)**2 + 
                                                                     (neighborColumn - currentColumn)**2 + 
                                                                     (neighborDepth - currentDepth)**2)
            
            neighborNode.heuristic = math.sqrt((neighborRow - end[0])**2 + 
                                               (neighborColumn - end[1])**2 + 
                                               (neighborDepth - end[2])**2)
            
            key = neighborNode.distance + neighborNode.heuristic # The priority to choose the next node to evaluate (choose the closest to the goal)

            unvisitedNodes.put((key, neighborNode)) # Add this node to the Unvisited Nodes

        visitedNodes.add(currentNode) #Once its neighbors have been checked, add it to the Visited Nodes

        if (currentNode == endNode):
            listNode = currentNode # Keep track of the end node (specifically with the right parent pointer) to construct a list

    #End of the while loop (All needed nodes processed)

    if (endNode not in visitedNodes):
        print("No Path")
        return []
    
    else:
        pathList = []
        # Generate a path based on each node's parent. Start with the end until the goal is reached.
        # Each node in the Visited Set has a parent that represents the shortest path
        # Because a node is only added once to the set and it is added by order of priority (shortest distance)

        startNode = Node(start) # To compare against for creating the list
        
        # (List node has been set to the end node)
        while (listNode != startNode):
            pathList.insert(0, listNode.position) # Put this position into the list (reverse order)
            listNode = listNode.parent # Move to the parent

        pathList.insert(0, startNode.position)


        return pathList