# Importing necessary libraries
import pil
import sys
sys.modules['PIL'] = pil
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np

####################################
#           Vertex                 #
####################################
class Vertex:
    '''
    Object used to define the vertices of RRT
    '''
    def __init__(self, position):
        self.position = position
        self.closest_vertex = None
        self.closest_vertex_i = None
        self.origin = False
        self.goal = False
    
    def add_closest_vertex(self, closest_vertex):
        '''
        This adds the closest vertex to the current vertex's member variables
        Parameter:
        closest_vertex (Vertex): closest vertex to current vertex
        '''
        self.closest_vertex = closest_vertex

    def add_closest_vertex_i(self, closest_vertex_i):
        '''
        This adds the index of a vertex within a graph to the current vertex's member variables
        Parameter:
        closest_vertex_i (int): the index within a graph of the vertex closest to the current vertex
        '''
        self.closest_vertex_i = closest_vertex_i

    def set_origin(self):
        '''
        This sets a vertex as the origin
        '''
        self.origin = True

    def set_goal(self):
        '''
        This sets the vertex as the goal
        '''
        self.goal = True

#############################################
#                 RRT                       #
#############################################
class RRT():
    def __init__(self, origin, goal):
        self.obstacles = self.get_obstacles() #it gets the obstacles from the provided csv file
        self.obstacle_radius = 5 #radius of the obstacles
        self.obstacle_number = len(self.obstacles[:,0]) 
        self.origin = origin #sets the origin of the graph
        self.goal = goal #sets the goal of the graph

        #Visualization Parameters
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.scatter0 = self.ax.scatter([], [], color='green', marker='x') #plots the goal position
        self.scatter1 = self.ax.scatter([], [], color='green', marker='o') #plots the origin position
        self.scatter2 = self.ax.scatter([], [], color='red', marker='o') #plots the obstacle centers
        self.scatter3 = self.ax.scatter([], [], color='black', marker='o') #plots the random sample point
        self.direct_line, = self.ax.plot([], [], color='black', linestyle='dashed',linewidth=.1) #plots the branches
        self.path_line, = self.ax.plot([], [], color='green', linewidth=5) #plots the final path
        #Set the plot limits
        self.xlim_min = -10
        self.xlim_max = 110
        self.ylim_min = -10
        self.ylim_max = 110

    def get_obstacles(self):
        '''
        reads in the obstacles and return a N by 2 np array
        '''
        df = pd.read_csv('obstacles.csv')
        obstacles = df.to_numpy()
        return obstacles


    def generate_rand_point(self):
        '''
        Generates a random point to be used as a potential node
        Parametes: None
        Return: 
        random_point (np array): a 2d point 
        '''
        x_point = random.uniform(1,100)
        y_point = random.uniform(1,100)
        return np.array([x_point, y_point])

    def check_point_feasibility(self,point):
        '''
        Checks to see if a point is too close to an obstacle
        Parameters:
        point (np array): point in 2D space
        Return:
        feasible (Bool): True or False if point is feasible or not
        '''
        #find euclidean distance from a point and the center of the obstacles
        distance_to_obstacles = np.sqrt( (self.obstacles[:,0] - point[0])**2 + (self.obstacles[:,1] - point[1])**2 ) 
        #find distance to closest obstacle 
        distance_to_closest_obstacle = np.min(distance_to_obstacles)
        print("DISTANCE TO CLOSEST OBSTACLE: ", distance_to_closest_obstacle)
        #check and see if the distance is within the obstacle radius (collision)
        if distance_to_closest_obstacle < self.obstacle_radius:
            return False
        else:
            return True

    def check_path_feasibility(self, start_point, end_point):
        '''
        checks the feasibility of a path between two points
        Parameters:
        start_point (Vertex): starting vertex of path
        end_point (Vertex): end vertex of path
        Returns:
        feasible (Bool): True or False if path is feasible or not
        '''
        path = self.generate_path(start_point, end_point)
        #seperate the path x and y coordinates
        path_x_coordinates = path[:,0]
        path_y_coordinates = path[:,1]
        #make a matrix that repeats the x and y coordinates as many tiles as there are obstacles
        path_x_rollout = np.tile(path_x_coordinates, ( self.obstacle_number, 1 ))
        path_y_rollout = np.tile(path_y_coordinates, ( self.obstacle_number, 1 ))

        #slice the obstacle vectors and ensure that it is a Nx1 array
        x_obstacles = np.reshape(self.obstacles[:,0], (len(self.obstacles[:,0]), 1))
        y_obstacles = np.reshape(self.obstacles[:,1], (len(self.obstacles[:,1]), 1))

        #get x and y distance of each part of path to each obstacle
        distance_x_rollout = x_obstacles - path_x_rollout
        distance_y_rollout = y_obstacles - path_y_rollout 

        #Get the eulcidean distance from each point on path to each obstacles center
        distance_to_obstacle_center = np.sqrt(distance_x_rollout**2 + distance_y_rollout**2)
        #Get distance from each point on path to each obstacles edge
        distance_to_obstacle_edge = distance_to_obstacle_center - self.obstacle_radius
        #Find smallest distance on path to each edge
        smallest_distance_to_edge = np.min(distance_to_obstacle_edge)

        #A negative value represents a path that intersects with an obstacle
        if smallest_distance_to_edge <= 0:
            feasible = False
        else:
            feasible = True
        return feasible



    def generate_path(self, start_point, end_point):
        '''
        Generates a straight path between two vertices
        Parameters:
        start_point (Vertex): starting vertex of path
        end_point (Vertex): end vertex of path
        Returns:
        path (np array): Nx2 (x and y) path connecting vertices
        '''
        x_path = np.linspace(start_point.position[0], end_point.position[0], 100)
        x_path = np.reshape(x_path, (len(x_path), 1))

        y_path = np.linspace(start_point.position[1], end_point.position[1], 100)
        y_path = np.reshape(y_path, (len(y_path), 1))

        path = np.hstack((x_path, y_path))

        return path
    
    def extend(self, Graph, sample_vertex):
        '''
        This uses the Graph and the newest sample_vertex to extend the graph
        Parameters:
        Graph (list of Verices): list of all the Vertices added to the graph
        sample_vertex (Vertex): Newest random sample (already a feasible point)
        Returns:
        Graph (lis of Vertices): updated list of the Vertices within the graph
        '''

        distance_array =[]
        infeasible_count = 0
        #Go through each vertex in graph and the path feasibility to each sample and distance to each sample
        for vertex in Graph:
            #check path feasibility
            feasible = self.check_path_feasibility(vertex, sample_vertex)
            #check distance
            distance = np.sqrt( (vertex.position[0] - sample_vertex.position[0])**2 + (vertex.position[1] - sample_vertex.position[1])**2 )
            #if path is feasible, just add the distance to the distance_array; if not, then add infinity (cannot be reached)
            if feasible:
                distance_array.append(distance)
            else:
                distance_array.append(np.inf)
                infeasible_count+=1

        #The point may be feasible, but there may be no feasible path to it given the current graph. This statement accounts for this
        if infeasible_count == len(Graph):
            print("NO PATH TO VERTEX") 
            return Graph
        #If there is a feasible path, then add the closest vertex and location within the graph to the member variable of the current vertex
        else:
            i = np.argmin(distance_array)
            sample_vertex.add_closest_vertex(Graph[i])
            sample_vertex.add_closest_vertex_i(i)
            
            #For visualization: plots the branches
            new_xbranch = np.array([[sample_vertex.closest_vertex.position[0]],
                                    [sample_vertex.position[0]]])
            new_ybranch = np.array([[sample_vertex.closest_vertex.position[1]],
                                    [sample_vertex.position[1]]])
            
            self.ax.plot(new_xbranch, new_ybranch, color='black', linestyle='dashed')
            
            #Add the vertex to the graph
            Graph.append(sample_vertex)
            return Graph
        
    def check_finished(self, Graph):
        '''
        Checks to see if a sample can reach the goal directly
        Parameters:
        new_point_vertex (Vertex): the newest sampled vertex
        '''
        graph_length = len(Graph)-1 #how many vertices are in graph
        new_point_vertex = Graph[graph_length] #gets the most recently added vertex
        finished = self.check_path_feasibility(new_point_vertex, self.goal) #checks to see if the most recently added vertex can reach goal

        #if the points within the current graph can reach the goal
        if finished:
            #add the vertex and location within the graph to member variables of the goal vertex
            self.goal.add_closest_vertex(new_point_vertex)
            self.goal.add_closest_vertex_i(graph_length)
            #add goal vertex to the graph
            Graph.append(self.goal)
            #For visualization: plots branch to goal
            new_xbranch = np.array([[self.goal.closest_vertex.position[0]],
                                    [self.goal.position[0]]])
            new_ybranch = np.array([[self.goal.closest_vertex.position[1]],
                                    [self.goal.position[1]]])
            self.ax.plot(new_xbranch, new_ybranch, color='black', linestyle='dashed')
        return finished

    def produce_final_path(self, GRAPH):
        '''
        This produces the final path to goal
        Parameters:
        GRAPH (list of Vertices): list of all the vertices in the graph
        Returns:
        final_path (np array): a 2xN path to goal
        '''
        back_to_start = False
        graph_i = len(GRAPH)-1
        path_x = []
        path_y = []
        while back_to_start == False:
            path_x.append(GRAPH[graph_i].position[0])
            path_y.append(GRAPH[graph_i].position[1])
            graph_i = GRAPH[graph_i].closest_vertex_i
            back_to_start = GRAPH[graph_i].origin
        #Append Origin    
        path_x.append(GRAPH[graph_i].position[0])
        path_y.append(GRAPH[graph_i].position[1])
        
        #visualize path to goal
        self.visualize_path(path_x, path_y)
        #The list is in reverse order so flip it
        path_x_array = np.flip(np.array(path_x))
        path_y_array = np.flip(np.array(path_y))

        #make the path a 2xN path
        path_x_array = np.reshape(path_x_array, (len(path_x_array), 1))
        path_y_array = np.reshape(path_y_array, (len(path_y_array), 1))
        final_path = np.hstack((path_x_array, path_y_array))
        return final_path    
        

    #THE REST OF THE FUNCTIONS ARE JUST USED FOR VISUALIZATIONS

    def visualize_random_point(self, point):
        '''
        visualizes the random sample points
        Parameters (np array): position of random sample
        '''
        self.scatter3.set_offsets(point)
        plt.xlim(self.xlim_min, self.xlim_max)
        plt.ylim(self.ylim_min, self.ylim_max)
        plt.draw()
        plt.pause(0.01)  # Pause to allow the plot to update

    def visualize_start(self):
        '''
        visualizes the goal, origin, and all the obstacles (with a radius of obstacle_radius)
        '''
        self.scatter0.set_offsets(self.goal.position)
        self.scatter1.set_offsets(self.origin.position)
        self.scatter2.set_offsets(self.obstacles)
        for i in range(len(self.obstacles[:,0])):
            circle = plt.Circle((self.obstacles[i,0], self.obstacles[i,1]), self.obstacle_radius, color = 'r',fill = False)
            self.ax.add_artist(circle)


    def visualize_path(self, x_path, y_path):
        '''
        Used to visualize path to goal
        '''
        x_path_array = np.array([x_path])
        y_path_array = np.array([y_path])
        self.path_line.set_data(x_path_array,y_path_array)
        plt.draw()
        plt.pause(0.01)

    


def main():
    #goal position  
    goal_position = np.array([100,100])
    goal_vertex = Vertex(goal_position)
    goal_vertex.set_goal()

    #origin position
    origin_position = np.array([0,0])
    origin_vertex = Vertex(origin_position)
    origin_vertex.set_origin()

    #Create graph were all vertices will be stored
    GRAPH = []
    GRAPH.append(origin_vertex)
    
    #Define RRT Object
    rrt = RRT(origin_vertex, goal_vertex)
    rrt.visualize_start()

    
    finished_search = False
    #Keep looping until the search is finished
    while finished_search == False:
        
        #generate a new point to sample
        new_point = rrt.generate_rand_point()
        new_point_vertex = Vertex(new_point)

        #check if the new point is feasible
        point_feasible = rrt.check_point_feasibility(new_point)
        
        #if the point is feasible try and extend the graph
        if point_feasible:
            GRAPH = rrt.extend(GRAPH, new_point_vertex)
            #check and see if the search is complete
            finished_search = rrt.check_finished(GRAPH)

        
        rrt.visualize_random_point(new_point)
        
        print(" ")
        print("----------------------------------")
        print(" ")
        # Just used to step through time (comment out if you want it done automatically)
        # input("Press the Enter key to continue")

    #Trace the path back  to the start
    final_path = rrt.produce_final_path(GRAPH)
    print("FINAL PATH:")
    print(final_path)
    
    input("Press the Enter key exit")

if __name__ == "__main__":
    # Calling the main function if the script is executed directly
    main()