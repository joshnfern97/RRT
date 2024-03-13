# Importing necessary libraries
import pil
import sys
sys.modules['PIL'] = pil
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import time

class Vertex:
    def __init__(self, position):
        self.position = position
    
    def add_closest_vertex(self, vertex):
        self.vertex = vertex


class RRT():
    def __init__(self, origin, goal):
        self.obstacles = self.get_obstacles()
        self.obstacle_radius = 5
        self.obstacle_number = len(self.obstacles[:,0])
        self.origin = origin
        self.goal = goal

        #Visualization Parameters
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.scatter0 = self.ax.scatter([], [], color='green', marker='x')
        self.scatter1 = self.ax.scatter([], [], color='green', marker='o')
        self.scatter2 = self.ax.scatter([], [], color='red', marker='o')
        self.scatter3 = self.ax.scatter([], [], color='black', marker='o')
        self.direct_line, = self.ax.plot([], [], color='black', linestyle='dashed')
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
        x_point (float): number between 1 and 100
        y_point (float): number between 1 and 100
        '''
        x_point = random.uniform(1,100)
        y_point = random.uniform(1,100)
        return np.array([x_point, y_point])

    def check_point_feasibility(self,point):
        '''
        Checks to see if a point is too close to an obstacle
        Parameters:
        points (float of np array): point in 2D space
        Return:
        feasible (Bool): True or False if point is feasible or not
        '''
        distance_to_obstacles = np.sqrt( (self.obstacles[:,0] - point[0])**2 + (self.obstacles[:,1] - point[1])**2 )        
        distance_to_closest_obstacle = np.min(distance_to_obstacles)
        print("DISTANCE TO CLOSEST OBSTACLE: ", distance_to_closest_obstacle)
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
    

        

    #THE REST OF THE FUNCTIONS ARE JUST USED FOR VISUALIZATIONS

    def visualize_random_point(self, point):
        self.scatter3.set_offsets(point)
        plt.xlim(self.xlim_min, self.xlim_max)
        plt.ylim(self.ylim_min, self.ylim_max)
        plt.draw()
        plt.pause(0.01)  # Pause to allow the plot to update

    def visualize_start(self):
        self.scatter0.set_offsets(self.goal.position)
        self.scatter1.set_offsets(self.origin.position)
        self.scatter2.set_offsets(self.obstacles)
        for i in range(len(self.obstacles[:,0])):
            circle = plt.Circle((self.obstacles[i,0], self.obstacles[i,1]), self.obstacle_radius, color = 'r',fill = False)
            self.ax.add_artist(circle)
        # self.visualize_obstacles(obstacles)
            
    def visualize_direct_path(self, sample_point):
        '''
        Used to see direct lines from origin to sample and sample to goal 
        '''
        #Columns represent individual datasets
        #First column is x position of origin and sample point (dataset 1) and second column is x posiiton of sample and goal (dataset 2)
        x = np.array([[self.origin.position[0],  sample_point.position[0]  ],
                      [sample_point.position[0], self.goal.position[0]] ])
        #First column is y position of origin and sample point (dataset 1) and second column is y posiiton of sample and goal (dataset 2)
        y = np.array([[self.origin.position[1],  sample_point.position[1] ], 
                      [sample_point.position[1], self.goal.position[1] ]])
        self.direct_line.set_data(x,y)
        plt.draw()
        plt.pause(0.01)


def main():
    
    
    #Static goal position  
    goal_position = np.array([100,100])
    goal_vertex = Vertex(goal_position)
    #Static origin position
    origin_position = np.array([0,0])
    origin_vertex = Vertex(origin_position)

    #Create graph were all vertices will be stored
    GRAPH = []
    GRAPH.append(origin_vertex)
    
    #Define RRT Object
    rrt = RRT(origin_vertex, goal_vertex)
    rrt.visualize_start()

    i = 0

    while i < 10:

        new_point = rrt.generate_rand_point()
        rrt.visualize_random_point(new_point)
        point_feasible = rrt.check_point_feasibility(new_point)
        new_point_vertex = Vertex(new_point)
        path_origin_point_feasible = rrt.check_path_feasibility(origin_vertex, new_point_vertex)
        path_point_goal_feasible = rrt.check_path_feasibility(new_point_vertex, goal_vertex)
        rrt.visualize_direct_path(new_point_vertex)
        if point_feasible:
            print("POINT feasible")
        else:
            print("POINT not feasible")
        if path_origin_point_feasible:
            print("PATH ORIGIN -> POINT feasible")
        else:
            print("PATH ORIGIN -> POINT not feasible")
        if path_point_goal_feasible:
            print("PATH POINT -> GOAL feasible")
        else:
            print("PATH POINT -> GOAL not feasible")
        
        print(" ")
        print("----------------------------------")
        print(" ")
        time.sleep(5)
        i+=1


if __name__ == "__main__":
    # Calling the main function if the script is executed directly
    main()