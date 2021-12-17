#! /usr/bin/python

# First developed in the context of the subject 'Complementos de Matematica I'
# by Ines Cipullo and Katherine Sullivan

import math
import argparse
import matplotlib.pyplot as plt
import numpy as np

DIMENSION = 1000 # graphic range
EPSILON = 0.05 # minimum distance between two vertices


def read_graph_file(file_path):
    '''
    Reads a graph from a file and returns its representation as a list.
    Example file: 
        3
        A
        B
        C
        A B
        B C
        C B
    Example return: 
        (['A','B','C'],[('A','B'),('B','C'),('C','B')])
    '''
    vertices = []
    edges = []
    graph = (vertices, edges)
    with open(file_path, 'r') as f:
        n = int(f.readline())
        while (n != 0):
            vertex = f.readline().rstrip()
            vertices.append(vertex)
            n-=1
        for line in f:
            edge = line.split()
            edges.append(tuple(edge))
    return graph


# Returns a list with N random numbers from 0 to 1000
def random_coordinates(N):
    return np.random.rand(N) * DIMENSION

# Initializes accumulators in cero. N is the vertices set.
def initialize_accumulators(N):
    accum_x = {v: 0 for v in N}
    accum_y = {v: 0 for v in N}
    return accum_x, accum_y

# Calculates force of attraction
def f_attraction(d,k):
    return (d**2)/k

# Calculates force of repulsion
def f_repultion(d,k):
    return -(k**2)/d

# Calculates force of gravity
def f_gravity(d,k):
    return 0.1 * f_attraction(d,k)

# Handles collisions between vertices
def avoid_collisions (i, x_coordinates, y_coordinates):
    changed = []

    for j in range(i):
        distance = math.sqrt((x_coordinates[i] - x_coordinates[j])**2 + (y_coordinates[i] - y_coordinates[j])**2)
        if distance < EPSILON:
            changed += [j]
            direction_vector = (np.random.rand(), np.random.rand())
            direction_vector_op = (-direction_vector[0], -direction_vector[1])
            x_coordinates[i]= x_coordinates[i] * direction_vector[0]
            y_coordinates[i]= y_coordinates[i] * direction_vector[1]
            x_coordinates[j]= x_coordinates[j] * direction_vector_op[0]
            y_coordinates[j]= y_coordinates[j] * direction_vector_op[1]

    if changed == [] :
        return
    else:
        for elem in changed: 
            avoid_collisions(elem, x_coordinates, y_coordinates)
        avoid_collisions(i, x_coordinates, y_coordinates)

    return



class LayoutGraph:

    def __init__(self, graph, iters, refresh, c1, c2, temperature, c_temp, verbose=False, color_graph=False):
        """
        Parameters:
        graph: graph in list format
        iters: number of iterations to perform
        refresh: how many iterations before refreshing. If value is set to 0, the graph only appears at the end.
        c1: repusion constant
        c2: attraction constant
        temperature: initial temperature
        c_temp: temperature reduction constant
        verbose: if on, activates comments
        color_graph: if on, graph colored
        """

        # Save the graph
        self.graph = graph

        # Save the options
        self.iters = iters
        self.refresh = refresh
        self.c1 = c1
        self.c2 = c2
        self.temperature = temperature
        self.c_temp = c_temp
        self.verbose = verbose
        self.color_graph = color_graph


    def draws_graph(self, x_coordinates, y_coordinates, color):
        n_vertices = len(self.graph[0])
        if (color):
            plt.scatter(x_coordinates,y_coordinates)
        else:
            plt.scatter(x_coordinates,y_coordinates, color='00')
        for i in range(n_vertices):
            x_edges = []
            y_edges = []
            for j in range(n_vertices):
                v1 = self.graph[0][i]
                v2 = self.graph[0][j]
                if (v1,v2) in self.graph[1]:
                    x_edges.append(x_coordinates[i])
                    x_edges.append(x_coordinates[j])
                    y_edges.append(y_coordinates[i])
                    y_edges.append(y_coordinates[j])
                    if (color):
                        plt.plot(x_edges,y_edges)
                    else:
                        plt.plot(x_edges,y_edges,color='00')


    def layout(self):
        """
        Aplies the Fruchtermann-Reingold's algorithm to obtain (and show) a layout
        """
        # If verbose is on, verbose print functions as print function, else it does nothing
        verboseprint = print if self.verbose else lambda *a: None
        
        n_vertices = len(self.graph[0])
        x_coordinates = random_coordinates(n_vertices)
        y_coordinates = random_coordinates(n_vertices)      
        # Dispersion constants of the nodes in the graph
        kr = self.c1 * math.sqrt((DIMENSION*DIMENSION) / n_vertices)
        ka = self.c2 * math.sqrt((DIMENSION*DIMENSION) / n_vertices)

        center = (DIMENSION/2,DIMENSION/2)
        
        dicc_vert_a_idx = {} # Dictionary vertices and indexes
        for i in range(n_vertices):
            dicc_vert_a_idx[self.graph[0][i]] = i
        
        # Initialize temperature
        t = self.temperature
        verboseprint("Initial temperature:",t)


        plt.show()
        for k in range(1, self.iters+1):
            verboseprint("Iteration: ",k)
            
            ### BEGIN STEP ###

            # Inirialize accumulators to zero
            accum_x, accum_y = initialize_accumulators(self.graph[0])

            # Calcute forces of attraction
            for e in self.graph[1]:
                distance = math.sqrt((x_coordinates[dicc_vert_a_idx[e[0]]] - x_coordinates[dicc_vert_a_idx[e[1]]])**2 + 
                    (y_coordinates[dicc_vert_a_idx[e[0]]] - y_coordinates[dicc_vert_a_idx[e[1]]])**2)
                mod_fa = f_attraction(distance,ka)
                fx = mod_fa * (x_coordinates[dicc_vert_a_idx[e[1]]] - x_coordinates[dicc_vert_a_idx[e[0]]]) / distance # ESTO ESTA BIEN? (EL *)
                fy = mod_fa * (y_coordinates[dicc_vert_a_idx[e[1]]] - y_coordinates[dicc_vert_a_idx[e[0]]]) / distance
                accum_x[e[0]] = accum_x[e[0]] + fx
                accum_y[e[0]] = accum_y[e[0]] + fy
                accum_x[e[1]] = accum_x[e[1]] - fx
                accum_y[e[1]] = accum_y[e[1]] - fy

            # Calculate forces of repulsion
            for i in range(n_vertices):
                for j in range(n_vertices):
                    if i != j:
                        distance = math.sqrt((x_coordinates[i] - x_coordinates[j])**2 + (y_coordinates[i] - y_coordinates[j])**2)
                        mod_fr = f_repultion(distance,kr)
                        fx = mod_fr * (x_coordinates[j] - x_coordinates[i]) / distance
                        fy = mod_fr * (y_coordinates[j] - y_coordinates[i]) / distance
                        accum_x[self.graph[0][i]] = accum_x[self.graph[0][i]] + fx
                        accum_y[self.graph[0][i]] = accum_y[self.graph[0][i]] + fy
                        accum_x[self.graph[0][j]] = accum_x[self.graph[0][j]] - fx
                        accum_y[self.graph[0][j]] = accum_y[self.graph[0][j]] - fy
            
            # Calculate forces of gravity
            for i in range(n_vertices):
                distance = math.sqrt((x_coordinates[i] - center[0])**2 + (y_coordinates[i] - center[1])**2)
                mod_fg = f_gravity(distance, ka)
                fx = mod_fg * (center[0] - x_coordinates[i]) / distance
                fy = mod_fg * (center[1] - y_coordinates[i]) / distance
                accum_x[self.graph[0][i]] = accum_x[self.graph[0][i]] + fx
                accum_y[self.graph[0][i]] = accum_y[self.graph[0][i]] + fy

            # Actualize positions
            for i in range(n_vertices):
                f = (accum_x[self.graph[0][i]],accum_y[self.graph[0][i]])
                modulo_f = math.sqrt((f[0])**2 + (f[1])**2)
                if modulo_f > t:
                    f = (f[0]/modulo_f*t, f[1]/modulo_f*t)
                    accum_x[self.graph[0][i]], accum_y[self.graph[0][i]] = f

                x_coordinates[i] = x_coordinates[i] + accum_x[self.graph[0][i]]
                y_coordinates[i] = y_coordinates[i] + accum_y[self.graph[0][i]]

                # Avoids vertices going off the window
                if x_coordinates[i] < 0:
                    x_coordinates[i] = 2
                elif x_coordinates[i] > DIMENSION:
                    x_coordinates[i] = DIMENSION - 2
                if y_coordinates[i] < 0:
                    y_coordinates[i] = 2
                elif y_coordinates[i] > DIMENSION:
                    y_coordinates[i] = DIMENSION - 2

                # We check for collisions 
                avoid_collisions(i, x_coordinates, y_coordinates)

                verboseprint("Position of vertex",self.graph[0][i],": (",x_coordinates[i],",",y_coordinates[i],")")

            # We actualize the temperature
            t = self.c_temp * t
            verboseprint("New temperature:",t)

            ### END STEP ###

            ### PLOTTING ###
            if (k % self.refresh) == 0:
                plt.axis([0,DIMENSION,0,DIMENSION])
                self.draws_graph(x_coordinates,y_coordinates, self.color_graph)
                plt.pause(0.5)
                plt.clf()

        return




def main():
    # We define the arguments we accept from command line
    parser = argparse.ArgumentParser()

    # Verbose, optional, False by default
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Shows useful information while running the program'
    )
    # Iterations, optional, 50 by default
    parser.add_argument(
        '--iters',
        type=int,
        help='Iterations to perform',
        default=50
    )
    # Initial temperature
    parser.add_argument(
        '--temp',
        type=float,
        help='Initial temperature',
        default=100.0
    )
    # File from where to read graph
    parser.add_argument(
        'file_name',
        help='File from where to read graph'
    )
    # Graph coloring, optional, not colored (black) by default
    parser.add_argument(
        '-c', '--color',
        action='store_true',
        help='The graph will appear with vertices and edges colored'
    )
    # Especifies how many iterations are to be performed before showing the graph again
    parser.add_argument(
        '-r', '--refresh',
        type=int,
        help='Especifies how many iterations are to be performed before showing the graph again',
        default=2
    )

    args = parser.parse_args()

    # We read the graph file
    graph = read_graph_file(args.file_name)

    # We create our object LayoutGraph
    layout_gr = LayoutGraph(
        graph = graph,
        iters = args.iters,
        refresh = args.refresh,
        c1 = 0.1,
        c2 = 5.0,
        temperature = args.temp,
        c_temp = 0.95,
        verbose = args.verbose,
        color_graph = args.color
    )

    # We execute the layout
    layout_gr.layout()
    return


if __name__ == '__main__':
    main()
