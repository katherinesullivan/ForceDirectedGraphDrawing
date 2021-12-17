# ForceDirectedGraphDrawing

Implementation of an algorithm based on the paper 'Graph Drawing by Force–Directed Placement', written by T. M. J. Fruchterman and E. M. Reingold.

## About the program:

The purpose of the program is to better visualize graphs.

## Requirements:

 * Python3
 * Matplotlib

## How to run the program?

The file GraphDrawing.py contains the source program, which is the one we must run.

The only madatory argument is the name of the file where the graph we want to visualize is. This file must be formatted in this way: 

    number of vertices
    name of vertex 1
    ...
    name of vertex n
    vertex1 of edge1 space vertex2 of edge1
    ...
    vertex1 of edgek space vertex2 of edgek

(Examples are provided in the folder [graphexamples](/graphexamples))

Then, there are several optional arguments:

+ Verboseness, optional, False by default: '-v', '--verbose'

+ Number of iterations, optional, 50 by default: '--iters'

+ Initial temperature, optional, 100 by default: '--temp'

+ Graph coloring, optional, not colored (black) by default: '-c', '--color'

+ How many iterations to be performed before showing the graph again: '-r', '--refresh'

