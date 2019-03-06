# Python code to implement Conway's Game Of Life 
import argparse 

import numpy as np
from numpy.random import rand

import matplotlib.pyplot as plt  
import matplotlib.animation as animation
import matplotlib.patches as mpatches
  
# setting up the values for the grid
S = 0
I = 1
R = 0.4
vals = np.array([S, I, R])
label = ['S', 'I', 'R']

p1_range = 1.0
def randomGrid(N): 
  
    """returns a grid of NxN random values"""
    return np.random.choice(vals, N*N, p=[0.7, 0.2, 0.1]).reshape(N, N)


def analyse(p1_range):
    for tempstep in tqdm(range(self.temp_point), desc='Temp point', unit='sweepstep'):
        self.beta = 1.0/self.T[tempstep]
        
        for i in tqdm(range(self.equistep), desc='Equilibrate sweep', leave=False, unit='sweep'):         # equilibrate
            self.montecarlo()                                                                             # Monte Carlo moves

        for j in tqdm(range(self.calcstep), desc='Measurement sweep', leave=False, unit='sweep'):         # measurement
            self.montecarlo()                                                                             # Monte Carlo moves
            if (j%10 == 0):
                self.getStats(int(j/10))                    

        self.E[tempstep] = np.sum(self.energy)/(self.factor)
        self.M[tempstep] = np.sum(self.magnet)/(self.factor)
        self.C[tempstep] = (np.sum(self.energy2)/(self.factor) - np.sum(self.energy)*np.sum(self.energy)/(self.factor)/(self.factor))*self.beta*self.beta/self.N
        self.X[tempstep] = (np.sum(self.magnet2)/(self.factor) - np.sum(self.magnet)*np.sum(self.magnet)/(self.factor)/(self.factor))*self.beta/self.N
        self.C_err[tempstep] = self.calcHeatError(tempstep)

        self.reinitialise_properties()
    #self.dumpData() 
    self.plotStats()
    self.reinitialise() 

def update(frameNum, img, grid, N, p): 
  
    # copy grid since we require 8 neighbors  
    # for calculation and we go line by line  
    newGrid = grid.copy() 
    for i in range(N): 
        for j in range(N): 
  
            # compute 8-neghbor sum 
            # using toroidal boundary conditions - x and y wrap around  
            # so that the simulaton takes place on a toroidal surface.
                                
            total_infect = (int(grid[i, (j-1)%N]) + int(grid[i, (j+1)%N]) + 
                            int(grid[(i-1)%N, j]) + int(grid[(i+1)%N, j]) +
                            int(grid[(i-1)%N, (j-1)%N]) + int(grid[(i-1)%N, (j+1)%N]) + 
                            int(grid[(i+1)%N, (j-1)%N]) + int(grid[(i+1)%N, (j+1)%N]))/1
  
            # apply Conway's rules 
            if grid[i, j]  == 0.0: 
                if total_infect > 0 :
                    if rand() < p[0]:
                        newGrid[i, j] = 1
            
            elif grid[i, j]  == 1:
                if rand() < p[1]:
                    newGrid[i, j] = 0.4
            
            elif grid[i, j]  == 0.4:
                if rand() < p[2]:
                    newGrid[i, j] = 0.0
  
    # update data 
    img.set_data(newGrid) 
    grid[:] = newGrid[:] 
    return img, 
  
# main() function 
def main(): 
  
    # Command line args are in sys.argv[1], sys.argv[2] .. 
    # sys.argv[0] is the script name itself and can be ignored 
    # parse arguments 
    parser = argparse.ArgumentParser(description="Runs Conway's Game of Life simulation.") 
  
    # add arguments 
    parser.add_argument('--grid-size', dest='N', required=False)
    parser.add_argument('--interval', dest='interval', required=False)
    parser.add_argument('--p1', dest='p1', required=False)
    parser.add_argument('--p2', dest='p2', required=False)
    parser.add_argument('--p3', dest='p3', required=False)
    args = parser.parse_args() 
      
    # set grid size 
    N = 100
    if args.N and int(args.N) > 8: 
        N = int(args.N) 
    
    # set p1, p2 and p3
    # p1, p2, p3 = 0.5, 0.3, 0.2    
    p1, p2, p3 = 0.05, 0.5, 0.05

    # print(args.p1)
    if args.p1: 
        p1 = float(args.p1)
    if args.p2:
        p2 = float(args.p2)
    if args.p3: 
        p3 = float(args.p3)

    p = [p1, p2, p3]

    # set animation update interval 
    updateInterval = 50
    if args.interval: 
        updateInterval = int(args.interval) 
  
    # declare grid 
    grid = np.array([]) 

    # populate grid with random on/off - 
    grid = randomGrid(N) 

    # set up animation 
    fig, ax = plt.subplots() 
    img = ax.imshow(grid, interpolation='nearest', cmap='viridis')
    
    colors = [ img.cmap(img.norm(value)) for value in vals]

    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=label[i]) ) for i in range(len(vals)) ]

    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

    ani = animation.FuncAnimation(fig, update, fargs=(img, grid, N, p ), 
                                  frames = 10, 
                                  interval=updateInterval, 
                                  save_count=50)
    plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)  
    plt.show() 
  
# call main 
if __name__ == '__main__': 
    main() 