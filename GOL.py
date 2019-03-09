# Python code to implement Conway's Game Of Life 
import argparse 
import numpy as np 
import matplotlib.pyplot as plt  
import matplotlib.animation as animation
import datetime

# setting up the values for the grid 
ON = 255
OFF = 0
vals = [ON, OFF] 
  
def randomGrid(N): 
  
    """returns a grid of NxN random values"""
    return np.random.choice(vals, N*N, p=[0.2, 0.8]).reshape(N, N) 

def addBlinker(i, j, grid): 
  
    """adds a blinker with top left cell at (i, j)"""
    blinker = np.array([[0,    255, 0],  
                       [0,  255, 0],  
                       [0,  255, 0]]) 
    grid[i:i+3, j:j+3] = blinker 
  
def addGlider(i, j, grid): 
  
    """adds a glider with top left cell at (i, j)"""
    glider = np.array([[0,    0, 255],  
                       [255,  0, 255],  
                       [0,  255, 255]]) 
    grid[i:i+3, j:j+3] = glider 

def calcSpeed(N, grid, compute):
    x, y = np.where(grid == 255)
    r_cm = np.sqrt((np.sum(x)/len(x))**2 + (np.sum(y)/len(y))**2)
    compute['com'].append(r_cm)
    #lebih satu baru boleh compute difference
    if len(compute['com']) > 1:
        # difference negative maksudnya dia tgh cross boundaries
        if (compute['com'][-1] - compute['com'][-2]) < 0:
            return compute
        else:
            # print(len(compute['com']))
            #sebab position sama lepas 4 kali, amik plot position bila lepas 4kali ni baru comparable
            if len(compute['com'])%4 == 0:
                # print('------------------')
                if len(compute['com'])%8 == 0:
                #     print('%%%%%%%%%%%%%%%%%%%%%%')
                    return compute
                else:
                #     print('$$$$$$$$$$$$$$$')
                    compute['diff com'].append(compute['com'][-1] - compute['com'][-2])
                    compute['time'].append( datetime.datetime.now() )
                    return compute
            else:
                return compute
    else:
        return compute

# The function to call each time the plot is updated
def updatePlot(i, grid, N, compute):
    newGrid = grid.copy() 
    for i in range(N): 
        for j in range(N): 
  
            # compute 8-neighbour sum 
            # using toroidal boundary conditions - x and y wrap around  
            # so that the simulaton takes place on a toroidal surface. 
            total = int((grid[i, (j-1)%N] + grid[i, (j+1)%N] + 
                         grid[(i-1)%N, j] + grid[(i+1)%N, j] + 
                         grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, (j+1)%N] + 
                         grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, (j+1)%N])/255)
                         
            # apply Conway's rules 
            if grid[i, j]  == ON: 
                if (total < 2) or (total > 3): 
                    newGrid[i, j] = OFF 
            else: 
                if total == 3: 
                    newGrid[i, j] = ON 
  
    # update data

    calcSpeed(N, newGrid, compute) 
    grid[:] = newGrid[:]

    # Clear the old plot
    plt.clf()

    # Make the new plot
    # plt.subplot(2, 1, 1)
    ax0 = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
    ax0.imshow(grid, interpolation='nearest', cmap='winter')
    ax0.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False) 

    # plt.subplot(2, 1, 2)
    ax1 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    ax1.set_xlabel(r'$Real\/Time$')
    # ax1.set_ylabel(r'$Velocity$')
    ax1.set_title('Difference in Centre of Mass Position', fontsize='x-large')
    ax1.plot(compute['time'], compute['diff com'], linewidth=0.5, color='black')
    ax1.scatter(compute['time'], compute['diff com'], marker='o', s=10, color='ForestGreen') 
    ax1.set_ylim((0, 1))

    plt.subplots_adjust(hspace=0.6)

# main() function 
def main(): 
  
    # Command line args are in sys.argv[1], sys.argv[2] .. 
    # sys.argv[0] is the script name itself and can be ignored 
    # parse arguments 
    parser = argparse.ArgumentParser(description="Runs Conway's Game of Life simulation.") 
  
    # add arguments 
    parser.add_argument('--grid-size', dest='N', required=False)
    parser.add_argument('--mov-file', dest='movfile', required=False)
    parser.add_argument('--interval', dest='interval', required=False)
    parser.add_argument('--glider', action='store_true', required=False)
    parser.add_argument('--blinker', action='store_true', required=False)
    args = parser.parse_args() 

    compute              = {}
    compute['com']       = []
    compute['clock']     = []
    compute['row']       = []
    compute['col']       = []
    compute['diff com']  = []
    compute['time']      = []

    # set grid size 
    N = 50
    if args.N and int(args.N) > 8: 
        N = int(args.N) 
          
    # set animation update interval 
    updateInterval = 50
    if args.interval: 
        updateInterval = int(args.interval) 

    # declare grid 
    grid = np.array([])

    # check if "glider" demo flag is specified 
    if args.glider: 
        grid = np.zeros(N*N).reshape(N, N) 
        addGlider(1, 1, grid) 
        calcSpeed(N, grid, compute)

    elif args.blinker: 
        grid = np.zeros(N*N).reshape(N, N) 
        addBlinker(1, 1, grid)

    # elif args.gosper: 
    #     grid = np.zeros(N*N).reshape(N, N) 
    #     addGosperGliderGun(10, 10, grid) 
  
    else:   # populate grid with random on/off - 
            # more off than on 
        grid = randomGrid(N) 
  
    # set up animation
    plotFigure = plt.figure() 
    ani = animation.FuncAnimation(plotFigure, updatePlot, fargs=(grid, N, compute, ), frames = 10, interval=updateInterval, save_count=50)
    plt.show()
# call main 
if __name__ == '__main__': 
    main() 