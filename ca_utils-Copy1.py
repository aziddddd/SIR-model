from __future__ import division

import numpy as np
from numpy.random import rand

import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as mpatches

from IPython.display import clear_output

from numba import jit
from tqdm import tqdm_notebook as tqdm

import os
import warnings
import time

np.set_printoptions(threshold=np.inf)

warnings.filterwarnings("ignore",category =RuntimeWarning)
warnings.filterwarnings("ignore",category =FutureWarning)

from PIL import Image
from numpy import*
import pprint
pp = pprint.PrettyPrinter(indent=4)

class SIRError(Exception):
    """ An exception class for Ising """
    pass

# Class to simulate SIR model
class SIR():

############################################### INITIALISER #####################################################
    def __init__(self, N, fraction, p, factor, equistep, calcstep):
        self.N          = N
        self.p1         = p[0]
        self.p2         = p[1]
        self.p3         = p[2]
        self.ini_S      = fraction[0]
        self.ini_I      = fraction[1]
        self.ini_R      = fraction[2]
        self.vals       = np.array([0.0, 1.0, 0.4])
        self.label      = ['S', 'I', 'R']
#         self.grid       = np.random.choice(self.vals, N*N, p=[self.ini_S, self.ini_I, self.ini_R]).reshape(N, N)

        self.equistep   = equistep
        self.calcstep   = calcstep
        self.factor     = factor

        self.infected    = np.zeros(factor)
        self.infected2   = np.zeros(factor)
        self.Ii_I       = np.zeros(factor)

        self.p1_data    = np.linspace(0.0, 1.0, N)
        self.p3_data    = np.linspace(0.0, 1.0, N)
        self.p1p3       = np.meshgrid(self.p1_data, self.p3_data)

        # self.p1_cut     = np.linspace(0.0, 0.25, N)
        self.p1_cut     = np.linspace(0.2, 0.5, N)
        self.p3_cut     = np.zeros(N) + 0.5
        self.p1p3_cut   = np.meshgrid(self.p1_cut, self.p3_cut)

        self.I          = np.zeros((N, N))
        self.var_I      = np.zeros((N, N))

        self.var_I_cut  = np.zeros(N)
        self.I_immune   = np.zeros(N)
        self.I_immune_err = np.zeros(N)
        self.f_im       = np.linspace(0.0, 1.0, N)

    def pickup_grid(self):
        temp=Image.open('../../covid19/image/Italy.png')
        temp=temp.convert('1')      # Convert to black&white
        A = array(temp)             # Creates an array, white pixels==True and black pixels==False
        new_A=empty((A.shape[0],A.shape[1]),None)    #New array with same size as A

        for i in range(len(A)):
            for j in range(len(A[i])):
                if A[i][j]==True:
                    new_A[i][j]=0
                else:
                    new_A[i][j]=2

        B = new_A.flatten()
        C = [i for i in B if i != 2]
        D = np.random.choice(self.vals, len(C), p=[self.ini_S, self.ini_I, self.ini_R])

        counter = 0
        B_filled = []
        for i in tqdm(B):
            if i == 2:
                B_filled.append(i)
            else:
                B_filled.append(D[counter])
                counter+=1

        self.width, self.height = A.shape[0], A.shape[1]
        self.grid = np.asarray(B_filled).reshape((A.shape[0], A.shape[1]))

    def reinitialise(self):
        self.grid = self.pickup_grid()
#         self.grid       = np.random.choice(self.vals, self.N*self.N, p=[self.ini_S, self.ini_I, self.ini_R]).reshape(self.N, self.N)

        self.p1p3       = np.meshgrid(np.linspace(0.0, 1.0, self.N), np.linspace(0.0, 1.0, self.N))
        self.I          = np.zeros((self.N, self.N))
        self.var_I      = np.zeros((self.N, self.N))
        self.var_I_cut  = np.zeros(self.N)
        self.I_immune   = np.zeros(self.N)
        self.I_immune_err   = np.zeros(self.N)
        self.f_im       = np.linspace(0.0, 1.0, self.N)

    def reinitialise_properties(self):
        self.infected      = np.zeros(self.factor)
        self.infected2     = np.zeros(self.factor)

        self.recovered     = np.zeros(self.factor)

        np.random.seed(0)
        self.grid = self.pickup_grid()
#         self.grid          = np.random.choice(self.vals, self.N*self.N, p=[self.ini_S, self.ini_I, self.ini_R]).reshape(self.N, self.N)

################################################### UPDATE FUNCTIONS #######################################################

    def update_anim(self):
  
        # copy grid since we require 8 neighbors  
        # for calculation and we go line by line  
        newGrid = self.grid.copy() 
        for i in range(self.height): 
            for j in range(self.width): 
                # compute 8-neghbor sum 
                # using toroidal boundary conditions - x and y wrap around  
                # so that the simulaton takes place on a toroidal surface.   

                left = int(self.grid[i, (j-1)%self.N])
                right = int(self.grid[i, (j+1)%self.N])
                top = int(self.grid[(i-1)%self.N, j])
                bottom = int(self.grid[(i+1)%self.N, j])
                top_left = int(self.grid[(i-1)%self.N, (j-1)%self.N])
                top_right = int(self.grid[(i-1)%self.N, (j+1)%self.N])
                bottom_left = int(self.grid[(i+1)%self.N, (j-1)%self.N])
                bottom_right = int(self.grid[(i+1)%self.N, (j+1)%self.N])

                accumulate = [
                    left,
                    right,
                    top,
                    bottom,
                    top_left,
                    top_right,
                    bottom_left,
                    bottom_right
                ]

                # total_infect = sum(accumulate)
                # pp.pprint(accumulate)
                total_infect = sum([i for i in accumulate if i != 2])
    
                # apply SIR's rules
                if self.grid[i, j]  == 2.0:
                    continue 

                elif self.grid[i, j]  == 0.0: 
                    if total_infect > 0 :
                        if rand() < self.p1:
                            newGrid[i, j] = 1
                
                elif self.grid[i, j]  == 1:
                    if rand() < self.p2:
                        newGrid[i, j] = 0.4
                
                elif self.grid[i, j]  == 0.4:
                    if rand() < self.p3:
                        newGrid[i, j] = 0.0
  
        # update data 
        self.grid[:] = newGrid[:]

    def update(self, idx1, idx2):

        # copy grid since we require 8 neighbors  
        # for calculation and we go line by line  
        newGrid = self.grid.copy() 
        for i in range(self.N): 
            for j in range(self.N): 
                # compute 8-neghbor sum 
                # using toroidal boundary conditions - x and y wrap around  
                # so that the simulaton takes place on a toroidal surface.                                    

                left = int(self.grid[i, (j-1)])
                right = int(self.grid[i, (j+1)])
                top = int(self.grid[(i-1), j])
                bottom = int(self.grid[(i+1), j])
                top_left = int(self.grid[(i-1), (j-1)])
                top_right = int(self.grid[(i-1), (j+1)])
                bottom_left = int(self.grid[(i+1), (j-1)])
                bottom_right = int(self.grid[(i+1), (j+1)])

                accumulate = [
                    left,
                    right,
                    top,
                    bottom,
                    top_left,
                    top_right,
                    bottom_left,
                    bottom_right
                ]

                # total_infect = sum(accumulate)
                total_infect = sum([i for i in accumulate if i != 2])

                if self.grid[i, j]  == 0.0:
                    if total_infect > 0 :
                        if rand() < self.p1p3[0][idx1][idx2]:
                            newGrid[i, j] = 1
                
                elif self.grid[i, j]  == 1.0:
                    if rand() < 0.5:
                        newGrid[i, j] = 0.4
                
                elif self.grid[i, j]  == 0.4:
                    if rand() < self.p1p3[1][idx1][idx2]:
                        newGrid[i, j] = 0.0

        # update data 
        self.grid[:] = newGrid[:]

################################################### VISUALISATION #####################################################

    def DynamicPlot(self, numstep):
        fig, ax = plt.subplots() 

        img = ax.imshow(self.grid, interpolation='nearest', cmap='magma')
        colors = [ img.cmap(img.norm(value)) for value in self.vals]

        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=colors[k], label="{l}".format(l=self.label[k]) ) for k in range(len(self.vals)) ]

        for i in range(numstep):
            self.update_anim()
            self.dynamicplotStep(i, patches)
        self.reinitialise_properties()
        self.reinitialise()

                                         
    def dynamicplotStep(self, i, patches):
        clear_output(wait=True)

        fig, ax = plt.subplots() 
        img = ax.imshow(self.grid, interpolation='nearest', cmap='magma', vmin=0.0, vmax=1.0)
        plt.title('Time=%d'%i); plt.axis('tight')

        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        plt.tick_params(axis='both', left=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
        plt.show()

################################################## METHOD ########################################################

    def phaseDiagram(self):
        for idx1 in tqdm(range(self.N), desc='Row', leave=False, unit='row'):
            for idx2 in tqdm(range(self.N), desc='Col', leave=False, unit='col'):
                for i in tqdm(range(self.equistep), desc='Equilibrate sweep', leave=False, unit='sweep'):              # equilibrate
                    self.update(idx1, idx2)                                                                            # Monte Carlo moves

                for j in tqdm(range(self.calcstep), desc='Measurement sweep', leave=False, unit='sweep'):              # measurement
                    self.update(idx1, idx2)                                                                            # Monte Carlo moves
                    if (j%10 == 0):
                        self.infected[int(j/10)]  = self.calcInfectedSite()                    

                self.I[idx1][idx2] = np.sum(self.infected)/(self.factor)/self.N
                self.reinitialise_properties()
        #self.dumpData() 
        self.plotStats(phase_diagram=True)
        self.reinitialise_properties()
        self.reinitialise()

    def variance(self):
        for idx1 in tqdm(range(self.N), desc='Row', leave=False, unit='row'):
            for idx2 in tqdm(range(self.N), desc='Col', leave=False, unit='col'):
                for i in tqdm(range(self.equistep), desc='Equilibrate sweep', leave=False, unit='sweep'):              # equilibrate
                    self.update(idx1, idx2)                                                                            # Monte Carlo moves

                for j in tqdm(range(self.calcstep), desc='Measurement sweep', leave=False, unit='sweep'):              # measurement
                    self.update(idx1, idx2)                                                                            # Monte Carlo moves
                    if (j%10 == 0):
                        self.infected[int(j/10)]  = self.calcInfectedSite()
                        self.infected2[int(j/10)] = self.calcInfectedSite()**2                    

                self.var_I[idx1][idx2] = (np.sum(self.infected2)/(self.factor) - np.sum(self.infected)*np.sum(self.infected)/(self.factor)/(self.factor))/self.N
                self.reinitialise_properties()
        #self.dumpData() 
        self.plotStats(variance=True)
        self.reinitialise_properties()
        self.reinitialise()


    def cutVariance(self, p1_cut):
        self.p1_cut = p1_cut
        self.p2, self.p3 = 0.5, 0.5
        for idx, p1step in tqdm(enumerate(self.p1_cut), desc='Col', leave=False, unit='-th'):
            self.p1 = p1step
            for j in tqdm(range(self.calcstep), desc='Measurement sweep', leave=False, unit='sweep'):           # measurement
                self.update_anim()                                                                              # Monte Carlo moves
                if (j%10 == 0):
                    self.infected[int(j/10)]  = self.calcInfectedSite()
                    self.infected2[int(j/10)] = self.calcInfectedSite()**2                   

            self.var_I_cut[idx] = (np.sum(self.infected2)/(self.factor) - np.sum(self.infected)*np.sum(self.infected)/(self.factor)/(self.factor))/self.N
            self.reinitialise_properties()
        #self.dumpData() 
        self.plotStats(cut=True)
        self.reinitialise_properties()
        self.reinitialise()


    def compare_infected_immune(self):
        for idx, frac in tqdm(enumerate(self.f_im), desc='Immune Fraction', leave=False, unit='th fraction'):
            self.setImmuneState(frac)
            for j in tqdm(range(self.calcstep), desc='Measurement sweep', leave=False, unit='sweep'):           # measurement
                self.update_anim()                                                                              # Monte Carlo moves
                if (j%10 == 0):
                    self.infected[int(j/10)]  = self.calcInfectedSite()
            self.I_immune[idx] = np.sum(self.infected)/(self.factor)/self.N
            self.I_immune_err[idx] = self.calcIError(idx)
        #self.dumpData() 
        self.plotStats(infected_immune=True)
        self.reinitialise()

################################################## CALCULATION FUNCTIONS ########################################################

    def calcInfectedSite(self):
        return np.sum(np.round(self.grid))

    def setImmuneState(self, frac):
        idx = np.argwhere(self.grid == 0.4)
        for i in idx:
            if rand() < frac:
                self.grid[i[0]][i[1]] = 0.4001

    def calcIError(self, pointstep):
        for idx, i in enumerate(range(self.factor)):
            Ii =  np.sum(np.delete(self.infected, [i]))/(self.factor - 1)/self.N
            self.Ii_I[i] = (Ii - self.I_immune[pointstep])**2
        return np.sqrt((self.factor - 1)/(self.factor)*np.sum(self.Ii_I))

################################################## PLOTTING ########################################################

    def plotStats(self, phase_diagram=False, variance=False, cut=False, infected_immune=False):
        if phase_diagram:
            plt.figure()
            cp = plt.contourf(self.p1p3[0], self.p1p3[1], self.I)
            plt.colorbar(cp)
            plt.title('Phase Diagram of SIR model')
            plt.xlabel('p1')
            plt.ylabel('p3')
            plt.show()

        elif variance:
            plt.figure()
            cp = plt.contourf(self.p1p3[0], self.p1p3[1], self.var_I)
            plt.colorbar(cp)
            plt.title('Variance')
            plt.xlabel('p1')
            plt.ylabel('p3')
            plt.show()

        elif cut:
            plt.scatter(self.p1_cut, self.var_I_cut, marker='o', s=20, color='RoyalBlue')
            plt.xlabel("p1");
            plt.ylabel("Cut for Wave Variance");         
            plt.axis('tight');
            plt.show()

        elif infected_immune:
            # plt.scatter(self.f_im, self.I_immune, marker='o', s=20, color='RoyalBlue')
            plt.errorbar(self.f_im, self.I_immune, yerr=self.I_immune_err, fmt='.', color='RoyalBlue', ecolor='black', elinewidth=0.2, capsize=2)
            plt.xlabel("Immune Fraction");
            plt.ylabel("Infected Fraction");         
            plt.axis('tight');
            plt.show()

        else:
            raise SIRError('Unknown method type given : [ phase_diagram ] [ variance ] [ cut ] [ infected_immune ]')

    def dumpData(self, phase_diagram=False, variance=False, cut=False, infected_immune=False):
        if phase_diagram:
            if not os.path.isfile('./Phase_Diagram'):
                os.makedirs('./Phase_Diagram')
            else:
                np.savetxt('{}/P1.txt'.format('./Phase_Diagram'), self.p1p3[0])
                np.savetxt('{}/P3.txt'.format('./Phase_Diagram'), self.p1p3[1])
                np.savetxt('{}/I.txt'.format('./Phase_Diagram'), self.I)

        elif variance:
            if not os.path.isfile('./Variance'):
                os.makedirs('./Variance')
            else:
                np.savetxt('{}/P1.txt'.format('./Variance'), self.p1p3[0])
                np.savetxt('{}/P3.txt'.format('./Variance'), self.p1p3[1])
                np.savetxt('{}/Varience_I.txt'.format('./Variance'), self.var_I)

        elif cut:
            if not os.path.isfile('./Cut_Variance'):
                os.makedirs('./Cut_Variance')
            else:
                np.savetxt('{}/P1.txt'.format('./Cut_Variance'), self.p1_cut)
                np.savetxt('{}/Varience_I.txt'.format('./Cut_Variance'), self.var_I_cut)

        elif infected_immune:
            if not os.path.isfile('./Infected_Immune'):
                os.makedirs('./Infected_Immune')
            else:
                np.savetxt('{}/Immune_Fraction.txt'.format('./Infected_Immune'), self.f_im)
                np.savetxt('{}/Infected_Fraction.txt'.format('./Infected_Immune'), self.I_immune)
                np.savetxt('{}/Infected_Fraction_Error.txt'.format('./Infected_Immune'), self.I_immune_err)

        else:
            raise SIRError('Unknown method type given : [ phase_diagram ] [ variance ] [ cut ] [ infected_immune ]')

