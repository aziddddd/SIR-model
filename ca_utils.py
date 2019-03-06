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

class SIRError(Exception):
    """ An exception class for Ising """
    pass

# Class to simulate SIR model
class SIR():

############################################### INITIALISER #####################################################
    def __init__(self, N, p, factor, equistep, calcstep):
        self.N          = N
        self.p1         = p[0]
        self.p2         = p[1]
        self.p3         = p[2]
        # self.temp_point = temp_point
        # self.beta       = 1.0/temp
        self.vals       = np.array([0.0, 1.0, 0.4])
        self.label      = ['S', 'I', 'R']
        self.grid       = np.random.choice(self.vals, N*N, p=[0.5, 0.3, 0.2]).reshape(N, N)
        # self.low_T      = temp_range[0]
        # self.high_T     = temp_range[1]

        # self.RUN_NAME   = 'p1{0} p2{1} p3{2}'.format(p[0], p[1], p[2])
        self.equistep   = equistep
        self.calcstep   = calcstep
        self.factor     = factor

        # divide by number of samples, and by system size to get intensive values
        # self.n1         = 1.0/(calcstep*N*N)
        # self.n2         = 1.0/(calcstep*calcstep*N*N)

        # self.energy     = np.zeros(factor)
        # self.magnet     = np.zeros(factor)
        # self.energy2    = np.zeros(factor)
        # self.magnet2    = np.zeros(factor)
        # self.ci_c       = np.zeros(factor)
        self.infected    = np.zeros(factor)
        self.infected2   = np.zeros(factor)

        self.p1_data    = np.linspace(0.0, 1.0, N)
        self.p3_data    = np.linspace(0.0, 1.0, N)
        self.p1p3       = np.meshgrid(self.p1_data, self.p3_data)

        self.p1_cut     = np.linspace(0.0, 0.25, N)
        # self.p1_cut     = np.linspace(0.2, 0.5, N)
        self.p3_cut     = np.zeros(N) + 0.5
        self.p1p3_cut   = np.meshgrid(self.p1_cut, self.p3_cut)

        self.I          = np.zeros((N, N))
        self.var_I      = np.zeros((N, N))
        self.var_I_cut  = np.zeros(N)

        # self.I_err      = np.zeros(temp_point)

    
    def reinitialise(self):
        self.grid       = np.random.choice(self.vals, self.N*self.N, p=[0.7, 0.2, 0.1]).reshape(self.N, self.N)

        self.p1p3       = np.meshgrid(np.linspace(0.0, 1.0, self.N), np.linspace(0.0, 1.0, self.N))
        self.I          = np.zeros((self.N, self.N))
        self.var_I      = np.zeros((self.N, self.N))
        self.var_I_cut  = np.zeros(self.N)

        # self.I_err      = np.zeros(self.temp_point)

    
    def reinitialise_properties(self):
        self.infected      = np.zeros(self.factor)
        self.infected2     = np.zeros(self.factor)

        np.random.seed(0)
        self.grid          = np.random.choice(self.vals, self.N*self.N, p=[0.7, 0.2, 0.1]).reshape(self.N, self.N)

################################################### ANIMATION #######################################################

    def update_anim(self):
  
        # copy grid since we require 8 neighbors  
        # for calculation and we go line by line  
        newGrid = self.grid.copy() 
        for i in range(self.N): 
            for j in range(self.N): 
    
                # compute 8-neghbor sum 
                # using toroidal boundary conditions - x and y wrap around  
                # so that the simulaton takes place on a toroidal surface.
    
                total_infect = (int(self.grid[i, (j-1)%self.N]) + int(self.grid[i, (j+1)%self.N]) + 
                                int(self.grid[(i-1)%self.N, j]) + int(self.grid[(i+1)%self.N, j]) +
                                int(self.grid[(i-1)%self.N, (j-1)%self.N]) + int(self.grid[(i-1)%self.N, (j+1)%self.N]) + 
                                int(self.grid[(i+1)%self.N, (j-1)%self.N]) + int(self.grid[(i+1)%self.N, (j+1)%self.N]))/1
    
                # apply SIR's rules 
                if self.grid[i, j]  == 0.0: 
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
        return self.grid 

    # def update(frameNum, img, grid, N, p): 
    def update(self, idx1, idx2):
    #     if not cut:
    #         p = self.p1p3
    #     else:
    #         p = self.p1p3_cut

        # copy grid since we require 8 neighbors  
        # for calculation and we go line by line  
        newGrid = self.grid.copy() 
        # print('p1 = {0:0.1f}    p3 = {1:0.1f}'. format(p[0][idx1][idx2], p[1][idx1][idx2]))
        for i in range(self.N): 
            for j in range(self.N): 
    
                # compute 8-neghbor sum 
                # using toroidal boundary conditions - x and y wrap around  
                # so that the simulaton takes place on a toroidal surface.
                                    
                total_infect = (int(self.grid[i, (j-1)%self.N]) + int(self.grid[i, (j+1)%self.N]) + 
                                int(self.grid[(i-1)%self.N, j]) + int(self.grid[(i+1)%self.N, j]) +
                                int(self.grid[(i-1)%self.N, (j-1)%self.N]) + int(self.grid[(i-1)%self.N, (j+1)%self.N]) + 
                                int(self.grid[(i+1)%self.N, (j-1)%self.N]) + int(self.grid[(i+1)%self.N, (j+1)%self.N]))/1
                # print(total_infect)
                # apply SIR's rules
                # print('p1 = {0:0.1f}    p3 = {1:0.1f}'. format(self.p1p3[0][idx1][idx2], self.p1p3[1][idx1][idx2]))
                if self.grid[i, j]  == 0.0:
                    if total_infect > 0 :
                        # print('{0}    {1:0.2f}    {2}'.format(total_infect, rand(), self.p1p3[0][idx1][idx2]))
                        if rand() < self.p1p3[0][idx1][idx2]:
                            # print('sini \n ------------------------')
                            newGrid[i, j] = 1
                
                elif self.grid[i, j]  == 1.0:
                    if rand() < 0.5:
                        newGrid[i, j] = 0.4
                
                elif self.grid[i, j]  == 0.4:
                    if rand() < self.p1p3[1][idx1][idx2]:
                        newGrid[i, j] = 0.0
                        # print('jadi S')
    
        # update data 
        # img.set_data(newGrid) 
        self.grid[:] = newGrid[:] 
        return self.grid

################################################### DYNAMIC PLOT #####################################################

    def DynamicPlot(self, numstep):
        fig, ax = plt.subplots() 
        img = ax.imshow(self.grid, interpolation='nearest', cmap='viridis')
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
        img = ax.imshow(self.grid, interpolation='nearest', cmap='viridis', vmin=0.0, vmax=1.0)
        plt.title('Time=%d'%i); plt.axis('tight')

        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        plt.tick_params(axis='both', left=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
        plt.show()

################################################## STATS ########################################################

    # def analyse(self, cut=False):
    #     for idx1 in tqdm(range(self.N), desc='Row', leave=False, unit='-th'):
    #         for idx2 in tqdm(range(self.N), desc='Col', leave=False, unit='-th'):
    #             for i in tqdm(range(self.equistep), desc='Equilibrate sweep', leave=False, unit='sweep'):                                                                         # equilibrate
    #                 self.update(idx1, idx2, cut)                                                                            # Monte Carlo moves

    #             for j in tqdm(range(self.calcstep), desc='Measurement sweep', leave=False, unit='sweep'):                                                                         # measurement
    #                 self.update(idx1, idx2, cut)                                                                            # Monte Carlo moves
    #                 if (j%10 == 0):
    #                     self.getStats(int(j/10))                    

    #             if not cut:
    #                 self.I[idx1][idx2] = np.sum(self.infected)/(self.factor)/self.N
    #             else:
    #                 # self.var_I[idx1][idx2] = (np.sum(self.infected2)/(self.factor) - np.sum(self.infected)*np.sum(self.infected)/(self.factor)/(self.factor))/self.N
    #                 self.var_I[idx2] = (np.sum(self.infected2)/(self.factor) - np.sum(self.infected)*np.sum(self.infected)/(self.factor)/(self.factor))/self.N
    #                 # print(self.var_I[idx1][idx2])

    #             self.reinitialise_properties()
    #         if cut:
    #             break
    #     #self.dumpData() 
    #     self.plotStats(cut)
    #     self.reinitialise()

    def analyse(self, variance=False):
        for idx1 in tqdm(range(self.N), desc='Row', leave=False, unit='row'):
            for idx2 in tqdm(range(self.N), desc='Col', leave=False, unit='col'):
                for i in tqdm(range(self.equistep), desc='Equilibrate sweep', leave=False, unit='sweep'):                                                                         # equilibrate
                    self.update(idx1, idx2)                                                                            # Monte Carlo moves

                for j in tqdm(range(self.calcstep), desc='Measurement sweep', leave=False, unit='sweep'):                                                                         # measurement
                    self.update(idx1, idx2)                                                                            # Monte Carlo moves
                    if (j%10 == 0):
                        self.getStats(int(j/10))                    

                if not variance:
                    self.I[idx1][idx2] = np.sum(self.infected)/(self.factor)/self.N
                else:
                    # self.var_I[idx1][idx2] = (np.sum(self.infected2)/(self.factor) - np.sum(self.infected)*np.sum(self.infected)/(self.factor)/(self.factor))/self.N
                    self.var_I[idx1][idx2] = (np.sum(self.infected2)/(self.factor) - np.sum(self.infected)*np.sum(self.infected)/(self.factor)/(self.factor))/self.N
                    # print(self.var_I[idx1][idx2])

                self.reinitialise_properties()
        #self.dumpData() 
        self.plotStats(variance)
        self.reinitialise_properties()
        self.reinitialise()

    def waveVariance(self):
        self.p2, self.p3 = 0.5, 0.5
        for idx, p1step in tqdm(enumerate(self.p1_cut), desc='Col', leave=False, unit='-th'):
            self.p1 = p1step
            # for i in tqdm(range(self.equistep), desc='Equilibrate sweep', leave=False, unit='sweep'):                                                                         # equilibrate
            #     self.update_anim()                                                                            # Monte Carlo moves

            for j in tqdm(range(self.calcstep), desc='Measurement sweep', leave=False, unit='sweep'):                                                                         # measurement
                self.update_anim()                                                                            # Monte Carlo moves
                if (j%10 == 0):
                    self.getStats(int(j/10))                    

            # self.var_I[idx1][idx2] = (np.sum(self.infected2)/(self.factor) - np.sum(self.infected)*np.sum(self.infected)/(self.factor)/(self.factor))/self.N
            self.var_I_cut[idx] = (np.sum(self.infected2)/(self.factor) - np.sum(self.infected)*np.sum(self.infected)/(self.factor)/(self.factor))/self.N
            # print(self.var_I[idx1][idx2])

            self.reinitialise_properties()
        #self.dumpData() 
        self.plotStatsCut()
        self.reinitialise()


    def getStats(self, j):
        self.infected[j]  = self.calcInfectedSite()
        self.infected2[j] = self.calcInfectedSite()**2
    
    def calcInfectedSite(self):
        return np.sum(np.round(self.grid))

    def plotStats(self, variance):
        if not variance:
            plt.figure()
            cp = plt.contourf(self.p1p3[0], self.p1p3[1], self.I)
            plt.colorbar(cp)
            plt.title('Phase Diagram of SIR model')
            plt.xlabel('p1')
            plt.ylabel('p3')
            plt.show()

        else:
            plt.figure()
            cp = plt.contourf(self.p1p3[0], self.p1p3[1], self.var_I)
            plt.colorbar(cp)
            plt.title('Variance')
            plt.xlabel('p1')
            plt.ylabel('p3')
            plt.show()

    def plotStatsCut(self):
            plt.plot(self.p1_cut, self.var_I_cut, linewidth=0.3, color='black')
            plt.scatter(self.p1_cut, self.var_I_cut, marker='o', s=20, color='RoyalBlue')
            plt.xlabel("p1");
            plt.ylabel("Cut for Wave Variance");         
            plt.axis('tight');
            plt.show()

    # def plotStats(self):
    #     plt.subplot(2, 2, 1 );
    #     plt.plot(self.T, self.E, linewidth=0.3, color='black')
    #     plt.scatter(self.T, self.E, marker='o', s=20, color='RoyalBlue')
    #     plt.xlabel("Temperature (T)");
    #     plt.ylabel("Energy (E)");         plt.axis('tight');

    #     plt.subplot(2, 2, 2 );
    #     plt.plot(self.T, np.abs(self.M), linewidth=0.3, color='black')
    #     plt.scatter(self.T, np.abs(self.M), marker='o', s=20, color='ForestGreen')
    #     plt.xlabel("Temperature (T)"); 
    #     plt.ylabel("Magnetization (M)");   plt.axis('tight');

    #     plt.subplot(2, 2, 3 );
    #     plt.plot(self.T, self.C, linewidth=0.3, color='black')
    #     plt.errorbar(self.T, self.C, yerr=self.C_err, fmt='.', color='ForestGreen', ecolor='black', elinewidth=0.2, capsize=2)
    #     plt.xlabel("Temperature (T)");  
    #     plt.ylabel("Specific Heat (C)");   plt.axis('tight');   

    #     plt.subplot(2, 2, 4 );
    #     plt.plot(self.T, self.X, linewidth=0.3, color='black')
    #     plt.scatter(self.T, self.X, marker='o', s=20, color='ForestGreen')
    #     plt.xlabel("Temperature (T)"); 
    #     plt.ylabel("Susceptibility (X)");   plt.axis('tight');
    #     plt.tight_layout()
    #     #plt.savefig('{}/analysis_graph'.format(self.RUN_NAME))

    # 
    # def dumpData(self):
    #     np.savetxt('{}/Temperature.txt'.format(self.RUN_NAME), self.T)
    #     np.savetxt('{}/Energy.txt'.format(self.RUN_NAME), self.E)
    #     np.savetxt('{}/Magnetization.txt'.format(self.RUN_NAME), self.M)
    #     np.savetxt('{}/Specific Heat Error.txt'.format(self.RUN_NAME), self.C_err)
    #     np.savetxt('{}/Specific Heat.txt'.format(self.RUN_NAME), self.C)
    #     np.savetxt('{}/Susceptibility.txt'.format(self.RUN_NAME), self.X)

