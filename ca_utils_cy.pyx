from __future__ import division

import numpy as np
from numpy.random import rand

import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as mpatches
from IPython.display import clear_output

from tqdm import tqdm_notebook as tqdm

import os
import warnings

##########################################
cimport cython
cimport numpy as np

from libc.math cimport sqrt
from libc.stdlib cimport rand
cdef extern from "limits.h":
    int RAND_MAX

warnings.filterwarnings("ignore",category =RuntimeWarning)
warnings.filterwarnings("ignore",category =FutureWarning)

cdef class SIRError(Exception):
    """ An exception class for Ising """
    pass

# Class to simulate SIR model
cdef class SIR(object):

    cdef public int N, equistep, calcstep, factor
    cdef public double p1, p2, p3, ini_S, ini_I, ini_R
    cdef public double[:, :] grid, I, var_I
    cdef public list label
    cdef public np.double_t[:] vals, infected, infected2, Ii_I, p1_data, p3_data, p1_cut, p3_cut, var_I_cut, I_immune, I_immune_err, f_im
    cdef public list p1p3, p1p3_cut
    cdef public str RUN_NAME

############################################### INITIALISER #####################################################
    def __init__(self, int N, list fraction, list p, int factor, int equistep, int calcstep):
        self.N               = N
        self.p1              = p[0]
        self.p2              = p[1]
        self.p3              = p[2]
        self.ini_S           = fraction[0]
        self.ini_I           = fraction[1]
        self.ini_R           = fraction[2]
        self.vals            = np.array([0.0, 1.0, 0.4])
        self.label           = ['S', 'I', 'R']
        self.grid            = np.random.choice(self.vals, N*N, p=[self.ini_S, self.ini_I, self.ini_R]).reshape(N, N)

        self.RUN_NAME        = 'N_{0} CALCSTEP_{1}'.format(N, calcstep)
        self.equistep        = equistep
        self.calcstep        = calcstep
        self.factor          = factor

        self.infected        = np.zeros(factor)
        self.infected2       = np.zeros(factor)
        self.Ii_I            = np.zeros(factor)

        self.p1_data         = np.linspace(0.0, 1.0, N)
        self.p3_data         = np.linspace(0.0, 1.0, N)
        self.p1p3            = np.meshgrid(self.p1_data, self.p3_data)

        self.p1_cut          = np.linspace(0.2, 0.5, N)
        self.p3_cut          = np.zeros(N) + 0.5
        self.p1p3_cut        = np.meshgrid(self.p1_cut, self.p3_cut)

        self.I               = np.zeros((N, N))
        self.var_I           = np.zeros((N, N))

        self.var_I_cut       = np.zeros(N)
        self.I_immune        = np.zeros(N)
        self.I_immune_err    = np.zeros(N)
        self.f_im            = np.linspace(0.0, 1.0, N)
    
    def reinitialise(self):
        np.random.seed(0)
        self.grid            = np.random.choice(self.vals, self.N*self.N, p=[self.ini_S, self.ini_I, self.ini_R]).reshape(self.N, self.N)

        self.p1p3            = np.meshgrid(np.linspace(0.0, 1.0, self.N), np.linspace(0.0, 1.0, self.N))

        self.I               = np.zeros((self.N, self.N))
        self.var_I           = np.zeros((self.N, self.N))
        self.var_I_cut       = np.zeros(self.N)

        self.I_immune        = np.zeros(self.N)
        self.I_immune_err    = np.zeros(self.N)
        self.f_im            = np.linspace(0.0, 1.0, self.N)

    def reinitialise_properties(self):
        self.infected        = np.zeros(self.factor)
        self.infected2       = np.zeros(self.factor)

        np.random.seed(0)
        self.grid            = np.random.choice(self.vals, self.N*self.N, p=[self.ini_S, self.ini_I, self.ini_R]).reshape(self.N, self.N)

################################################### UPDATE FUNCTIONS #######################################################

    def update_anim(self):
        cdef int i, j, total_infect
        cdef double[:, :] newGrid

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
                                int(self.grid[(i+1)%self.N, (j-1)%self.N]) + int(self.grid[(i+1)%self.N, (j+1)%self.N]))
    
                # apply SIR's rules 
                if self.grid[i, j]  == 0.0: 
                    if total_infect > 0 :
                        if rand() < self.p1*RAND_MAX:
                            newGrid[i, j] = 1
                
                elif self.grid[i, j]  == 1:
                    if rand() < self.p2*RAND_MAX:
                        newGrid[i, j] = 0.4
                
                elif self.grid[i, j]  == 0.4:
                    if rand() < self.p3*RAND_MAX:
                        newGrid[i, j] = 0.0
  
        # update data 
        self.grid[:] = newGrid[:]

    def update(self, int idx1, int idx2):
        cdef int i, j, total_infect
        cdef double[:, :] newGrid

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
                                int(self.grid[(i+1)%self.N, (j-1)%self.N]) + int(self.grid[(i+1)%self.N, (j+1)%self.N]))
                                    
                # apply SIR's rules 
                if self.grid[i, j]  == 0.0:
                    if total_infect > 0 :
                        if rand() < self.p1p3[0][idx1][idx2]*RAND_MAX:
                            newGrid[i, j] = 1
                
                elif self.grid[i, j]  == 1.0:
                    if rand() < 0.5*RAND_MAX:
                        newGrid[i, j] = 0.4
                
                elif self.grid[i, j]  == 0.4:
                    if rand() < self.p1p3[1][idx1][idx2]*RAND_MAX:
                        newGrid[i, j] = 0.0

        # update data 
        self.grid[:] = newGrid[:]

################################################### VISUALISATION #####################################################

    def DynamicPlot(self, int numstep):
        cdef int i
        cdef list colors, patches

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
                                         
    def dynamicplotStep(self, int i, list patches):
        clear_output(wait=True)

        fig, ax = plt.subplots() 
        img = ax.imshow(self.grid, interpolation='nearest', cmap='viridis', vmin=0.0, vmax=1.0)
        plt.title('Time=%d'%i); plt.axis('tight')

        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        plt.tick_params(axis='both', left=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
        plt.show()

################################################## METHOD ########################################################

    def phaseDiagram(self):
        cdef int idx1, idx2, i, j
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
        cdef int idx1, idx2, i, j
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


    def cutVariance(self, np.double_t[:] p1_cut):
        cdef int idx, j
        cdef double p1step
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
        cdef int idx, j
        cdef double frac
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

    def setImmuneState(self, double frac):
        cdef int idx1, idx2
        for idx1 in range(self.N):
            for idx2 in range(self.N):
                if self.grid[idx1][idx2] == 0.4:
                    if rand() < frac*RAND_MAX:
                        self.grid[idx1][idx2] = 0.4001

    def calcIError(self, int pointstep):
        cdef double Ii
        cdef int i
        for i in range(self.factor):
            Ii =  np.sum(np.delete(self.infected, [i]))/(self.factor - 1)/self.N
            self.Ii_I[i] = (Ii - self.I_immune[pointstep])**2
        return sqrt((self.factor - 1)/(self.factor)*np.sum(self.Ii_I))

################################################## PLOTTING ########################################################

    def plotStats(self, phase_diagram=False, variance=False, cut=False, infected_immune=False):
        if not os.path.isdir('./{}'.format(self.RUN_NAME)):
            os.makedirs('./{}'.format(self.RUN_NAME))        
        if phase_diagram:
            plt.figure()
            cp = plt.contourf(self.p1p3[0], self.p1p3[1], self.I)
            plt.colorbar(cp)
            plt.title('Phase Diagram of SIR model')
            plt.xlabel('p1')
            plt.ylabel('p3')
            plt.show()
            plt.savefig('{}/phase_diagram'.format(self.RUN_NAME))

        elif variance:
            plt.figure()
            cp = plt.contourf(self.p1p3[0], self.p1p3[1], self.var_I)
            plt.colorbar(cp)
            plt.title('Variance')
            plt.xlabel('p1')
            plt.ylabel('p3')
            plt.show()
            plt.savefig('{}/variance'.format(self.RUN_NAME))


        elif cut:
            plt.scatter(self.p1_cut, self.var_I_cut, marker='o', s=20, color='RoyalBlue')
            plt.title('Cut for Wave Variance')
            plt.xlabel("p1");
            plt.ylabel("Variance");         
            plt.axis('tight');
            plt.show()
            plt.savefig('{}/cut'.format(self.RUN_NAME))


        elif infected_immune:
            plt.errorbar(self.f_im, self.I_immune, yerr=self.I_immune_err, fmt='.', color='RoyalBlue', ecolor='black', elinewidth=0.2, capsize=2)
            plt.xlabel("Immune Fraction");
            plt.ylabel("Infected Fraction");         
            plt.axis('tight');
            plt.show()
            plt.savefig('{}/infected_immune'.format(self.RUN_NAME))


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