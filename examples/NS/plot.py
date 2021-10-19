#! /usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib as matplotlib
#matplotlib.use('pgf')
import numpy as np
#import matplotlib.path as path
import matplotlib.pyplot as plt
#from matplotlib2tikz import save as tikz_save
#\usepgfplotslibrary{external}
#\tikzexternalize
#from pylab import *
#from matplotlib import rc
#params = {'legend.fontsize': 6}
#plt.rcParams.update(params)
for i in range(300000):
    if i%5000==0:
        if i >= 240000:
            u = np.loadtxt('u'+str(i))
            v = np.loadtxt('v'+str(i))
            x = np.loadtxt('x'+str(i))
            y = np.loadtxt('y'+str(i))
            """
            v_total = (u**2.0+v**2.0)**0.5
            fig, ax = plt.subplots(dpi=600)
            p = ax.contourf(x,y,v_total)
            vector_field = ax.quiver(x,y,u,v,headlength=7)
            ax.set_aspect('equal')
            colorbar = fig.colorbar(p)
            plt.savefig("vector_plots/"+ "vel_field_" + str(i)+".png")

            plt.close()
            fig, ax = plt.subplots()
            p = ax.contourf(x,y,v)
            ax.set_aspect('equal')
            colorbar = fig.colorbar(p)
            plt.savefig("graphs_v/"+ "v_" + str(i)+".png")


            plt.close()
            """
            fig, ax = plt.subplots(dpi=600)
            #fig, ax = plt.subplots()
            levels= np.linspace(-1,15,num=100)
            p = ax.contourf(x,y,u,levels=levels)
            ax.set_aspect('equal')
            colorbar = fig.colorbar(p)
            plt.savefig("graphs_u/"+ "u_" + str(i)+".png")

        #p = ax.contourf(x,y,v_total)
        #ax.set_aspect('equal')
        #colorbar = fig.colorbar(p)
        #plt.savefig("graphs_total/"+ "v_total_" + str(i)+".png")


