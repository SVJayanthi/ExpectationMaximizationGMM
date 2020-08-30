# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 09:37:51 2020

@author: srava
"""

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

import numpy as np

from gmm import GMM

from random import uniform


if __name__ == '__main__':    
    Mean1 = uniform(-10, 10)
    Standard_dev1 = uniform(1, 4)
    Mean2 = uniform(-10, 10) 
    Standard_dev2 = uniform(1, 4) 
    
    
    y1 = np.random.normal(Mean1, Standard_dev1, 1000)
    y2 = np.random.normal(Mean2, Standard_dev2, 500)
    data=np.append(y1,y2)
        
    
    Min_graph = min(data)
    Max_graph = max(data)
    x = np.linspace(Min_graph, Max_graph, 2000)
    
    n_iter = 20
    best_model = None
    best_loglike = float('-inf')
    
    model = GMM(data)
    
    model.iterate(n_iter)
    best_model = model
    best_loglike = model.loglike
    
    
    print('Input Gaussian {:}: μ = {:.2}, σ = {:.2}'.format("1", Mean1, Standard_dev1))
    print('Input Gaussian {:}: μ = {:.2}, σ = {:.2}'.format("2", Mean2, Standard_dev2))
    print('Gaussian {:}: μ = {:.2}, σ = {:.2}, weight = {:.2}'.format("1", best_model.G1.mu, best_model.G1.sigma, best_model.Prior1))
    print('Gaussian {:}: μ = {:.2}, σ = {:.2}, weight = {:.2}'.format("2", best_model.G2.mu, best_model.G2.sigma, best_model.Prior2))

    
    sns.distplot(data, bins=20, kde=False, norm_hist=True);
    g_both = [best_model.pdf(e) for e in x]
    plt.plot(x, g_both, label='gaussian mixture');
    g_left = [best_model.G1.pdf(e) * best_model.Prior1 for e in x]
    plt.plot(x, g_left, label='gaussian one');
    g_right = [best_model.G2.pdf(e) * best_model.Prior2 for e in x]
    plt.plot(x, g_right, label='gaussian two');
    plt.legend();