# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 10:47:47 2020

@author: srava
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 09:37:51 2020

@author: srava
"""
# For plotting
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
#for matrix math
import numpy as np
#for normalization + probability density function computation
from scipy import stats

from gaussian import Gaussian


if __name__ == '__main__':
    random_seed=36788765
    np.random.seed(random_seed)
    
    Mean1 = 2.0  # Input parameter, mean of first normal probability distribution
    Standard_dev1 = 4.0 #@param {type:"number"}
    Mean2 = 9.0 # Input parameter, mean of second normal  probability distribution
    Standard_dev2 = 2.0 #@param {type:"number"}
    
    # generate data
    y1 = np.random.normal(Mean1, Standard_dev1, 1000)
    y2 = np.random.normal(Mean2, Standard_dev2, 500)
    data=np.append(y1,y2)
    
    # For data visiualisation calculate left and right of the graph
    Min_graph = min(data)
    Max_graph = max(data)
    x = np.linspace(Min_graph, Max_graph, 2000) # to plot the data
    
    print('Input Gaussian {:}: μ = {:.2}, σ = {:.2}'.format("1", Mean1, Standard_dev1))
    print('Input Gaussian {:}: μ = {:.2}, σ = {:.2}'.format("2", Mean2, Standard_dev2))
    #sns.distplot(data, bins=20, kde=False);
    
    best_single = Gaussian(np.mean(data), np.std(data))
    print('Best single Gaussian: μ = {:.2}, σ = {:.2}'.format(best_single.mu, best_single.sigma))
    #fit a single gaussian curve to the data
    g_single = stats.norm(best_single.mu, best_single.sigma).pdf(x)
    sns.distplot(data, bins=20, kde=False, norm_hist=True);
    plt.plot(x, g_single, label='single gaussian');
    plt.legend();