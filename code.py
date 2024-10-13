# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:55:20 2024

@author: Fayçal
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import binom


def simulate_galton_board(n, num_balls):
    # Create a triangular matrix to hold the counts of balls in each cell
    triangle = np.zeros((n + 1, n + 1), dtype=int)

    for _ in range(num_balls):
        # Start the ball at the top of the triangle
        row, col = 0, 0
        
        # Simulate the dropping of the ball
        for _ in range(n):
            # Randomly choose to go left (down) or right (down-right)
            if random.random() < 0.5:
                row += 1  
            else:
                col += 1  
            
        # Increment the count in the final position
        triangle[row, col] += 1

    return triangle

    
def plot_histogram(triangle, n, num_balls,num_figure=1):
    # Extract counts from the last row of the triangle
    counts = [triangle[i, n - i]/num_balls for i in range(n+1)]
    # print(counts)
    
    # Plot the histogram
    plt.figure(num_figure,figsize=(10, 6))
    plt.bar(range(n+1), counts, color='blue', alpha=0.7)
    plt.title(f"Results for {num_balls} balls and {n} cells")
    plt.xlabel("Position")
    plt.ylabel("Number of Balls")
    plt.xticks(range(n+1))
    plt.show()
    
    
    
def plot_normal(n,triangle, num_balls):
    plt.figure(1)
    # results_board = [triangle[i, n - i]/num_balls for i in range(n+1)]
    mu,sigma = get_mean(triangle,num_balls,n), get_std(triangle,num_balls,n)
    x= np.linspace(0,n+1,10*n)
    # pdf = [1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2)) for x in np.linspace(-n//2,n//2+1,10*n)]
    plt.plot(x,norm.pdf(x,mu,sigma))
    # print(mu)
    
def get_mean(triangle,num_balls,n):
    results_board = [triangle[i, n - i] for i in range(n+1)]
    i=0
    mean = 0
    for num in results_board:
        mean+= i*num
        i+=1
    mean/=num_balls
    # print(mean)
    return mean
 
def get_std(triangle,num_balls,n):
    results_board = [triangle[i, n - i] for i in range(n+1)]
    i=0
    var=0
    mean = get_mean(triangle,num_balls,n)
    for num in results_board:
        var+= num * (i-mean)**2
        i+=1
    var/=num_balls
    std=np.sqrt(var)
    # print(std)
    return std

def compare_exp_normal(triangle,num_balls,n,plot=True):
    exp_values = [triangle[i, n - i]/num_balls for i in range(n+1)]
    mu = get_mean(triangle,num_balls,n)
    sigma = get_std(triangle,num_balls,n)
    diff = []
    if sigma == 0:
        return 0
    for i in range(n+1):
        th=norm.pdf(i,mu,sigma)
        if th!=0:
            diff.append((exp_values[i]-th)**2)
        elif exp_values[i]!=0:
            diff.append((exp_values[i]-th)**2)
        else:
            diff.append(0)
        # if i==0:
        #     print(exp_values[i],th,diff)
    if plot:
        plt.figure(2)
        plt.plot(diff,"*")
    return np.sum(diff)/num_balls

def mean_squared_error(array1, array2):
    if array1.shape != array2.shape:
        raise ValueError("The arrays must have the same shape.")
    
    squared_diffs = (array1 - array2) ** 2
    
    return np.mean(squared_diffs)


def comparison_graph_exp_normal(num_n,num_rep,factor_balls,figure):
    diff=[]
    N = []
    mean_diff = []
    for n in num_n:
        for j in range(num_rep):
            num_balls = factor_balls * n   
            triangle = simulate_galton_board(n, num_balls)
            diff.append(compare_exp_normal(triangle,num_balls,n,False))
            N.append(n)
            if get_std(triangle,num_balls,n)==0:
                plot_histogram(triangle, n, num_balls,5)
                plot_normal(n, triangle, num_balls)
                return 0
        mean_diff.append(sum(diff[-num_rep:])/num_rep)
    figure.plot(N,diff,'*')
    figure.plot(num_n,mean_diff)
    figure.set_title(f"Results for N = {factor_balls}*n balls")
    figure.set_xlabel("n : number of cells")
    figure.set_ylabel("Quadratic error")
    figure.set_xticks(num_n)
    plt.show()
    return mean_diff
    
    
def compare_exp_bin(triangle,num_balls,n,plot=True):
    exp_values = [triangle[i, n - i]/num_balls for i in range(n+1)]
    diff = []
    for i in range(n+1):
        th=binom.pmf(i,n,0.5)
        if th!=0:
            diff.append((exp_values[i]-th)**2)
        elif exp_values[i]!=0:
            diff.append((exp_values[i]-th)**2)
        else:
            diff.append(0)
    if plot:
        plt.figure(2)
        plt.plot(diff,"*")
    return np.sum(diff)/num_balls


def comparison_graph_exp_bin(num_n,num_rep,factor_balls,figure):
    diff=[]
    N = []
    mean_diff = []
    for n in num_n:
        for j in range(num_rep):
            num_balls = factor_balls*n  
            triangle = simulate_galton_board(n, num_balls)
            diff.append(compare_exp_bin(triangle,num_balls,n,False))
            N.append(n)
            if get_std(triangle,num_balls,n)==0:
                plot_histogram(triangle, n, num_balls,5)
                plot_normal(n, triangle, num_balls)
                return 0
        mean_diff.append(sum(diff[-num_rep:])/num_rep)
    figure.plot(N,diff,'*')
    figure.plot(num_n,mean_diff)
    figure.set_title(f"Results for N = {factor_balls}*n balls")
    figure.set_xlabel("n : number of cells")
    figure.set_ylabel("Quadratic error")
    figure.set_xticks(num_n)
    return mean_diff



def effect_of_n_and_N(Two_ns = [10, 100],Two_Ns = [1000, 100000]):
    fig, axs = plt.subplots(len(Two_ns), len(Two_Ns), figsize=(10, 8))
    
    
    
    for i in range(len(Two_ns)):
        for j in range(len(Two_Ns)):
            n, num_balls = Two_ns[i], Two_Ns[j]
            new_triangle = simulate_galton_board(n, num_balls)
            y = [new_triangle[k, n - k]/num_balls for k in range(n+1)]
            x = range(n+1)
            axs[i, j].bar(x, y, color='orange', alpha=0.7, label = "Experiment")
            axs[i, j].set_title(f"Results for {n} cells and {num_balls} balls")
            axs[i, j].set_xlabel("Position")
            axs[i, j].set_ylabel("Proportion")
            
            axs[i, j].bar(x,binom.pmf(x,n,0.5),width = 0.4, label = "B(n,1/2)")
            mu,sigma = get_mean(new_triangle,num_balls,n), get_std(new_triangle,num_balls,n)
            x_norm= np.linspace(0,n+1,10*n)
            # pdf = [1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2)) for x in np.linspace(-n//2,n//2+1,10*n)]
            axs[i, j].plot(x_norm,norm.pdf(x_norm,mu,sigma), label = "N(mu,sigma)")
            axs[i, j].legend()
    plt.tight_layout() # Adjust the layout to prevent overlap
    plt.show()
    
def mean_diff_plots_exp_norm():
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    YY= []
    for i in range(3):
        num_n = list(range(10,101,10))
        num_rep = 10
        factor_balls = 10**(i+1)
        YY.append(comparison_graph_exp_normal(num_n,num_rep,factor_balls,axs[i//2, i%2]))
    axs[1, 1].plot(num_n, YY[0], color='orange', alpha=0.7, label = "N = n")
    axs[1, 1].plot(num_n, YY[1], color='blue', alpha=0.7, label = "N = 10*n")
    axs[1, 1].plot(num_n, YY[2], color='green', alpha=0.7, label = "N = 100*n")
    axs[1, 1].set_title("Comparison graph")
    axs[1, 1].set_xlabel("n")
    axs[1, 1].set_ylabel("MSE")
    axs[1, 1].legend()
    plt.tight_layout() # Adjust the layout to prevent overlap
    plt.show()


def mean_diff_plots_exp_bin():
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    YY= []
    for i in range(3):
        num_n = list(range(10,101,10))
        num_rep = 10
        factor_balls = 10**(i+1)
        YY.append(comparison_graph_exp_bin(num_n,num_rep,factor_balls,axs[i//2, i%2]))
    axs[1, 1].plot(num_n, YY[0], color='orange', alpha=0.7, label = "N = n")
    axs[1, 1].plot(num_n, YY[1], color='blue', alpha=0.7, label = "N = 10*n")
    axs[1, 1].plot(num_n, YY[2], color='green', alpha=0.7, label = "N = 100*n")
    axs[1, 1].set_title("Comparison graph")
    axs[1, 1].set_xlabel("n")
    axs[1, 1].set_ylabel("MSE")
    axs[1, 1].legend()
    plt.tight_layout() # Adjust the layout to prevent overlap
    plt.show()

def mean_squared_error(array1, array2):
    if array1.shape != array2.shape:
        raise ValueError("The arrays must have the same shape.")
    
    squared_diffs = (array1 - array2) ** 2
    
    return np.mean(squared_diffs)

def mean_diff_plots_norm_bin():
    YY= []
    num_n = list(range(10,1001,10))
    for n in num_n:
        
        YY.append(mean_squared_error(binom.pmf(range(n+1),n,1/2),norm.pdf(range(n+1),n/2,n/4)))
    plt.figure()
    plt.plot(num_n, YY, color='orange', alpha=0.7, label = "N = n")
    plt.title("Comparison graph between theoretical Bin(n,1/2) and N(n/2,n/4)")
    plt.xlabel("n")
    plt.ylabel("MSE")
    plt.tight_layout() # Adjust the layout to prevent overlap
    plt.show()
    
