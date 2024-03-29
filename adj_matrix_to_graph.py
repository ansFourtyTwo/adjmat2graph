import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tkinter import *
from tkinter import filedialog


def show_circular_adj_graph(adj_df, center=None, show_plot=False):

    fig, ax = plt.subplots()
    fig.tight_layout()
    
    # Convert to numpy matrix
    mat = adj_df.to_numpy()
    rows, cols = np.where(~np.isnan(mat))
    
    # Plot circles
    circle_factor = 2
    Rlvl = np.array([3, 2, 1])*circle_factor
    Rcircles = Rlvl + 0.5 * circle_factor
    for R in Rcircles:
        cf = plt.Circle((0,0), R, color='#ffff00', alpha=0.3, fill=True, zorder=0)
        cb = plt.Circle((0,0), R, fill=False, edgecolor='k', zorder=0)
        ax.add_artist(cf)
        ax.add_artist(cb)
        
    # Label cirles
    labels = ['Weniger wichtige Personen', 'Wichtige Personen', 'Sehr wichtige Personen']
    Rlabels = Rlvl + 0.3 * circle_factor
    boxprops = dict(boxstyle='round', facecolor='white')
    for R, label  in zip(Rlabels, labels):
        plt.text(0, R, label, fontsize=6, horizontalalignment='center', bbox=boxprops)
        

    # Extract names and convert to dict
    names = list(df.columns)
    nlabels = {k: v for k, v in enumerate(names)} 

    # Get center index
    for k,v in nlabels.items():
        if v == center:
            ind_center = k
    
    # Make center node bigger
    nsize = [100] * len(names)
    nsize[ind_center] = 500
    
    # Calculate node positions
    npos = dict()
    N = len(nlabels)-1
    i = 0
    for k,v in nlabels.items():
        if v == center:
            npos[k]=(0,0)
        else:
            mat_pos = [ind_center, k]
            mat_pos.sort()
            lvl = int(mat[mat_pos[0]][mat_pos[1]])
            R = Rlvl[lvl]
            
            t = 2*np.pi*i/N
            npos[k] = (R*np.sin(t), R*np.cos(t))
            i += 1
    
    # Generate edges and graph
    edges = zip(rows.tolist(), cols.tolist()) 
    gr = nx.Graph()
    for edge in edges:
        u,v = edge
        if int(mat[u,v])==2 and ind_center in edge:
            gr.add_edge(u, v, weight=2, color='r')
        else:
            gr.add_edge(u,v, weight=1, color='#AAAAAA')
            
    edges = gr.edges()
    colors = [gr[u][v]['color'] for u,v in edges]
    weigths = [gr[u][v]['weight'] for u,v in edges]
     
    nx.draw(gr, node_size=nsize, pos=npos, labels=nlabels, edge_color=colors, width=weigths, with_labels=True, fontsize=10)
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.gca().set_aspect('equal', adjustable='box')
    
    if show_plot:
        plt.show()
        
    # Save figure
    pre, ext = os.path.splitext(xlsxfile)
    imfile = pre + '_' + center + '.pdf'
    plt.savefig(imfile, bbox_inches='tight')

    
    

if __name__ == '__main__':

    root = Tk()
    xlsxfile =  filedialog.askopenfilename(initialdir = "D:/Temp", title = "Select file", filetypes = (("xlsx files","*.xlsx"),("all files","*.*")))
    
    if xlsxfile is None:
        sys.exit()
    
    # Read excel data
    df = pd.read_excel(xlsxfile, index_col=0)
    names = list(df.columns)

    # Generate and plot graph        
    for name in names:
        show_circular_adj_graph(df, center=name, show_plot=False)
    
    



