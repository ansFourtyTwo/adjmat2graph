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
    mat = df.to_numpy()
    rows, cols = np.where(~np.isnan(mat))
    
    circle_factor = 2
    Rlvl = np.array([3, 2, 1])*circle_factor
    for R in Rlvl:
        c = plt.Circle((0,0), R-0.5*circle_factor, color='k', fill=False)
        ax.add_artist(c)



    # Extract names and convert to dict
    names = list(df.columns)
    nlabels = {k: v for k, v in enumerate(names)} 

    # Calc node positions
    for k,v in nlabels.items():
        if v == center:
            ind_center = k
    
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
            print("Added position: " + str(npos[k]))
            i += 1
    

    print(rows)
    print(cols)
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
     
    nx.draw(gr, node_size=200, pos=npos, labels=nlabels, edge_color=colors, width=weigths, with_labels=True)
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.gca().set_aspect('equal', adjustable='box')
    
    if show_plot:
        plt.show()
        
    # Save figure
    pre, ext = os.path.splitext(xlsxfile)
    pngfile = pre + '_' + center + '.pdf'
    plt.savefig(pngfile, bbox_inches='tight')

    
    

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
    
    



