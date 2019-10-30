import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tkinter import *
from tkinter import filedialog


def show_circular_adj_graph(adj_df, center=None, show_plot=False):


    # Convert to numpy matrix
    mat = df.to_numpy()

    # Extract names and convert to dict
    names = list(df.columns)
    nlabels = {k: v for k, v in enumerate(names)} 

    # Calc node positions
    npos = dict()
    N = len(nlabels)-1
    R = 2
    i = 0
    for k,v in nlabels.items():
        if v == center:
            npos[k]=(0,0)
        else:
            t = 2*np.pi*i/N
            npos[k] = (R*np.sin(t), R*np.cos(t))
            print("Added position: " + str(npos[k]))
            i += 1
            
    rows, cols = np.where(mat != 0)
    print(rows)
    print(cols)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
     
    nx.draw(gr, node_size=1200, pos=npos, labels=nlabels, with_labels=True)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.gca().set_aspect('equal', adjustable='box')
    
    if show_plot:
        plt.show()
        
    # Save figure
    pre, ext = os.path.splitext(xlsxfile)
    pngfile = pre + '.png'
    plt.savefig(pngfile)

    
    

if __name__ == '__main__':

    root = Tk()
    xlsxfile =  filedialog.askopenfilename(initialdir = "D:/Temp", title = "Select file", filetypes = (("xlsx files","*.xlsx"),("all files","*.*")))
    
    if xlsxfile is None:
        sys.exit()
    
    # Read excel data
    df = pd.read_excel(xlsxfile, index_col=0)

    # Generate and plot graph        
    show_circular_adj_graph(df, center="Simon")
    
    



