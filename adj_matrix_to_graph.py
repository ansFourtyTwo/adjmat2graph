import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tkinter import *
from tkinter import filedialog


def show_graph_with_labels(adjacency_matrix, labels, pos):
    rows, cols = np.where(adjacency_matrix == 1)
    print(rows)
    print(cols)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    
    
    nx.draw(gr, node_size=1200, pos=pos, labels=labels, with_labels=True)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    

if __name__ == '__main__':

    root = Tk()
    xlsxfile =  filedialog.askopenfilename(initialdir = "C:/", title = "Select file", filetypes = (("xlsx files","*.xlsx"),("all files","*.*")))
    
    # Read excel data
    df = pd.read_excel(xlsxfile, index_col=0)

    # Convert to numpy matrix
    mat = df.to_numpy()

    # Extract names and convert to dict
    names = list(df.columns)
    nlabels = {k: v for k, v in enumerate(names)} 

    # Calc node positions
    center = "Simon"
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

    # Generate and plot graph        
    show_graph_with_labels(mat,labels=nlabels, pos=npos)



