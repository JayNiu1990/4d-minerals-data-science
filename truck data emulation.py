if __name__ == '__main__':
    import warnings
    import matplotlib.pyplot as plt
    import plotly.express as px
    import numpy as np
    import pandas as pd    
    import plotly.io as pio
    import plotly.graph_objs as go
    import arviz as az
    from scipy import stats
    import math
    pio.renderers.default='browser'
    df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Escondida Trucks\\truck data emulation1.csv", skipinitialspace=True)
    truck = []
    for i in range(12107):
        truck.append(df[0+20*i:20+20*i])
    percent5 = []
    for i in range(len(truck)):
        percent5.append(truck[i][0:1])
        
    percent5 = []  
    [percent5.append(truck[i][0:1].values[0][0]) for i in range(len(truck))]
    
    percent10 = []  
    [percent10.append(truck[i][0:2].mean().values[0]) for i in range(len(truck))]

    percent20 = []  
    [percent20.append(truck[i][0:4].mean().values[0]) for i in range(len(truck))]
    
    percent50 = []  
    [percent50.append(truck[i][0:10].mean().values[0]) for i in range(len(truck))]
    
    percent100 = []  
    [percent100.append(truck[i][0:20].mean().values[0]) for i in range(len(truck))]
    