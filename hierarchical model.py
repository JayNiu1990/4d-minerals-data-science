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
    fields = ['BHID','Fe_dh','As_dh','CuT_dh',"X","Y","Z","LITH"]
    pio.renderers.default='browser'
    df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)
    df = df.dropna()
    df = df[(pd.to_numeric(df["CuT_dh"], errors='coerce')>0) & (pd.to_numeric(df["Fe_dh"], errors='coerce')>0)& (pd.to_numeric(df["As_dh"], errors='coerce')>0)
            & (pd.to_numeric(df["X"], errors='coerce')>=17000)& (pd.to_numeric(df["X"], errors='coerce')<17500)
            & (pd.to_numeric(df["Y"], errors='coerce')>=107000)& (pd.to_numeric(df["Y"], errors='coerce')<107500)
            & (pd.to_numeric(df["Z"], errors='coerce')>=2500)& (pd.to_numeric(df["Z"], errors='coerce')<3000)]
    
    df['LITH'] = df['LITH'].astype(int)
    df = df.reset_index(drop=True)
    df["CuT_dh"] = df["CuT_dh"].astype("float")
    df["Fe_dh"] = df["Fe_dh"].astype("float")
    df["As_dh"] = df["As_dh"].astype("float")

    # add gaussian noise
    df['X'] = round(df['X'],2)
    df['Y'] = round(df['Y'],2)
    df['Z'] = round(df['Z'],2)
    
    name = df['BHID'].unique()
    df_singlebore = []
    for i in range(len(name)):
        df_singlebore.append(df[df['BHID'] == name[i]])
        
    lith_number = []    
    data_number = []
    for i in df_singlebore:
        lith_number.append(i['LITH'].nunique())
        data_number.append(len(i))
    df1 = pd.DataFrame()
    df1['BHID'] = name
    df1['LITH'] = lith_number
    df1['NUM'] = data_number
    
    