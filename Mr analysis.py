import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import seaborn as sns
#df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\2min_6min_time_elapse_individual_truck_grade_Oct21_Mar22.csv")
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\all_grade_over_2000tonnage.csv")
df = df[0:5000]
df1 = df.groupby(np.arange(len(df))//2).mean()
plt.plot(df1)
