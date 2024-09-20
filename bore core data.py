# if __name__ == '__main__':
#     import pymc3 as pm
#     import numpy as np
#     import theano
#     import arviz as az
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     theano.config.compute_test_value = "ignore"
#     az.style.use("arviz-darkgrid")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections
fields = ['BHID', 'X','Y','Z']
data = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)
each_name = [item for item, count in collections.Counter(data["BHID"]).items() if count > 1]
index_pos = []
for stri in each_name:
    index_pos.append(list(data["BHID"]).index(stri))
list1 = []
for i in index_pos:
    list1.append(data[i:i+1])
list1 = pd.concat(list1)
x1= np.array(list1["X"])
y1 = np.array(list1["Y"])


data2 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\includedcores.csv", skipinitialspace=True, usecols=fields)
each_name2 = [item for item, count in collections.Counter(data2["BHID"]).items() if count > 1]
index_pos2 = []
for stri in each_name2:
    index_pos2.append(list(data2["BHID"]).index(stri))
list2 = []
for i in index_pos2:
    list2.append(data2[i:i+1])
list2 = pd.concat(list2)
x2= np.array(list2["X"])
y2 = np.array(list2["Y"])
plt.scatter(x2,y2,label="data2")
plt.legend(loc = "upper right")
plt.show()


parameter1 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\includedcores.csv", decimal=',', 
                         skipinitialspace=True, usecols=['BHID', 'CuT_dh','CuS_dh','Fe_dh','As_dh'])
#parameter2 = data_parameter1[data_parameter1['BHID'].str.match('RC-1674')]



parameter2 = parameter1[(pd.to_numeric(parameter1["CuT_dh"], errors='coerce')>0.1)]
CuT = parameter2["CuT_dh"][0:10000]
plt.hist(pd.to_numeric(CuT),bins=200)
plt.xlim(0,5)
plt.show()








