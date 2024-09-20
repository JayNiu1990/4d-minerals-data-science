# import numpy as np
# import csv
# import matplotlib.pyplot as plt
# import pandas as pd
# df = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\correlation1.csv')
# data = []
# for i in range(len(df)):
#     x1 = df['X_central'][i] - 50
#     x2 = df['X_central'][i] + 50
#     y1 = df['Y_central'][i] - 50
#     y2 = df['Y_central'][i] + 50
#     z1 = df['Z_central'][i] - 25
#     z2 = df['Z_central'][i] + 25
#     xx = np.arange(x1, x2, 1).astype('int')
#     yy = np.arange(y1, y2, 1).astype('int')
#     zz = np.arange(z1, z2, 1).astype('int')
#     value = []
#     slope = df['slope'][i]
#     for k in zz:
#         for j in yy:
#             for i in xx:
#                 value.append([i,j,k,slope])
#     sub_df = pd.DataFrame(value,columns = ['X','Y','Z','slope'])
#     data.append(sub_df)
# data1 = pd.concat(data)
# #data1.to_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\correlation.csv')

# # Step 2: Determine the dimensions of the 3D image array based on X, Y, and Z coordinates
# x_min = data1['X'].min()
# x_max = data1['X'].max()
# y_min = data1['Y'].min()
# y_max = data1['Y'].max()
# z_min = data1['Z'].min()
# z_max = data1['Z'].max()

# # Determine the size of the 3D image array (adjust the resolution as needed)
# resolution = 1  # Number of voxels per unit
# x_size = int((x_max - x_min) * resolution)
# y_size = int((y_max - y_min) * resolution)
# z_size = int((z_max - z_min) * resolution)

# # Step 3: Create an empty 3D image array
# image_3d = np.zeros((x_size, y_size, z_size))

import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
#df = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\correlation1.csv')
# df = df[(pd.to_numeric(df["label"], errors='coerce')==1)]
# x_min = df['X_central'].min()
# x_max = df['X_central'].max()
# y_min = df['Y_central'].min()
# y_max = df['Y_central'].max()
# z_min = df['Z_central'].min()
# z_max = df['Z_central'].max()
# resolution = 1  # Number of voxels per unit
# x_size = int(((x_max+50) - (x_min-50))/100)
# y_size = int(((y_max+50) - (y_min-50))/100)
# z_size = int(((z_max+25) - (z_min-25))/50)
# image_3d1 = np.zeros((x_size, y_size, z_size))
# # Step 4: Map coordinates to voxel positions and set voxel values
# for index, row in df.iterrows():
#     x, y, z, value = row['X_central'], row['Y_central'], row['Z_central'], row['label']
#     voxel_x = int((x - x_min)/100)
#     voxel_y = int((y - y_min)/100)
#     voxel_z = abs(int((z - z_min)/50))
#     image_3d1[voxel_x, voxel_y, voxel_z] = value
    
# import numpy as np
# import csv
# import matplotlib.pyplot as plt
# import pandas as pd
# df = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\correlation1.csv')
# df = df[(pd.to_numeric(df["label"], errors='coerce')==2)]
# x_min = df['X_central'].min()
# x_max = df['X_central'].max()
# y_min = df['Y_central'].min()
# y_max = df['Y_central'].max()
# z_min = df['Z_central'].min()
# z_max = df['Z_central'].max()
# resolution = 1  # Number of voxels per unit
# x_size = int(((x_max+50) - (x_min-50))/100)
# y_size = int(((y_max+50) - (y_min-50))/100)
# z_size = int(((z_max+25) - (z_min-25))/50)
# image_3d2 = np.zeros((x_size, y_size, z_size))
# # Step 4: Map coordinates to voxel positions and set voxel values
# for index, row in df.iterrows():
#     x, y, z, value = row['X_central'], row['Y_central'], row['Z_central'], row['label']
#     voxel_x = int((x - x_min)/100)
#     voxel_y = int((y - y_min)/100)
#     voxel_z = abs(int((z - z_min)/50))
#     image_3d2[voxel_x, voxel_y, voxel_z] = value  
    
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\correlation1.csv')
x_min = df['X_central'].min()
x_max = df['X_central'].max()
y_min = df['Y_central'].min()
y_max = df['Y_central'].max()
z_min = df['Z_central'].min()
z_max = df['Z_central'].max()
resolution = 1  # Number of voxels per unit
x_size = int(((x_max+50) - (x_min-50))/100)
y_size = int(((y_max+50) - (y_min-50))/100)
z_size = int(((z_max+25) - (z_min-25))/50)
image_3d = np.zeros((x_size, y_size, z_size))
# Step 4: Map coordinates to voxel positions and set voxel values
for index, row in df.iterrows():
    x, y, z, value = row['X_central'], row['Y_central'], row['Z_central'], row['label']
    voxel_x = int((x - x_min)/100)
    voxel_y = int((y - y_min)/100)
    voxel_z = abs(int((z - z_min)/50))
    image_3d[voxel_x, voxel_y, voxel_z] = value  

image_3d = image_3d.astype('uint8')
image_3d = np.transpose(image_3d,(2,0,1))
from skimage import io
io.imsave('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\correlation.tif',image_3d)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.voxels(image_3d)
# ax.legend()
# plt.show()





