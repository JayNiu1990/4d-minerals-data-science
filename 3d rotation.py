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
    import random
    from sklearn.metrics import mean_squared_error
    from scipy.stats import invgamma
    fields = ['BHID','Fe_dh','As_dh','CuT_dh',"X","Y","Z","LITH"]
    pio.renderers.default='browser'
    df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)
    df = df.dropna()
    df = df[(pd.to_numeric(df["CuT_dh"], errors='coerce')>0.5) & (pd.to_numeric(df["Fe_dh"], errors='coerce')>0)& (pd.to_numeric(df["As_dh"], errors='coerce')>0)
            & (pd.to_numeric(df["X"], errors='coerce')>=16000)& (pd.to_numeric(df["X"], errors='coerce')<16500)
            & (pd.to_numeric(df["Y"], errors='coerce')>=106500)& (pd.to_numeric(df["Y"], errors='coerce')<107000)
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
    
    # fig = px.scatter_3d(df, x="X",y="Y",z="Z",color="CuT_dh")
    # fig.update_traces(marker_size=3)
    # fig.update_layout(font=dict(size=22))
    # fig.update_layout(scene_aspectmode='data')
    # fig.show()    
    import numpy as np
    import plotly.express as px
    import pandas as pd
    import plotly.io as pio
    pio.renderers.default='browser'
    def rotate_3d_coordinates1(coordinates, central_point, angle_degrees, axis):
        """
        Rotate 3D coordinates around a central point.
    
        Parameters:
            coordinates (np.array): The 3D coordinates to be rotated. Should be a 2D NumPy array with shape (N, 3),
                                    where N is the number of points, and each row represents the (x, y, z) coordinates.
            central_point (np.array): The central point of rotation. Should be a 1D NumPy array with shape (3,) representing
                                      the (x, y, z) coordinates of the central point.
            angle_degrees (float): The angle of rotation in degrees.
            axis (str): The axis of rotation. It can be 'x', 'y', or 'z'.
    
        Returns:
            np.array: The rotated 3D coordinates.
        """
        angle_rad = np.radians(angle_degrees)
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
    
        if axis == 'x':
            rotation_matrix = np.array([[1, 0, 0],
                                        [0, cos_theta, -sin_theta],
                                        [0, sin_theta, cos_theta]])
        elif axis == 'y':
            rotation_matrix = np.array([[cos_theta, 0, sin_theta],
                                        [0, 1, 0],
                                        [-sin_theta, 0, cos_theta]])
        elif axis == 'z':
            rotation_matrix = np.array([[cos_theta, -sin_theta, 0],
                                        [sin_theta, cos_theta, 0],
                                        [0, 0, 1]])
        else:
            raise ValueError("Invalid axis. Use 'x', 'y', or 'z'.")
    
        # Translate the coordinates to the origin
        translated_coords = coordinates - central_point
    
        # Apply the rotation matrix
        rotated_coords = np.dot(translated_coords, rotation_matrix.T)
    
        # Translate the rotated coordinates back to the original position
        rotated_coords += central_point
    
        return rotated_coords
    def rotate_3d_coordinates2(coordinates, center_point, angles):
        # Convert the angles to radians
        angles_rad = np.radians(angles)
    
        # Extract the individual rotation angles
        angle_x, angle_y, angle_z = angles_rad
    
        # Translation to center the coordinates
        translated_coordinates = coordinates - center_point
    
        # Rotation matrices around the X, Y, and Z axes
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])
    
        rotation_matrix_y = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])
    
        rotation_matrix_z = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ])
    
        # Combine the rotation matrices
        rotation_matrix = rotation_matrix_z @ rotation_matrix_y @ rotation_matrix_x
    
        # Perform the rotation by multiplying the rotation matrix with the translated coordinates
        rotated_coordinates = np.dot(translated_coordinates, rotation_matrix.T)
    
        # Translate the coordinates back to their original position
        rotated_coordinates += center_point
    
        return rotated_coordinates
    coordinates = np.array(df[["X", "Y", "Z"]])

    central_point = np.array([16249, 106749, 2749])
    angle_degrees = np.array([30, 30, 30])  
    rotated_coordinates = rotate_3d_coordinates2(coordinates, central_point, angle_degrees)
    grade = np.array(df['CuT_dh'])
    coordinates_df = pd.DataFrame(coordinates,columns=['X','Y','Z'])
    rotated_coordinates_df = pd.DataFrame(rotated_coordinates,columns=['X','Y','Z'])
    coordinates_df['grade'] = grade
    rotated_coordinates_df['grade'] = grade
    # Sample 3D coordinates (replace this with your own coordinates)
    # coordinates = np.array([[1, 2, 3],
    #                         [4, 5, 6],
    #                         [7, 8, 9]])
    
    # Central point of rotation (replace this with your desired central point)
    # central_point = np.array([4, 5, 6])
    
    
    # # Perform rotation
    # angle_degrees = np.array([90, 0, 180])  
    # rotated_coordinates = rotate_3d_coordinates2(coordinates, central_point, angle_degrees)
    
    # grade = [1,2,3]
    # coordinates_df = pd.DataFrame(coordinates,columns=['X','Y','Z'])
    # rotated_coordinates_df = pd.DataFrame(rotated_coordinates,columns=['X','Y','Z'])
    # coordinates_df['grade'] = grade
    # rotated_coordinates_df['grade'] = grade
    
    fig = px.scatter_3d(coordinates_df, x="X",y="Y",z="Z",color='grade')
    fig.update_traces(marker_size=5)
    fig.update_layout(font=dict(size=22))
    fig.update_layout(scene_aspectmode='cube')
    fig.show()    
    fig = px.scatter_3d(rotated_coordinates_df, x="X",y="Y",z="Z",color='grade')
    fig.update_traces(marker_size=5)
    fig.update_layout(font=dict(size=22))
    fig.update_layout(scene_aspectmode='cube')
    fig.show()    
