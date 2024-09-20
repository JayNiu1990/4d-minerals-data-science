if __name__ == '__main__':
    import warnings
    import matplotlib.pyplot as plt
    import plotly.express as px
    import numpy as np
    import pandas as pd    
    import plotly.io as pio
    import plotly.graph_objs as go
    from scipy import stats
    import random
    from sklearn.metrics import mean_squared_error
    from scipy.stats import invgamma

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
    
    def profile_timer(f, *args, **kwargs):
        """
        Times a function call f() and prints how long it took in seconds
        (to the nearest millisecond).
        :param func:  the function f to call
        :return:  same return values as f
        """
        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        print ("time to run {}: {:.3f} sec".format(f.__name__, t1-t0))
        return result
    
    
    class OutlierRegressionMixture():
        def __init__(self, y, phi_x, p):
            self.y = y
            self.phi_x = phi_x
            self.p = p

        def log_likelihood(self, theta):
            """
            Mixture likelihood accounting for outliers
            """
            w,v1,v2 = theta[0:2],theta[2:3],theta[3:4]
            resids = self.y - np.dot(w, self.phi_x)
            # Each mixture component is a Gaussian with baseline or inflated variance
            S2_in,S2_out = v1,v2
            exp_in  = np.exp(-0.5*resids**2/S2_in)/np.sqrt(2*np.pi*S2_in)
            exp_out = np.exp(-0.5*resids**2/S2_out)/np.sqrt(2*np.pi*S2_out)
            # The final log likelihood sums over the log likelihoods for each point
            logL = np.sum(np.log((1-self.p)*exp_in + self.p*exp_out))
            return logL
        def log_prior(self, theta):
            """
            Priors over parameters 
            """
            w,v1,v2 = theta[0:2],theta[2:3],theta[3:4]
            # alpha1 = stats.uniform.rvs(1,10)
            # beta1 = stats.uniform.rvs(0,0.2)
            # alpha2 = stats.uniform.rvs(0,1)
            # beta2 = stats.uniform.rvs(0,0.2)
            # DANGER:  improper uniform for now, assume data are good enough
            return 0.0 + np.log(invgamma.pdf(v1, 10, scale = 0.5)) + np.log(invgamma.pdf(v2, 1, scale = 1))
        def log_posterior(self, theta):
            logpost = self.log_prior(theta) + self.log_likelihood(theta)
            if np.isnan(logpost):
                return -np.inf
            return logpost
        def __call__(self, theta):
            return self.log_posterior(theta)
        # logpost_outl = OutlierRegressionMixture(Y, phi_x, p)
        # sampler =  MHSampler(logpost_outl, GaussianProposal([0.1,0.1,0.1,0.1]))
        # profile_timer(sampler.run, np.array(theta0), Nsamp)
        # chain_array.append(sampler.chain()[1001:,:])
    class MHSampler():
        """
        Run a Metropolis-Hastings algorithm given a Model and Proposal.
        """

        def __init__(self, model, proposal, debug=False):
            """
            Initialize a Sampler with a model, a proposal, data, and a guess
            at some reasonable starting parameters.
            :param model: callable accepting a np.array parameter vector
                of shape matching the initial guess theta0, and returning
                a probability (such as a posterior probability)
            :param proposal: callable accepting a np.array parameter vector
                of shape matching the initial guess theta0, and returning
                a proposal of the same shape, as well as the log ratio
                    log (q(theta'|theta)/q(theta|theta'))
            :param theta0: np.array of shape (Npars,)
            :param debug: Boolean flag for whether to turn on the debugging
                print messages in the sample() method
            """
            self.model = model
            self.proposal = proposal
            self._chain_thetas = [ ]
            self._chain_logPs = [ ]
            self._debug = debug

        def run(self, theta0, Nsamples):
            """
            Run the Sampler for Nsamples samples.
            """
            self._chain_thetas = [ theta0 ]
            self._chain_logPs = [ self.model(theta0) ]
            for i in range(Nsamples):
                theta, logpost = self.sample()
                self._chain_thetas.append(theta)
                self._chain_logPs.append(logpost)
            self._chain_thetas = np.array(self._chain_thetas)
            self._chain_logPs = np.array(self._chain_logPs)

        def sample(self):
            """
            Draw a single sample from the MCMC chain, and accept or reject
            using the Metropolis-Hastings criterion.
            """
            theta_old = self._chain_thetas[-1]
            logpost_old = self._chain_logPs[-1]
            theta_prop, logqratio = self.proposal(theta_old)
            if logqratio is -np.inf:
                # flag that this is a Gibbs sampler, auto-accept and skip the rest,
                # assuming the modeler knows what they're doing
                return theta_prop, logpost
            logpost = self.model(theta_prop)
            mhratio = min(1, np.exp(logpost - logpost_old - logqratio))
            if self._debug:
                # this can be useful for sanity checks
                print("theta_old, theta_prop =", theta_old, theta_prop)
                print("logpost_old, logpost_prop =", logpost_old, logpost)
                print("logqratio =", logqratio)
                print("mhratio =", mhratio)
            if np.random.uniform() < mhratio:
                return theta_prop, logpost
            else:
                return theta_old, logpost_old
            
        def chain(self):
            """
            Return a reference to the chain.
            """
            return self._chain_thetas
        
        def accept_frac(self):
            """
            Calculate and return the acceptance fraction.  Works by checking which
            parameter vectors are the same as their predecessors.
            """
            samesame = (self._chain_thetas[1:] == self._chain_thetas[:-1])
            if len(samesame.shape) == 1:
                samesame = samesame.reshape(-1, 1)
            samesame = np.all(samesame, axis=1)
            return 1.0 - (np.sum(samesame) / np.float(len(samesame)))

    # Stub for MCMC stuff

    class GaussianProposal(object):
        """
        A standard isotropic Gaussian proposal for Metropolis Random Walk.
        """
        
        def __init__(self, stepsize):
            """
            :param stepsize:  either float or np.array of shape (d,)
            """
            self.stepsize = stepsize
            
        def __call__(self, theta):
            """
            :param theta:  parameter vector = np.array of shape (d,)
            :return: tuple (logpost, logqratio)
                logpost = log (posterior) density p(y) for the proposed theta
                logqratio = log(q(x,y)/q(y,x)) for asymmetric proposals
            """
            # this proposal is symmetric so the Metropolis q-ratio is 1
            return theta + self.stepsize*(np.random.normal(size=4)),0.0 #

    angle_list = [[0,0,0]]#,[10,10,10],[20,20,20],[30,30,30],[-10,-10,-10],[-20,-20,-20],[-30,-30,-30]]
    outlier_number_list = []
    lith = []
    alt = []
    unique_lith = []
    unique_alt = []
    df_list = []
    df2_new1_list = []
    coordinate_list = []
    x_central_coordinate = []
    y_central_coordinate = []
    z_central_coordinate = []
    for angle in angle_list:
        fields = ['BHID','Fe_dh','As_dh','CuT_dh',"X","Y","Z","LITH","AL_ALT"]
        pio.renderers.default='browser'
        df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)
        #df = df.dropna()
        df = df[(pd.to_numeric(df["CuT_dh"], errors='coerce')>=0.5)& (pd.to_numeric(df["Fe_dh"], errors='coerce')>0)& (pd.to_numeric(df["As_dh"], errors='coerce')>0)]
    
        df = df[(pd.to_numeric(df["CuT_dh"], errors='coerce')>=0.5) & (pd.to_numeric(df["Fe_dh"], errors='coerce')>0)& (pd.to_numeric(df["As_dh"], errors='coerce')>0)
                & (pd.to_numeric(df["X"], errors='coerce')>=16000)& (pd.to_numeric(df["X"], errors='coerce')<16500)
                & (pd.to_numeric(df["Y"], errors='coerce')>=106500)& (pd.to_numeric(df["Y"], errors='coerce')<107000)
                & (pd.to_numeric(df["Z"], errors='coerce')>=2500)& (pd.to_numeric(df["Z"], errors='coerce')<3000)]
        
        df['LITH'] = df['LITH'].astype(int)
        df = df.reset_index(drop=True)
        df["CuT_dh"] = df["CuT_dh"].astype("float")
        df["Fe_dh"] = df["Fe_dh"].astype("float")
        df["As_dh"] = df["As_dh"].astype("float")
        df["CuT_dh_log"] = np.log10(df['CuT_dh'])
        
        fig = px.scatter_3d(df, x="X",y="Y",z="Z",color="CuT_dh_log")
        fig.update_traces(marker_size=3)
        fig.update_layout(font=dict(size=18))
        fig.update_layout(scene_aspectmode='data')
        # fig.update_layout(scene=dict(
        #     xaxis=dict(showticklabels=False),
        #     yaxis=dict(showticklabels=False),
        #     zaxis=dict(showticklabels=False),
        # ))
        fig.update_layout(scene = dict(
                            xaxis_title='',
                            yaxis_title='',
                            zaxis_title=''))
        fig.update_layout(coloraxis_colorbar_thickness=100) 
        fig.update_layout(coloraxis_colorbar_len=1)  
        fig.show()    
        
        plt.hist(np.log10(df['Fe_dh']),bins=100)



        import numpy as np
        import plotly.express as px
        import pandas as pd
        import plotly.io as pio
        pio.renderers.default='browser'
        coordinates = np.array(df[["X", "Y", "Z"]])
        
        central_point = np.array([16249, 106749, 2749])
        angle_degrees = np.array([angle[0], angle[1],  angle[2]])  
        rotated_coordinates = rotate_3d_coordinates2(coordinates, central_point, angle_degrees)
        grade = np.array(df['CuT_dh'])
        coordinates_df = pd.DataFrame(coordinates,columns=['X','Y','Z'])
        rotated_coordinates_df = pd.DataFrame(rotated_coordinates,columns=['X','Y','Z'])
        coordinates_df['grade'] = grade
        rotated_coordinates_df['grade'] = grade
    
        # add gaussian noise
        df['X'] = round(df['X'],2)
        df['Y'] = round(df['Y'],2)
        df['Z'] = round(df['Z'],2)
        # mu, sigma = 0.1, 0.01
        df['X_rotate'] = rotated_coordinates_df['X']
        df['Y_rotate'] = rotated_coordinates_df['Y']
        df['Z_rotate'] = rotated_coordinates_df['Z']
        
        # fig = px.scatter_3d(df, x="X",y="Y",z="Z",color='CuT_dh')
        # fig.update_traces(marker_size=5)
        # fig.update_layout(font=dict(size=22))
        # fig.update_layout(scene_aspectmode='cube')
        # fig.show()    
        
        # fig = px.scatter_3d(df, x="X_rotate",y="Y_rotate",z="Z_rotate",color='CuT_dh')
        # fig.update_traces(marker_size=5)
        # fig.update_layout(font=dict(size=22))
        # fig.update_layout(scene_aspectmode='cube')
        # fig.show()    

        
        df['CuT_dh_transfered'] = df['CuT_dh']#np.log(df['CuT_dh']) #stats.zscore(df['CuT_dh'])#df['CuT_dh'] ##stats.zscore(df['CuT_dh'])
        df['CuT_dh_transfered'] = round(df['CuT_dh_transfered'],3)
        
        df['Fe_dh_transfered'] = df['Fe_dh']#np.log(df['Fe_dh']) #stats.zscore(df['Fe_dh']) #df['Fe_dh'] ##stats.zscore(df['Fe_dh'])
        df['Fe_dh_transfered'] = round(df['Fe_dh_transfered'],3)
        
        df['As_dh_transfered'] = df['As_dh']#np.log(df['As_dh']) #stats.zscore(df['As_dh'])# df['As_dh'] ##stats.zscore(df['As_dh'])
        df['As_dh_transfered'] = round(df['As_dh_transfered'],3)
        
        # df_new1 = pd.concat([df['CuT_dh_log'],noise['noise']],axis=1)
        # df_new1['CuT_dh_log_noise'] = df_new1.sum(axis=1)
        # df = pd.concat([df,df_new1],axis=1)
        
        # df_new2 = pd.concat([df['Fe_dh_log'],noise['noise']],axis=1)
        # df_new2['Fe_dh_log_noise'] = df_new2.sum(axis=1)
        # df = pd.concat([df,df_new2],axis=1)
        
        # df_new3 = pd.concat([df['As_dh_log'],noise['noise']],axis=1)
        # df_new3['As_dh_log_noise'] = df_new3.sum(axis=1)
        # df = pd.concat([df,df_new3],axis=1)
        
        df2 = df[['BHID','X','Y','Z','CuT_dh','Fe_dh','As_dh','CuT_dh_transfered','Fe_dh_transfered','As_dh_transfered','X_rotate','Y_rotate','Z_rotate','LITH','AL_ALT']]
        # df2['Cu'] = df2['CuT_dh_transfered']
        # df2['Fe'] = df2['Fe_dh_transfered']
        #df2.groupby(['LITH']).size()
        #df2 = df2.loc[df2['LITH']==31]
        #df2 = df2.reset_index(drop=True)
        
        n = 100
        m = 50
        # xx1 = np.arange(16000, 16500, n).astype('float64')
        # yy1 = np.arange(106500,107000, n).astype('float64')
        # zz1 = np.arange(df2["Z_rotate"].min(), df2["Z_rotate"].max(), m).astype('float64')
    
        xx1 = np.arange(16000, 16500, n).astype('float64')
        yy1 = np.arange(106500,107000, n).astype('float64')
        zz1 = np.arange(2500, 3000, m).astype('float64')
        # xx1 = np.arange(df2["X_rotate"].min(), df2["X_rotate"].max(), n).astype('float64')
        # yy1 = np.arange(df2["Y_rotate"].min(), df2["Y_rotate"].max(), n).astype('float64')
        # zz1 = np.arange(df2["Z_rotate"].min(), df2["Z_rotate"].max(), m).astype('float64')

        ###### x min: 15887.128 x max: 16581.225
        ###### y min: 106381.778 y max: 107113.78
        ###### z min: 2404.15 z max: 3111.057
        blocks = []
        for k in zz1:
            for j in yy1:
                for i in xx1:
                    sub_block = df2.loc[(pd.to_numeric(df2["X_rotate"], errors='coerce')>=i) & (pd.to_numeric(df2["X_rotate"], errors='coerce')<i+n) &
                                 (pd.to_numeric(df2["Y_rotate"], errors='coerce')>=j) & (pd.to_numeric(df2["Y_rotate"], errors='coerce')<j+n)
                                 &(pd.to_numeric(df2["Z_rotate"], errors='coerce')>=k) & (pd.to_numeric(df2["Z_rotate"], errors='coerce')<k+m)]
                    x_central_coordinate.append(i+(1/2)*n)
                    y_central_coordinate.append(j+(1/2)*n)
                    z_central_coordinate.append(k+(1/2)*m)
                    blocks.append(sub_block)
        
        indice1 = [i for i,n in enumerate(blocks) if len(n) > 5]
        x_central_coordinate_block_morethan5 = [x_central_coordinate[index] for index in indice1]
        y_central_coordinate_block_morethan5 = [y_central_coordinate[index] for index in indice1]
        z_central_coordinate_block_morethan5 = [z_central_coordinate[index] for index in indice1]
        blocks1 = [blocks[index] for index in indice1]
        
        indice2 = [i for i,n in enumerate(blocks) if len(n) <= 5]
        x_central_coordinate_block_lessthan5 = [x_central_coordinate[index] for index in indice2]
        y_central_coordinate_block_lessthan5 = [y_central_coordinate[index] for index in indice2]
        z_central_coordinate_block_lessthan5 = [z_central_coordinate[index] for index in indice2]
        blocks2 = [blocks[index] for index in indice2]
        
        
        # for i,j in enumerate(blocks):
        #     if len(j)>5:
        #         blocks1.append(j)
        for i, j in enumerate(blocks1):
            blocks1[i]['blocks'] = i
     
        df2_new = pd.concat(blocks1)   
        block_idxs1 = np.array(df2_new['blocks'])
        n_blocks = len(df2_new['blocks'].unique())
        
        # fig = px.scatter_3d(df2_new, x="X",y="Y",z="Z",color="Cu")
        # fig.update_traces(marker_size=3)
        # fig.update_layout(font=dict(size=22))
        # fig.update_layout(scene_aspectmode='data')
        # fig.show()    
    
        fig, axis = plt.subplots(1,1,figsize=(12,8))
        axis.hist(df2_new.groupby(['blocks']).size(),bins=100,color='b')
        axis.set_xlim(0,230)
        axis.set_ylim(0,10)
        axis.set_xlabel('Number of bore core data',fontsize=24)
        axis.set_ylabel('Frequency',fontsize=24)
        axis.tick_params(axis='both', which='major', labelsize=24)
        #fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Journal paper2\\Fig.2.png',dpi=300) 
        
        subdata = df2_new[df2_new['blocks']==32]
        fig,axis = plt.subplots(1,1,figsize=(12,8),sharey=True,sharex=False)
        axis.scatter(subdata['CuT_dh'],subdata['Fe_dh'],color='m')
        axis.set_xlabel('Cu w.t%',fontsize=22)
        axis.set_ylabel('Fe w.t%',fontsize=22)
        axis.tick_params(axis='both', which='major', labelsize=22) 
        axis.legend(loc='upper right',fontsize=22)


        import time
        import warnings
        import matplotlib.pyplot as plt
        import plotly.express as px
        import numpy as np
        import pandas as pd    
        import plotly.io as pio
        import plotly.graph_objs as go
        from scipy import linalg, stats
        import time
        p=0.2
        Nsamp = 2000
        import matplotlib.pyplot as plt
        from scipy.stats import mode
        # ax.set_xlim([0, 3])
        fig,axis = plt.subplots(2,3,figsize=(22,14),sharey=False,sharex=False);  #32 48 53 56 85 122  131
        axis = axis.ravel()
        Rhat_list= []
        MSE_GMM_nonoutliers = []
        MSE_MLE_outliers = []  
        #138
        for i,j in zip([32,56,65,88,122,137],np.arange(0,6)):  #[32,56,65,88,122,137]
            df3= df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_transfered'])
            X = np.array(df3['CuT_dh_transfered'])
            Y = np.array(df3['Fe_dh_transfered'])
            phi_x = np.vstack([X**0, X**1])
            logpost_outl = OutlierRegressionMixture(Y, phi_x, p)
            sampler =  MHSampler(logpost_outl, GaussianProposal([0.1,0.1,0.1,0.1]))
            chain_array = [ ]
            for n in range(4):
                np.random.seed(1)
                w_0 = np.random.uniform(0,1,size=2)
                v1_0 = np.array(invgamma.rvs(10,loc=0,scale = 0.5,random_state=1)).reshape(-1)
                v2_0 = np.array(invgamma.rvs(1,loc=0,scale = 1,random_state=1)).reshape(-1)
                theta0 = np.concatenate((w_0,v1_0,v2_0))
                profile_timer(sampler.run, np.array(theta0), Nsamp)
                chain_array.append(sampler.chain()[1001:,:])
               
            chain_array = np.array(chain_array)
            flatchain = chain_array.reshape(-1, chain_array.shape[-1])
            wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))
            func_samples = np.dot(flatchain[:,:2], phi_x)
            post_mu = np.mean(func_samples, axis=0)
            Y_pred = np.dot(wML, phi_x)   ###MLE
            intercept = flatchain[:,:1].mean(axis=0)
            slope = flatchain[:,1:2].mean(axis=0)
            MSE_MLE_outliers.append(mean_squared_error(Y,Y_pred))

            percentage_outlier = []
            percentage_nonoutlier = []
            v11 = flatchain[:,2:3].mean(axis=0)
            v22 = flatchain[:,3:4].mean(axis=0)
            for x,y in zip(X,Y):
                resids = y - np.dot(flatchain[:,0:2].mean(axis=0), np.vstack([x**0, x**1]))
                S2_in, S2_out = v11,v22
                exp_in  = np.exp(-0.5*resids**2/S2_in)/np.sqrt(2*np.pi*S2_in)
                exp_out = np.exp(-0.5*resids**2/S2_out)/np.sqrt(2*np.pi*S2_out)
                percentage_outlier.append(p*exp_out/((1-p)*exp_in + p*exp_out))
                percentage_nonoutlier.append((1-p)*exp_in/((1-p)*exp_in + p*exp_out))
   
            idx_badpoint_list = [(idx,data1[0],data2[0]) for idx, (data1,data2) in enumerate(zip(percentage_outlier,percentage_nonoutlier)) if data1[0] > data2[0] ]
            if len(idx_badpoint_list)>0:
                idx_badpoint_list = np.vstack(idx_badpoint_list)
                badpoint_index1 = list(idx_badpoint_list[:,0])
                badpoint_index1 = [int(item) for item in badpoint_index1]
                X_badpoint = X[badpoint_index1]
                Y_badpoint = Y[badpoint_index1]
               
            idx_goodpoint_list = [(idx,data1[0],data2[0]) for idx, (data1,data2) in enumerate(zip(percentage_outlier,percentage_nonoutlier)) if data1[0] < data2[0] ]
            if len(idx_goodpoint_list)>0:
                idx_goodpoint_list = np.vstack(idx_goodpoint_list)
                goodpoint_index1 = list(idx_goodpoint_list[:,0])
                goodpoint_index1 = [int(item) for item in goodpoint_index1]
                X_goodpoint = X[goodpoint_index1]
                Y_goodpoint = Y[goodpoint_index1]
               
            MSE_GMM_nonoutliers.append(mean_squared_error(Y_goodpoint,X_goodpoint*slope+intercept))    
            if len(idx_badpoint_list)>0:
                axis[j].plot(X_badpoint, Y_badpoint, ls='None', color='r',marker='o', ms=5, label="Outliers")
                axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Non-outliers")
                axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='b', lw=3, label="MLE")
                flatchain = chain_array.reshape(-1, chain_array.shape[-1])
                func_samples = np.dot(flatchain[:,:2], phi_x)
                post_mu = np.mean(func_samples, axis=0)
                post_sig = np.std(func_samples, axis=0)
                axis[j].plot(X, post_mu, ls='--', lw=3, color='black', label="Posterior Mean of GML")
                axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GML' + '\n' + '(95.5% confidence intervals)')
                axis[j].set_title('Block No.' + str(i+1),fontsize=18)
                axis[j].tick_params(axis='both', which='major', labelsize=18)
                axis[j].set_xlabel('Cu grade',fontsize=18)
                axis[j].set_ylabel('Fe grade',fontsize=18)
            else:
                axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Data")
                axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='b', lw=3, label="MLE")
                flatchain = chain_array.reshape(-1, chain_array.shape[-1])
                func_samples = np.dot(flatchain[:,:2], phi_x)
                post_mu = np.mean(func_samples, axis=0)
                post_sig = np.std(func_samples, axis=0)
                axis[j].plot(X, post_mu, ls='--', lw=3, color='black', label="Posterior Mean of GML")
                axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GML' + '\n' + '(95.5% confidence intervals)')
                axis[j].set_title('Block No.' + str(i+1),fontsize=18)
                axis[j].tick_params(axis='both', which='major', labelsize=18)
                axis[j].set_xlabel('Cu grade',fontsize=18)
                axis[j].set_ylabel('Fe grade',fontsize=18)
        axis[j].legend(loc='center left',bbox_to_anchor=(1, 0.5),fontsize=18)
        #fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Journal paper2\\Figure5.png', bbox_inches='tight',dpi=300)
        # # fig, ax = plt.subplots(1, 1)
        # # ax.hist(flatchain[:,2:3],bins=100,color='b')
        # # ax.set_xlabel('v1')
        # # ax.set_ylabel('frequency')
        
        # # fig, ax = plt.subplots(1, 1)
        # # ax.hist(flatchain[:,3:4],bins=100,color='b')
        # # ax.set_xlabel('v2')
        # # ax.set_ylabel('frequency')
    
        # fig, ax = plt.subplots(1, 1,figsize=(12,8))
        # x1 = np.linspace(0.01,1,50)
        # ax.plot(x1, invgamma.pdf(x1,10, scale = 0.5),
        #         'k-', lw=2, label='invgamma pdf')
        # ax.set_xlabel(r'$\sigma_1^2$',fontsize=28)
        # ax.set_ylabel('Density',fontsize=24)
        # ax.tick_params(axis='both', which='major', labelsize=24) 
        # #fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Journal paper2\\Figure4-1.png', format='png', dpi=300)
        
        
        # fig, ax = plt.subplots(1, 1,figsize=(12,8))
        # x2 = np.linspace(0.01,5,50)
        # ax.plot(x2, invgamma.pdf(x2, 1, scale = 1),
        #         'k-', lw=2, label='invgamma pdf')
        # ax.set_xlabel(r'$\sigma_2^2$',fontsize=28)
        # ax.set_ylabel('Density',fontsize=24)
        # ax.tick_params(axis='both', which='major', labelsize=24)
        # #fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Journal paper2\\Figure4-2.png', format='png', dpi=300)
        
        # fig, ax = plt.subplots(1, 1,figsize=(12,8))
        # x1 = np.linspace(0.01,5,50)
        # ax.plot(x1, invgamma.pdf(x1,10, scale = 0.5),
        #         'k-', lw=2, label='$\sigma_1^2$')
        # x2 = np.linspace(0.01,5,50)
        # ax.plot(x2, invgamma.pdf(x2, 1, scale = 1),
        #         'r-', lw=2, label='$\sigma_2^2$')
        # ax.set_xlabel('Value',fontsize=28)
        # ax.set_ylabel('Probability Density',fontsize=28)
        # ax.legend(fontsize=24)
        # ax.tick_params(axis='both', which='major', labelsize=24)


    
    
        # invgamma.median(alpha11,loc=0,scale=beta11)
        #lt.hist(flatchain[:,2:3],bins=50)
        # fig, ax = plt.subplots(1, 1)
        # x = np.linspace(0.01,5,50)
        # ax.plot(x, invgamma.pdf(x, 2, scale = 10),
        #         'r-', lw=2, alpha=0.6, label='invgamma pdf')
        
        # p=0.2
        # fig,axis = plt.subplots(2,3,figsize=(22,14),sharey=False,sharex=False);  #32 48 53 56 85 122  131
        # axis = axis.ravel()
        # Rhat_list= []
        # MSE_MLE_nonoutliers = []
        # MSE_GMM_nonoutliers = []
        # MSE_MLE_outliers = []  
        # MSE_GMM_outliers = []
        # for i,j in zip([27,23,94,22,5,65],np.arange(0,6)):  # [2,43,60,79,95,145] [32,48,56,85,122,131]
        #     df3= df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_transfered'])
        #     X = np.array(df3['CuT_dh_transfered'])
        #     Y = np.array(df3['Fe_dh_transfered'])
        #     phi_x = np.vstack([X**0, X**1])
        #     logpost_outl = OutlierRegressionMixture(Y, phi_x, p)
        #     sampler =  MHSampler(logpost_outl, GaussianProposal([0.1,0.1,0.1,0.1]))
        #     chain_array = [ ]
        #     for n in range(4):
        #         np.random.seed(1)
        #         w_0 = np.random.uniform(0,1,size=2)
        #         v1_0 = np.array(invgamma.rvs(10,loc=0,scale = 0.5,random_state=1)).reshape(-1)
        #         v2_0 = np.array(invgamma.rvs(1,loc=0,scale = 1,random_state=1)).reshape(-1)
        #         theta0 = np.concatenate((w_0,v1_0,v2_0))
    
        #         profile_timer(sampler.run, np.array(theta0), Nsamp)
        #         chain_array.append(sampler.chain()[1001:,:])
                
        #     chain_array = np.array(chain_array)
        #     flatchain = chain_array.reshape(-1, chain_array.shape[-1])
            
        #     wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))
        #     func_samples = np.dot(flatchain[:,:2], phi_x)
        #     post_mu = np.mean(func_samples, axis=0)
        #     Y_pred = np.dot(wML, phi_x) 

        #     MSE_MLE_nonoutliers.append(mean_squared_error(Y,Y_pred))
        #     MSE_GMM_nonoutliers.append(mean_squared_error(Y,post_mu))
        #     percentage_outlier = []
        #     percentage_nonoutlier = []
        #     # alpha11 = mode(flatchain[:,2:3])[0][0][0]
        #     # beta11 = mode(flatchain[:,3:4])[0][0][0]
        #     # alpha22 = mode(flatchain[:,4:5])[0][0][0]
        #     # beta22 = mode(flatchain[:,5:6])[0][0][0]
        #     # alpha11 = flatchain[:,2:3].mean(axis=0)
        #     # beta11 = flatchain[:,3:4].mean(axis=0)
        #     v11 = flatchain[:,2:3].mean(axis=0)
        #     v22 = flatchain[:,3:4].mean(axis=0)
        #     for x,y in zip(X,Y):
        #         resids = y - np.dot(flatchain[:,0:2].mean(axis=0), np.vstack([x**0, x**1]))
        #         S2_in, S2_out = v11,v22
        #         exp_in  = np.exp(-0.5*resids**2/S2_in)/np.sqrt(2*np.pi*S2_in)
        #         exp_out = np.exp(-0.5*resids**2/S2_out)/np.sqrt(2*np.pi*S2_out)
        #         percentage_outlier.append(p*exp_out/((1-p)*exp_in + p*exp_out))
        #         percentage_nonoutlier.append((1-p)*exp_in/((1-p)*exp_in + p*exp_out))
    
        #     idx_badpoint_list = [(idx,data1[0],data2[0]) for idx, (data1,data2) in enumerate(zip(percentage_outlier,percentage_nonoutlier)) if data1[0] > data2[0] ]
        #     if len(idx_badpoint_list)>0:
        #         idx_badpoint_list = np.vstack(idx_badpoint_list)
        #         badpoint_index1 = list(idx_badpoint_list[:,0])
        #         badpoint_index1 = [int(item) for item in badpoint_index1]
        #         X_badpoint = X[badpoint_index1]
        #         Y_badpoint = Y[badpoint_index1]
                
        #     idx_goodpoint_list = [(idx,data1[0],data2[0]) for idx, (data1,data2) in enumerate(zip(percentage_outlier,percentage_nonoutlier)) if data1[0] < data2[0] ]
        #     if len(idx_goodpoint_list)>0:
        #         idx_goodpoint_list = np.vstack(idx_goodpoint_list)
        #         goodpoint_index1 = list(idx_goodpoint_list[:,0])
        #         goodpoint_index1 = [int(item) for item in goodpoint_index1]
        #         X_goodpoint = X[goodpoint_index1]
        #         Y_goodpoint = Y[goodpoint_index1]
                
        #     if len(idx_badpoint_list)>0:
        #         axis[j].plot(X_badpoint, Y_badpoint, ls='None', color='r',marker='o', ms=5, label="Outliers")
        #         axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Non-outliers")
        #         axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='b', lw=3, label="MLE")
        #         flatchain = chain_array.reshape(-1, chain_array.shape[-1])
        #         func_samples = np.dot(flatchain[:,:2], phi_x)
        #         post_mu = np.mean(func_samples, axis=0)
        #         post_sig = np.std(func_samples, axis=0)
        #         axis[j].plot(X, post_mu, ls='--', lw=3, color='black', label="Posterior Mean of GMM")
        #         axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GMM' + '\n' + '(95.5% confidence intervals)')
        #         axis[j].set_title('Block No.' + str(i+1),fontsize=18)
        #         axis[j].tick_params(axis='both', which='major', labelsize=18)
        #         axis[j].set_xlabel('Cu grade',fontsize=18)
        #         axis[j].set_ylabel('Fe grade',fontsize=18)
        #     else:
        #         axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Data")
        #         axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='b', lw=3, label="MLE")
        #         flatchain = chain_array.reshape(-1, chain_array.shape[-1])
        #         func_samples = np.dot(flatchain[:,:2], phi_x)
        #         post_mu = np.mean(func_samples, axis=0)
        #         post_sig = np.std(func_samples, axis=0)
        #         axis[j].plot(X, post_mu, ls='--', lw=3, color='black', label="Posterior Mean of GMM")
        #         axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GMM' + '\n' + '(95.5% confidence intervals)')
        #         axis[j].set_title('Block No.' + str(i+1),fontsize=18)
        #         axis[j].tick_params(axis='both', which='major', labelsize=18)
        #         axis[j].set_xlabel('Cu grade',fontsize=18)
        #         axis[j].set_ylabel('Fe grade',fontsize=18)
        # axis[j].legend(loc='center left',bbox_to_anchor=(1, 0.5),fontsize=18)
        # # #fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Conference\\AusIMM-2023\\Fig.4.png', bbox_inches='tight',dpi=300)
    
    

        fig,axis = plt.subplots(2,3,figsize=(24,14),sharey=False,sharex=False);  #32 48 53 56 85 122  131
        axis = axis.ravel()
        Rhat_list= []
        for i,j,m,c in zip([56,56,56,56,56,56],np.arange(0,6),[0.0,0.1,0.2,0.3,0.4,0.5],['k','k','k','k','k','k']):
            df3= df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_transfered'])
            X = np.array(df3['CuT_dh_transfered'])
            Y = np.array(df3['Fe_dh_transfered'])
            phi_x = np.vstack([X**0, X**1])
            p=m
            Nsamp = 2000
            logpost_outl = OutlierRegressionMixture(Y, phi_x, p)
            sampler =  MHSampler(logpost_outl, GaussianProposal([0.1, 0.1, 0.1, 0.1]))
            chain_array = [ ]
            for n in range(4):
                np.random.seed(1)
                w_0 = np.random.uniform(0,1,size=2)
                v1_0 = np.array(invgamma.rvs(10,loc=0,scale = 0.5,random_state=1)).reshape(-1)
                v2_0 = np.array(invgamma.rvs(1,loc=0,scale = 1,random_state=1)).reshape(-1)
                theta0 = np.concatenate((w_0,v1_0,v2_0))
    
                profile_timer(sampler.run, np.array(theta0), Nsamp)
                chain_array.append(sampler.chain()[1001:,:])
            chain_array = np.array(chain_array)
            flatchain = chain_array.reshape(-1, chain_array.shape[-1])

            wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))
            func_samples = np.dot(flatchain[:,:2], phi_x)
            percentage = []
            v11 = flatchain[:,2:3].mean(axis=0)
            v22 = flatchain[:,3:4].mean(axis=0)
            for x,y in zip(X,Y):
                resids = y - np.dot(flatchain[:,0:2].mean(axis=0), np.vstack([x**0, x**1]))
                S2_in, S2_out = v11,v22
                exp_in  = np.exp(-0.5*resids**2/S2_in)/np.sqrt(2*np.pi*S2_in)
                exp_out = np.exp(-0.5*resids**2/S2_out)/np.sqrt(2*np.pi*S2_out)
                percentage.append(p*exp_out/((1-p)*exp_in + p*exp_out))
        
            idx_badpoint_list = [(idx,data[0]) for idx, data in enumerate(percentage) if data >0.5]
            if len(idx_badpoint_list)>0:
                idx_badpoint_list = np.vstack(idx_badpoint_list)
                badpoint_index1 = list(idx_badpoint_list[:,0])
                badpoint_index1 = [int(item) for item in badpoint_index1]
                X_badpoint = X[badpoint_index1]
                Y_badpoint = Y[badpoint_index1]
                
            idx_goodpoint_list = [(idx,data[0]) for idx, data in enumerate(percentage) if data <=0.5]
            if len(idx_goodpoint_list)>0:
                idx_goodpoint_list = np.vstack(idx_goodpoint_list)
                goodpoint_index1 = list(idx_goodpoint_list[:,0])
                goodpoint_index1 = [int(item) for item in goodpoint_index1]
                X_goodpoint = X[goodpoint_index1]
                Y_goodpoint = Y[goodpoint_index1]
                
            if len(idx_badpoint_list)>0:
                axis[j].plot(X_badpoint, Y_badpoint, ls='None', color='r',marker='o', ms=5, label="Outliers")
                axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Non-outliers")
                axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='b', lw=2, label="MLE")
                flatchain = chain_array.reshape(-1, chain_array.shape[-1])
                func_samples = np.dot(flatchain[:,:2], phi_x)
                post_mu = np.mean(func_samples, axis=0)
                post_sig = np.std(func_samples, axis=0)
                axis[j].plot(X, post_mu, ls='--', lw=2, color=str(c), label='Posterior Mean of BLR-GML' + '\n' + '(p=' + str(m)+')')
                axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GMM' + '\n' + '(95.5% confidence intervals)')
                axis[j].set_title('Block No.' + str(i+1),fontsize=18)
                axis[j].tick_params(axis='both', which='major', labelsize=18)
                axis[j].set_xlabel('Cu grade',fontsize=18)
                axis[j].set_ylabel('Fe grade',fontsize=18)
                axis[j].set_ylim(0,40)
                axis[j].legend(loc='upper right',fontsize=16)
            else:
                axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Data")
                axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='b', lw=2, label="MLE")
                flatchain = chain_array.reshape(-1, chain_array.shape[-1])
                func_samples = np.dot(flatchain[:,:2], phi_x)
                post_mu = np.mean(func_samples, axis=0)
                post_sig = np.std(func_samples, axis=0)
                axis[j].plot(X, post_mu, ls='--', lw=2, color=str(c), label='Posterior Mean of BLR-GML' + '\n' + '(p=' + str(m)+')')
                axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GMM' + '\n' + '(95.5% confidence intervals)')
                axis[j].set_title('Block No.' + str(i+1),fontsize=18)
                axis[j].tick_params(axis='both', which='major', labelsize=18)
                axis[j].set_xlabel('Cu grade',fontsize=18)
                axis[j].set_ylabel('Fe grade',fontsize=18)
                axis[j].set_ylim(0,40)
                axis[j].legend(loc='upper right',fontsize=18)
        #fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Journal paper2\\Figure6.png', bbox_inches='tight',dpi=300)
        
        
        # fig, ax = plt.subplots(1, 1,figsize=(10,6))
        # x = np.linspace(0.01,1,50)
        # ax.plot(x, invgamma.pdf(x,10, scale =1),
        #         'k-', lw=2, label='invgamma pdf')
        # ax.set_xlabel(r'$\theta_1$',fontsize=20)
        # ax.set_ylabel('Density',fontsize=20)
        # ax.tick_params(axis='both', which='major', labelsize=20) 
                
        
        

        
        
        p=0.2
        fig,axis = plt.subplots(2,3,figsize=(24,14),sharey=False,sharex=False);  #32 48 53 56 85 122  131
        axis = axis.ravel()
        Rhat_list= []
        for i,j,alpha1,beta1,alpha2,beta2,c in zip([56,56,56,56,56,56],np.arange(0,6),[10,5,5,1,0.5,0.5],[0.5,0.5,1,2,2,5],[1,0.5,1,2,2,5],[1,1,1,2,5,5],['k','k','k','k','k','k']):
            df3= df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_transfered'])
            X = np.array(df3['CuT_dh_transfered'])
            Y = np.array(df3['Fe_dh_transfered'])
            phi_x = np.vstack([X**0, X**1])
            Nsamp = 2000
            logpost_outl = OutlierRegressionMixture(Y, phi_x, p)
            sampler =  MHSampler(logpost_outl, GaussianProposal([0.1, 0.1, 0.1, 0.1]))
            chain_array = [ ]
            for n in range(4):
                np.random.seed(1)
                w_0 = np.random.uniform(0,1,size=2)
                v1_0 = np.array(invgamma.rvs(alpha1,loc=0,scale = beta1,random_state=1)).reshape(-1)
                v2_0 = np.array(invgamma.rvs(alpha2,loc=0,scale = beta2,random_state=1)).reshape(-1)
                theta0 = np.concatenate((w_0,v1_0,v2_0))
    
                profile_timer(sampler.run, np.array(theta0), Nsamp)
                chain_array.append(sampler.chain()[1001:,:])
            chain_array = np.array(chain_array)
            flatchain = chain_array.reshape(-1, chain_array.shape[-1])

            wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))
            
            wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))
            func_samples = np.dot(flatchain[:,:2], phi_x)
            percentage = []
            v11 = flatchain[:,2:3].mean(axis=0)
            v22 = flatchain[:,3:4].mean(axis=0)
            for x,y in zip(X,Y):
                resids = y - np.dot(flatchain[:,0:2].mean(axis=0), np.vstack([x**0, x**1]))
                S2_in, S2_out = v11,v22
                exp_in  = np.exp(-0.5*resids**2/S2_in)/np.sqrt(2*np.pi*S2_in)
                exp_out = np.exp(-0.5*resids**2/S2_out)/np.sqrt(2*np.pi*S2_out)
                percentage.append(p*exp_out/((1-p)*exp_in + p*exp_out))
        
            idx_badpoint_list = [(idx,data[0]) for idx, data in enumerate(percentage) if data >0.5]
            if len(idx_badpoint_list)>0:
                idx_badpoint_list = np.vstack(idx_badpoint_list)
                badpoint_index1 = list(idx_badpoint_list[:,0])
                badpoint_index1 = [int(item) for item in badpoint_index1]
                X_badpoint = X[badpoint_index1]
                Y_badpoint = Y[badpoint_index1]
                
            idx_goodpoint_list = [(idx,data[0]) for idx, data in enumerate(percentage) if data <=0.5]
            if len(idx_goodpoint_list)>0:
                idx_goodpoint_list = np.vstack(idx_goodpoint_list)
                goodpoint_index1 = list(idx_goodpoint_list[:,0])
                goodpoint_index1 = [int(item) for item in goodpoint_index1]
                X_goodpoint = X[goodpoint_index1]
                Y_goodpoint = Y[goodpoint_index1]
                
            if len(idx_badpoint_list)>0:
                axis[j].plot(X_badpoint, Y_badpoint, ls='None', color='r',marker='o', ms=5, label="Outliers")
                axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Non-outliers")
                axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='b', lw=2, label="MLE")
                flatchain = chain_array.reshape(-1, chain_array.shape[-1])
                func_samples = np.dot(flatchain[:,:2], phi_x)
                post_mu = np.mean(func_samples, axis=0)      
                post_sig = np.std(func_samples, axis=0)
                axis[j].plot(X, post_mu, ls='--', lw=2, color=str(c), label='Posterior Mean of BLR-GML' + '\n' +  r"($\alpha_1^{'}$="+str(alpha1)+','+r"$\beta_1^{'}$="+str(beta1)+ ',' + r"$\alpha_2^{'}$="+str(alpha2)+','+r"$\beta_2^{'}$="+str(beta2)+')')
                axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GMM' + '\n' + '(95.5% confidence intervals)')
                axis[j].set_title('Block No.' + str(i+1),fontsize=18)
                axis[j].tick_params(axis='both', which='major', labelsize=18)
                axis[j].set_xlabel('Cu grade',fontsize=18)
                axis[j].set_ylabel('Fe grade',fontsize=18)
                axis[j].set_ylim(0,40)
                axis[j].legend(loc='upper right',fontsize=16)
            else:
                axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Data")
                axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='b', lw=2, label="MLE")
                flatchain = chain_array.reshape(-1, chain_array.shape[-1])
                func_samples = np.dot(flatchain[:,:2], phi_x)
                post_mu = np.mean(func_samples, axis=0)
                post_sig = np.std(func_samples, axis=0)
                axis[j].plot(X, post_mu, ls='--', lw=2, color=str(c), label='Posterior Mean of BLR-GML' + '\n' +  r"($\alpha_1^{'}$="+str(alpha1)+','+r"$\beta_1^{'}$="+str(beta1)+ ',' + r"$\alpha_2^{'}$="+str(alpha2)+','+r"$\beta_2^{'}$="+str(beta2)+')')
                axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GMM' + '\n' + '(95.5% confidence intervals)')
                axis[j].set_title('Block No.' + str(i+1),fontsize=18)
                axis[j].tick_params(axis='both', which='major', labelsize=18)
                axis[j].set_xlabel('Cu grade',fontsize=18)
                axis[j].set_ylabel('Fe grade',fontsize=18)
                axis[j].set_ylim(0,40)
                axis[j].legend(loc='upper right',fontsize=16)
        #fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Journal paper2\\Figure7.png', bbox_inches='tight',dpi=300)
        
        
    ##########extract all outliers in all blocks
        p=0.2
        outliers = []
        Nsamp = 2000
        Rhat_list = []
        label_list = []
        slope_list = []
        intercept_list = []
        booleans_list = []
        df2_new=df2_new.reset_index(drop=True)
        for i in np.arange(0,len(blocks1),1):#len(blocks1)
            df3= df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_transfered'])
            X = np.array(df3['CuT_dh_transfered'])
            Y = np.array(df3['Fe_dh_transfered'])
            phi_x = np.vstack([X**0, X**1])
    
            logpost_outl = OutlierRegressionMixture(Y, phi_x, p)
            sampler =  MHSampler(logpost_outl, GaussianProposal([0.1, 0.1, 0.1, 0.1]))
            chain_array = [ ]
            for n in range(4):
                np.random.seed(1)
                w_0 = np.random.uniform(0,1,size=2)
                v1_0 = np.array(invgamma.rvs(10,loc=0,scale = 0.5,random_state=1)).reshape(-1)
                v2_0 = np.array(invgamma.rvs(1,loc=0,scale = 1,random_state=1)).reshape(-1)
                theta0 = np.concatenate((w_0,v1_0,v2_0))
                profile_timer(sampler.run, np.array(theta0), Nsamp)
                chain_array.append(sampler.chain()[1001:,:])
            chain_array = np.array(chain_array)
            flatchain = chain_array.reshape(-1, chain_array.shape[-1])
            v11 = flatchain[:,2:3].mean(axis=0)
            v22 = flatchain[:,3:4].mean(axis=0)
            func_samples = np.dot(flatchain[:,:2], phi_x)
            intercept = flatchain[:,:1].mean(axis=0)
            intercept_list.append(intercept[0])
            slope = flatchain[:,1:2].mean(axis=0)
            slope_list.append(slope[0])
            percentage = []
            for x,y in zip(X,Y):
                resids = y - np.dot(flatchain[:,0:2].mean(axis=0), np.vstack([x**0, x**1]))
                S2_in, S2_out = v11,v22
                exp_in  = np.exp(-0.5*resids**2/S2_in)/np.sqrt(2*np.pi*S2_in)
                exp_out = np.exp(-0.5*resids**2/S2_out)/np.sqrt(2*np.pi*S2_out)
                percentage.append(p*exp_out/((1-p)*exp_in + p*exp_out))
            idx_badpoint_list = [idx for idx, data in enumerate(percentage) if data >0.5]
            sub_df3 = df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_transfered'])
            outlier_df = sub_df3.iloc[idx_badpoint_list]
            outlier_df['pred'] = outlier_df['CuT_dh']*slope + intercept
            xx2 = list(outlier_df['CuT_dh'])
            yy2 = list(outlier_df['Fe_dh'])
            pred = list(outlier_df['pred'])
            booleans1 = [ii<jj for ii, jj in zip(pred,yy2)]

           
            if len(outlier_df) == 0:
                label_list.append('4')
            elif sum(bool(x) for x in [ii<jj for ii, jj in zip(pred,yy2)])/ len(outlier_df) >=9/10:
                #if outlier_df['CuT_dh'].max() - outlier_df['CuT_dh'].min() > 5:
                if  sum(bool(x) for x in [ii1>3 for ii1 in outlier_df['CuT_dh']])/len(outlier_df) > 2/10:
                    label_list.append('1')
                else:
                    label_list.append('2')
            elif sum(bool(x) for x in [ii<jj for ii, jj in zip(pred,yy2)])/ len(outlier_df) <9/10:
                    label_list.append('3')
            #number1 = sum(bool(x) for x in [ii<jj for ii, jj in zip(xx2,yy2)])
            outliers.append(sub_df3.iloc[idx_badpoint_list])
        
        outliers1 = pd.concat(outliers)
        outliers1 = outliers1.reset_index(drop=True)
        
        df2_new_nonoutliers = pd.concat([df2_new,outliers1]).drop_duplicates(keep=False)
        df2_new_nonoutliers['data type'] = 'non-outliers'
        outliers1['data type'] = 'outliers'
        df2_new1 = pd.concat([df2_new_nonoutliers,outliers1])
        df2_new1 = df2_new1.reset_index(drop=True)
        outlier_number_list.append(len(outliers1))
        df_list.append(outliers1)
        unique_lith.append(list(outliers1['LITH'].unique()))
        unique_alt.append(list(outliers1['AL_ALT'].unique()))
        lith.append(list(outliers1['LITH'].values))
        alt.append(list(outliers1['AL_ALT'].values))
        df2_new1_list.append(df2_new1)
        
        
        
        df2_new1_nonoutliers = df2_new1[df2_new1['data type']=='non-outliers']
        df2_new1_outliers = df2_new1[df2_new1['data type']=='outliers']
        
        ###########all model parameters#########
        p=0.2
        Nsamp = 2000
        import matplotlib.pyplot as plt
        from scipy.stats import mode
        # ax.set_xlim([0, 3])

        Rhat_list= []
        MSE_GMM_nonoutliers = []
        MSE_MLE_outliers = []  
        badpoint_num_list = []

        fig,axis = plt.subplots(2,3,figsize=(22,14),sharey=False,sharex=False);  #32 48 53 56 85 122  131
        axis = axis.ravel()
        for i,j in zip([32,56,65,88,122,137],np.arange(0,6)):  #[32,56,65,88,122,137]
            df3= df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_transfered'])
            X = np.array(df3['CuT_dh_transfered'])
            Y = np.array(df3['Fe_dh_transfered'])
            phi_x = np.vstack([X**0, X**1])
            logpost_outl = OutlierRegressionMixture(Y, phi_x, p)
            sampler =  MHSampler(logpost_outl, GaussianProposal([0.1,0.1,0.1,0.1]))
            chain_array = [ ]
            for n in range(4):
                np.random.seed(1)
                w_0 = np.random.uniform(0,1,size=2)
                v1_0 = np.array(invgamma.rvs(10,loc=0,scale = 0.5,random_state=1)).reshape(-1)
                v2_0 = np.array(invgamma.rvs(1,loc=0,scale = 1,random_state=1)).reshape(-1)
                theta0 = np.concatenate((w_0,v1_0,v2_0))
   
                profile_timer(sampler.run, np.array(theta0), Nsamp)
                chain_array.append(sampler.chain()[1001:,:])
               
            chain_array = np.array(chain_array)
            flatchain = chain_array.reshape(-1, chain_array.shape[-1])
            flatchain_unique = np.unique(flatchain,axis=0)
            wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))
            func_samples = np.dot(flatchain[:,:2], phi_x)
            post_mu = np.mean(func_samples, axis=0)
            Y_pred = np.dot(wML, phi_x)   ###MLE
            MSE_MLE_outliers.append(mean_squared_error(Y,Y_pred))
            badpoint_num_list1 = []
            
            percentage_outlier = []
            percentage_nonoutlier = []
            for intercept,slope,v11,v22 in zip(flatchain_unique[:,:1].flatten(),flatchain_unique[:,1:2].flatten(),flatchain_unique[:,2:3].flatten(),flatchain_unique[:,3:4].flatten()):
                percentage_outlier1 = []
                percentage_nonoutlier1 = []
                for x,y in zip(X,Y):
                    resids = y - np.dot(np.concatenate((intercept.reshape(-1),slope.reshape(-1))), np.vstack([x**0, x**1]))
                    S2_in, S2_out = v11,v22
                    exp_in  = np.exp(-0.5*resids**2/S2_in)/np.sqrt(2*np.pi*S2_in)
                    exp_out = np.exp(-0.5*resids**2/S2_out)/np.sqrt(2*np.pi*S2_out)
                    exp_in = exp_in[0]
                    exp_out = exp_out[0]
                    percentage_outlier1.append(p*exp_out/((1-p)*exp_in + p*exp_out))
                    percentage_nonoutlier1.append((1-p)*exp_in/((1-p)*exp_in + p*exp_out))
                percentage_outlier.append(percentage_outlier1)
                percentage_nonoutlier.append(percentage_nonoutlier1)
                
                percentage_outlier_df = pd.DataFrame(percentage_outlier)
                percentage_nonoutlier_df = pd.DataFrame(percentage_nonoutlier)
                percentage_outlier_mean = np.array(percentage_outlier_df.mean(axis=0))
                percentage_nonoutlier_mean = np.array(percentage_nonoutlier_df.mean(axis=0))
                
            idx_badpoint_list = [(idx,data1,data2) for idx, (data1,data2) in enumerate(zip(percentage_outlier_mean,percentage_nonoutlier_mean)) if data1 > data2]
            if len(idx_badpoint_list)>0:
                idx_badpoint_list = np.vstack(idx_badpoint_list)
                badpoint_index1 = list(idx_badpoint_list[:,0])
                badpoint_index1 = [int(item) for item in badpoint_index1]
                X_badpoint = X[badpoint_index1]
                Y_badpoint = Y[badpoint_index1]
               
            idx_goodpoint_list = [(idx,data1,data2) for idx, (data1,data2) in enumerate(zip(percentage_outlier_mean,percentage_nonoutlier_mean)) if data1 < data2]
            if len(idx_goodpoint_list)>0:
                idx_goodpoint_list = np.vstack(idx_goodpoint_list)
                goodpoint_index1 = list(idx_goodpoint_list[:,0])
                goodpoint_index1 = [int(item) for item in goodpoint_index1]
                X_goodpoint = X[goodpoint_index1]
                Y_goodpoint = Y[goodpoint_index1]
            badpoint_num_list1.append(len(X_badpoint))

            MSE_GMM_nonoutliers.append(mean_squared_error(Y_goodpoint,X_goodpoint*slope+intercept))    
            if len(idx_badpoint_list)>0:
                axis[j].plot(X_badpoint, Y_badpoint, ls='None', color='r',marker='o', ms=5, label="Outliers")
                axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Non-outliers")
                axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='b', lw=3, label="MLE")
                flatchain = chain_array.reshape(-1, chain_array.shape[-1])
                func_samples = np.dot(flatchain[:,:2], phi_x)
                post_mu = np.mean(func_samples, axis=0)
                post_sig = np.std(func_samples, axis=0)
                axis[j].plot(X, post_mu, ls='--', lw=3, color='black', label="Posterior Mean of GMM")
                axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GMM' + '\n' + '(95.5% confidence intervals)')
                axis[j].set_title('Block No.' + str(i+1),fontsize=18)
                axis[j].tick_params(axis='both', which='major', labelsize=18)
                axis[j].set_xlabel('Cu grade',fontsize=18)
                axis[j].set_ylabel('Fe grade',fontsize=18)
            else:
                axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Data")
                axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='b', lw=3, label="MLE")
                flatchain = chain_array.reshape(-1, chain_array.shape[-1])
                func_samples = np.dot(flatchain[:,:2], phi_x)
                post_mu = np.mean(func_samples, axis=0)
                post_sig = np.std(func_samples, axis=0)
                axis[j].plot(X, post_mu, ls='--', lw=3, color='black', label="Posterior Mean of GMM")
                axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GMM' + '\n' + '(95.5% confidence intervals)')
                axis[j].set_title('Block No.' + str(i+1),fontsize=18)
                axis[j].tick_params(axis='both', which='major', labelsize=18)
                axis[j].set_xlabel('Cu grade',fontsize=18)
                axis[j].set_ylabel('Fe grade',fontsize=18)
        axis[j].legend(loc='center left',bbox_to_anchor=(1, 0.5),fontsize=18)
        badpoint_num_list.append(badpoint_num_list1)
            # percentage_outlier.append(percentage_outlier1)
            # percentage_nonoutlier.append(percentage_nonoutlier1)






        df3 = pd.DataFrame()
        df3['X_central'] = x_central_coordinate_block_morethan5
        df3['Y_central'] = y_central_coordinate_block_morethan5
        df3['Z_central'] = z_central_coordinate_block_morethan5
        df3['slope'] = slope_list
        df3['intercept'] = intercept_list
        df3['label'] = label_list
        
        df3.loc[(df3.label =='4'), 'label annotation'] = 'No outlier blocks' 
        df3.loc[(df3.label =='1'), 'label annotation'] = 'Cu/Fe bearing mineral blocks'
        df3.loc[(df3.label =='2'), 'label annotation'] = 'Fe-bearing mineral blocks'
        df3.loc[(df3.label =='3'), 'label annotation'] = 'Hetergeneous blocks'
        # df3 = df3.loc[(pd.to_numeric(df3["correlation"], errors='coerce')<0)]
        # fig = px.scatter_3d(df3, x="X_central",y="Y_central",z="Z_central",color='slope')
        # fig.update_traces(marker_size=8)
        # fig.update_layout(font=dict(size=22))
        # fig.update_layout(scene_aspectmode='data')
        # fig.show()  
        
        df4 = pd.DataFrame()
        df4['X_central'] = x_central_coordinate_block_lessthan5
        df4['Y_central'] = y_central_coordinate_block_lessthan5
        df4['Z_central'] = z_central_coordinate_block_lessthan5
        df4['slope'] = 0 #N/A
        df4['intercept'] = 0 #N/A
        df4['label'] = 0 #N/A

        
        
        # df3.to_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\correlation1.csv')
        # df4.to_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\correlation2.csv')
        # i=1
        # sub_df = df2_new1_nonoutliers[df2_new1_nonoutliers['blocks']==i]
        # x,y = np.array(sub_df["CuT_dh_transfered"]), np.array(sub_df["Fe_dh_transfered"])
        # pred = x*slope_list[i] + intercept_list[i]
        # plt.scatter(x,y)
        # plt.plot(x,x*slope_list[i]+intercept_list[i])
        # print(Pearson_correlation(x,pred))


        # from sklearn.linear_model import LinearRegression
        # r_squared_list = []
        # for i in range(len(blocks1)):
        #     model = LinearRegression()
        #     sub_df = df2_new1_nonoutliers[df2_new1_nonoutliers['blocks']==i]
        #     x,y = np.array(sub_df["CuT_dh_transfered"]), np.array(sub_df["Fe_dh_transfered"])
        #     model.fit(x.reshape(-1,1), y)
        #     r_squared = model.score(x.reshape(-1,1), y)
        #     r_squared_list.append(r_squared)
        
        
        
        # fig = px.scatter_3d(df2_new1, x="X",y="Y",z="Z",color="data type")
        # fig.update_traces(marker_size=3)
        # fig.update_layout(font=dict(size=22))
        # fig.update_layout(scene_aspectmode='data')
        # fig.show()    
        
        # fig = px.scatter_3d(df_list[0], x="X",y="Y",z="Z",color="CuT_dh")
        # fig.update_traces(marker_size=3)
        # fig.update_layout(font=dict(size=22))
        # fig.update_layout(scene_aspectmode='data')
        # fig.show()    
        # plt.hist(alt[0],bins=50,label = 'rotation 0 0 0')
        # plt.hist(alt[1],bins=50,label = 'rotation -10 -10 -10')
        # plt.xlabel('lithology')
        # plt.ylabel('frequency')
        # plt.legend()





        x1 = np.linspace(0.01,5,50)
        x2 = np.linspace(0.01,5,50)
        fig,axis = plt.subplots(2,3,figsize=(22,14),sharey=False,sharex=False)
        axis = axis.ravel()
        axis[0].plot(x1, invgamma.pdf(x1,10, scale = 0.5),'k-', lw=2, label='$\sigma_1^2$')
        axis[0].plot(x2, invgamma.pdf(x2, 1, scale = 1),'r-', lw=2, label='$\sigma_2^2$')
        axis[0].set_xlabel('Value',fontsize=18)
        axis[0].set_ylabel('Probability Density',fontsize=18)
        axis[0].legend(fontsize=18)
        axis[0].tick_params(axis='both', which='major', labelsize=18)
        axis[0].set_title('Block No.',fontsize=18)
        axis[1].plot(x1, invgamma.pdf(x1,5, scale = 0.5),'k-', lw=2, label='$\sigma_1^2$')
        axis[1].plot(x2, invgamma.pdf(x2, 0.5, scale = 1),'r-', lw=2, label='$\sigma_2^2$')
        axis[1].set_xlabel('Value',fontsize=18)
        axis[1].set_ylabel('Probability Density',fontsize=18)
        axis[1].legend(fontsize=18)
        axis[1].tick_params(axis='both', which='major', labelsize=18)
        
        axis[2].plot(x1, invgamma.pdf(x1,5, scale = 1),'k-', lw=2, label='$\sigma_1^2$')
        axis[2].plot(x2, invgamma.pdf(x2, 1, scale = 1),'r-', lw=2, label='$\sigma_2^2$')
        axis[2].set_xlabel('Value',fontsize=18)
        axis[2].set_ylabel('Probability Density',fontsize=18)
        axis[2].legend(fontsize=18)
        axis[2].tick_params(axis='both', which='major', labelsize=18)
        
        axis[3].plot(x1, invgamma.pdf(x1,1, scale = 2),'k-', lw=2, label='$\sigma_1^2$')
        axis[3].plot(x2, invgamma.pdf(x2, 2, scale = 2),'r-', lw=2, label='$\sigma_2^2$')
        axis[3].set_xlabel('Value',fontsize=18)
        axis[3].set_ylabel('Probability Density',fontsize=18)
        axis[3].legend(fontsize=18)
        axis[3].tick_params(axis='both', which='major', labelsize=18)
        
        axis[4].plot(x1, invgamma.pdf(x1,0.5, scale = 2),'k-', lw=2, label='$\sigma_1^2$')
        axis[4].plot(x2, invgamma.pdf(x2, 2, scale = 5),'r-', lw=2, label='$\sigma_2^2$')
        axis[4].set_xlabel('Value',fontsize=18)
        axis[4].set_ylabel('Probability Density',fontsize=18)
        axis[4].legend(fontsize=18)
        axis[4].tick_params(axis='both', which='major', labelsize=18)

        axis[5].plot(x1, invgamma.pdf(x1,0.5, scale = 5),'k-', lw=2, label='$\sigma_1^2$')
        axis[5].plot(x2, invgamma.pdf(x2, 5, scale = 5),'r-', lw=2, label='$\sigma_2^2$')
        axis[5].set_xlabel('Value',fontsize=18)
        axis[5].set_ylabel('Probability Density',fontsize=18)
        axis[5].legend(fontsize=18)
        axis[5].tick_params(axis='both', which='major', labelsize=18)
        #fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Journal paper2\\Figure8.png', bbox_inches='tight













