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
        
        
        
        
        df3= df2_new[df2_new['blocks']==122].sort_values(by=['CuT_dh_transfered'])
        X = np.array(df3['CuT_dh_transfered'])
        Y = np.array(df3['Fe_dh_transfered'])
        phi_x = np.vstack([X**0, X**1])
        from scipy import linalg, stats
        wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))
        fig,axis = plt.subplots(1,1,figsize=(12,8),sharey=False,sharex=False); 
        axis.plot(X, Y, ls='None', color='black',marker='o', ms=10, label="data")
        axis.plot(X, np.dot(wML, phi_x), ls='--',color='b', lw=3, label="MLE")
        axis.set_title('Block No.' + str(122+1),fontsize=20)
        axis.tick_params(axis='both', which='major', labelsize=20)
        axis.set_xlabel('Cu grade',fontsize=24)
        axis.set_ylabel('Fe grade',fontsize=24)
        axis.set_xlabel('Cu grade',fontsize=24)
        axis.set_ylabel('Fe grade',fontsize=24)
        axis.legend(loc='upper right',fontsize=24)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
