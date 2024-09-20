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
    
    
    class OutlierRegressionMixture(object):
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

    class MHSampler(object):
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
    import math
    for angle in angle_list:
        with open('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Cadia East\\CE_Collarsmod.txt') as f:
            lines1 = f.readlines()
        list1 = []
        for line1 in lines1[1:]:
            line = line1.split()
            row = np.array(line[0:12])
            list1.append(row)
        data1 = pd.DataFrame(list1,columns=['NAME','REGION','DRILLHOLE','X','Y','Z','DEPTH','DATE1','DATE2','D','AZIMUTH','DIP'])
        
        # str_list = ["UE035","UE041","UE040","UE055","UE054","UE056","UE100","UE101","UE099",
        #  "UE102","UE051","UE049","UE050","UE048","UE047","UE103","UE097","UE104",
        #  "UE096","UE018","UE017","UE042","UE043","UE044","UE045","UE046","UE092",
        #  "UE095","UE113","UE090","UE091A","UE094","UE013","UE011","UE009","UE010",
        #  "UE036","UE019A","UE037","UE020","UE022","UE021","UE023","UE024","UE025",
        #  "UE026","UE027","UE028","UE029","UE014","UE012","UE015"]
        str_list = list(data1['NAME'].unique())
        #str_list.sort()
        
        data_list = []
        for _ in str_list:
            str1 = _
            AZIMUTH = list(data1[data1['NAME']==str1]['AZIMUTH'])[0].astype('float64')
            DIP = list(data1[data1['NAME']==str1]['DIP'])[0].astype('float64')
            X = list(data1[data1['NAME']==str1]['X'])[0].astype('float64')
            Y = list(data1[data1['NAME']==str1]['Y'])[0].astype('float64')
            Z = list(data1[data1['NAME']==str1]['Z'])[0].astype('float64')
            
            with open('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Cadia East\\all_data.txt') as f:
                lines2 = f.readlines()
            
            list2 = []
            for line2 in lines2[1:]:
                line = line2.split()
                row = np.concatenate((np.array(line[0:6]),np.array(line[11:12])))
                list2.append(row)
                
            data2 = pd.DataFrame(list2,columns=['SAMPLE','HOLEID','PROJECTCODE','FROM','TO','AU_ppm','CU_ppm'])
            data2 = data2.dropna()
            data2 = data2[data2['HOLEID']==str1]
            data_list.append(data2)
            data2['X'] = round(X + ((data2['FROM'].astype('float64')+data2['TO'].astype('float64'))*0.5 * math.sin(math.radians(AZIMUTH)) * math.cos(math.radians(DIP))),3)
            data2['Y'] = round(Y + ((data2['FROM'].astype('float64')+data2['TO'].astype('float64'))*0.5 * math.cos(math.radians(AZIMUTH)) * math.cos(math.radians(DIP))),3)
            data2['Z'] = round(Z + ((data2['FROM'].astype('float64')+data2['TO'].astype('float64'))*0.5 * math.sin(math.radians(DIP))),3)
        data = pd.concat(data_list)
        
        data = data[(data['HOLEID']!='UE011') & (data['HOLEID']!='UE010')& (data['HOLEID']!='UE009')]
        
        data = data[(pd.to_numeric(data["AU_ppm"], errors='coerce')>0) & (pd.to_numeric(data["CU_ppm"], errors='coerce')>0)]
        data['AU_ppm'] = data['AU_ppm'].astype('float')
        data['CU_ppm'] = data['CU_ppm'].astype('float')
        data['CU_wt'] = data['CU_ppm']/10000
        
        data = data[(pd.to_numeric(data["AU_ppm"], errors='coerce')>0) & (pd.to_numeric(data["CU_wt"], errors='coerce')>=0.25)]
        pio.renderers.default='browser'
        
        data = data[(pd.to_numeric(data["X"], errors='coerce')>15500) & (pd.to_numeric(data["X"], errors='coerce')<16000) &
                    (pd.to_numeric(data["Y"], errors='coerce')>21500) & (pd.to_numeric(data["Y"], errors='coerce')<22000) &
                    (pd.to_numeric(data["Z"], errors='coerce')>5000) & (pd.to_numeric(data["Z"], errors='coerce')<5500)]
        data = data.reset_index(drop=True) 
        
        fig = px.scatter_3d(data, x="X",y="Y",z="Z",color='CU_ppm')
        fig.update_traces(marker_size=5)
        fig.update_layout(font=dict(size=22))
        fig.update_layout(scene_aspectmode='data')
        fig.show()   
        
        # data['AU_ppm'] = data['AU_ppm'].astype('float')
        # data['CU_ppm'] = data['CU_ppm'].astype('float')
        # data['CU_wt'] = data['CU_ppm']/10000
        data['log Cu_wt'] = np.log(data['CU_wt'])
        data['log AU_ppm'] = np.log(data['AU_ppm'])

        coordinates = np.array(data[["X", "Y", "Z"]])
        
        central_point = np.array([15750, 21818, 5015])
        angle_degrees = np.array([angle[0], angle[1],  angle[2]])  
        rotated_coordinates = rotate_3d_coordinates2(coordinates, central_point, angle_degrees)
        grade = np.array(data['CU_ppm'])
        coordinates_df = pd.DataFrame(coordinates,columns=['X','Y','Z'])
        rotated_coordinates_df = pd.DataFrame(rotated_coordinates,columns=['X','Y','Z'])
        coordinates_df['grade'] = grade
        rotated_coordinates_df['grade'] = grade
        # add gaussian noise
        data['X'] = round(data['X'],2)
        data['Y'] = round(data['Y'],2)
        data['Z'] = round(data['Z'],2)
        # mu, sigma = 0.1, 0.01
        data['X_rotate'] = rotated_coordinates_df['X']
        data['Y_rotate'] = rotated_coordinates_df['Y']
        data['Z_rotate'] = rotated_coordinates_df['Z']
        
 
        
        # fig = px.scatter_3d(data, x="X_rotate",y="Y_rotate",z="Z_rotate",color='CU_ppm')
        # fig.update_traces(marker_size=5)
        # fig.update_layout(font=dict(size=22))
        # fig.update_layout(scene_aspectmode='cube')
        # fig.show()    

        
        
        
        n = 10
        m = 10
        xx1 = np.arange(data["X_rotate"].min(), data["X_rotate"].max(), n).astype('float64')
        yy1 = np.arange(data["Y_rotate"].min(), data["Y_rotate"].max(), n).astype('float64')
        zz1 = np.arange(data["Z_rotate"].min(), data["Z_rotate"].max(), m).astype('float64')
        
        blocks = []
        for k in zz1:
            for j in yy1:
                for i in xx1:
                    sub_block = data.loc[(pd.to_numeric(data["X_rotate"], errors='coerce')>=i) & (pd.to_numeric(data["X_rotate"], errors='coerce')<i+n) &
                                 (pd.to_numeric(data["Y_rotate"], errors='coerce')>=j) & (pd.to_numeric(data["Y_rotate"], errors='coerce')<j+n)
                                 &(pd.to_numeric(data["Z_rotate"], errors='coerce')>=k) & (pd.to_numeric(data["Z_rotate"], errors='coerce')<k+m)]
                    blocks.append(sub_block)
        blocks1 = []
        for i,j in enumerate(blocks):
            if len(j)>=5:
                blocks1.append(j)
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
    
        # fig, axis = plt.subplots(1,1,figsize=(12,6))
        # axis.hist(df2_new.groupby(['blocks']).size(),bins=50,color='b')
        # axis.set_xlim(0,230)
        # axis.set_ylim(0,20)
        # axis.set_xlabel('Number of bore core data',fontsize=18)
        # axis.set_ylabel('Frequency',fontsize=18)
        # axis.tick_params(axis='both', which='major', labelsize=18)
        #fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Conference\\AusIMM-2023\\Fig.2.png',dpi=300) 
        
        import pymc3 as pm
        import time
        import warnings
        import matplotlib.pyplot as plt
        import plotly.express as px
        import numpy as np
        import pandas as pd    
        import plotly.io as pio
        import plotly.graph_objs as go
        import arviz as az
        from scipy import linalg, stats
        import time
    
    
        # p=0.2
        # Nsamp = 2000
        # import matplotlib.pyplot as plt
        # from scipy.stats import mode
        # # ax.set_xlim([0, 3])
        # fig,axis = plt.subplots(2,3,figsize=(22,14),sharey=False,sharex=False);  #32 48 53 56 85 122  131
        # axis = axis.ravel()
        # Rhat_list= []
        # MSE_MLE_nonoutliers = []
        # MSE_GMM_nonoutliers = []
        # MSE_MLE_outliers = []  
        # MSE_GMM_outliers = []
        # for i,j in zip([2,43,60,79,95,145],np.arange(0,6)):  # [2,43,60,79,95,145] [32,48,56,85,122,131]
        #     df3= df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_transfered'])
        #     X = np.array(df3['CuT_dh_transfered'])
        #     Y = np.array(df3['Fe_dh_transfered'])
        #     phi_x = np.vstack([X**0, X**1])
        #     logpost_outl = OutlierRegressionMixture(Y, phi_x, p)
        #     sampler =  MHSampler(logpost_outl, GaussianProposal([0.1,0.1,0.1,0.1]))
        #     chain_array = [ ]
        #     for n in range(4):
        #         np.random.seed(42)
        #         w_0 = np.random.uniform(0,1,size=2)
        #         v1_0 = np.array(invgamma.rvs(10,loc=0,scale = 0.5,random_state=0)).reshape(-1)
        #         v2_0 = np.array(invgamma.rvs(1,loc=0,scale = 1,random_state=0)).reshape(-1)
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
        # #fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Conference\\AusIMM-2023\\Fig.3.png', bbox_inches='tight',dpi=300)
        # fig, ax = plt.subplots(1, 1)
        # ax.hist(flatchain[:,2:3],bins=100,color='b')
        # ax.set_xlabel('v1')
        # ax.set_ylabel('frequency')
        
        # fig, ax = plt.subplots(1, 1)
        # ax.hist(flatchain[:,3:4],bins=100,color='b')
        # ax.set_xlabel('v2')
        # ax.set_ylabel('frequency')
    
        
        # fig, ax = plt.subplots(1, 1)
        # x = np.linspace(0.01,1,50)
        # ax.plot(x, invgamma.pdf(x,10, scale = 0.5),
        #         'r-', lw=2, alpha=0.6, label='invgamma pdf')
        # ax.set_xlabel('v1')
        # ax.set_ylabel('frequency')
        
        # fig, ax = plt.subplots(1, 1)
        # x = np.linspace(0.01,5,50)
        # ax.plot(x, invgamma.pdf(x, 1, scale = 1),
        #         'r-', lw=2, alpha=0.6, label='invgamma pdf')
        # ax.set_xlabel('v2')
        # ax.set_ylabel('frequency')
        
    
    
        # fig, ax = plt.subplots(1, 1)
        # x = np.linspace(0.01,2,50)
        # ax.plot(x, invgamma.pdf(x, 1.35, scale = 0.8),
        #         'r-', lw=2, alpha=0.6, label='invgamma pdf')
        
        # invgamma.median(alpha11,loc=0,scale=beta11)
        #lt.hist(flatchain[:,2:3],bins=50)
        # fig, ax = plt.subplots(1, 1)
        # x = np.linspace(0.01,5,50)
        # ax.plot(x, invgamma.pdf(x, 2, scale = 10),
        #         'r-', lw=2, alpha=0.6, label='invgamma pdf')
        
        
        # fig,axis = plt.subplots(2,3,figsize=(22,14),sharey=False,sharex=False);  #32 48 53 56 85 122  131
        # axis = axis.ravel()
        # Rhat_list= []
        # MSE_MLE_nonoutliers = []
        # MSE_GMM_nonoutliers = []
        # MSE_MLE_outliers = []
        # MSE_GMM_outliers = []
        # #[32,48,56,85,122,131]
        # for i,j in zip([32,48,56,85,122,131],np.arange(0,6)):
        #     df3= df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_transfered'])
        #     X = np.array(df3['CuT_dh_transfered'])
        #     Y = np.array(df3['Fe_dh_transfered'])
        #     phi_x = np.vstack([X**0, X**1])
        #     logpost_outl = OutlierRegressionMixture(Y, phi_x,p)
        #     sampler =  MHSampler(logpost_outl, GaussianProposal([1,1,1,1]))
        #     chain_array = [ ]
        #     for n in range(4):
        #         theta0 = np.random.uniform(size=4)
        #         profile_timer(sampler.run, np.array(theta0), Nsamp)
        #         #print("chain.mean, chain.std =", sampler.chain().mean(), sampler.chain().std())
        #         #print("acceptance fraction =", sampler.accept_frac())
        #         chain_array.append(sampler.chain()[10001:,:])
        #     chain_array = np.array(chain_array)
        #     flatchain = chain_array.reshape(-1, chain_array.shape[-1])
        #     #traceplots(chain_array, xnames=['b', 'a'])
        #     #rho_k, tau = autocorr(chain_array[1], 1000, plot=False)
        #     #print("chain_array.shape =", chain_array.shape)
        #     #print("chain.mean =", flatchain.mean(axis=0))
        #     #print("chain.std =", flatchain.std(axis=0))
        #     #print("tau.shape =", tau.shape)
        #     #Rhat = gelman_rubin(chain_array)
        #     #print("psrf =", Rhat)
        #     #Rhat_list.append(Rhat)
        #     wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))
        #     func_samples = np.dot(flatchain[:,:2], phi_x)
        #     post_mu = np.mean(func_samples, axis=0)
        #     percentage = []
        #     sigma11 = flatchain[:,2:3].mean(axis=0)
        #     sigma22 = flatchain[:,3:4].mean(axis=0)
        #     for x,y in zip(X,Y):
        #         resids = y - np.dot(flatchain[:,0:2].mean(axis=0), np.vstack([x**0, x**1]))
        #         S2_in, S2_out = sigma11, sigma22
        #         exp_in  = np.exp(-0.5*resids**2/S2_in)/np.sqrt(2*np.pi*S2_in)
        #         exp_out = np.exp(-0.5*resids**2/S2_out)/np.sqrt(2*np.pi*S2_out)
        #         percentage.append(p*exp_out/((1-p)*exp_in + p*exp_out))
        
        #     idx_badpoint_list = [(idx,data[0]) for idx, data in enumerate(percentage) if data >0.5]
        #     if len(idx_badpoint_list)>0:
        #         idx_badpoint_list = np.vstack(idx_badpoint_list)
        #         badpoint_index1 = list(idx_badpoint_list[:,0])
        #         badpoint_index1 = [int(item) for item in badpoint_index1]
        #         X_badpoint = X[badpoint_index1]
        #         Y_badpoint = Y[badpoint_index1]
                
        #     idx_goodpoint_list = [(idx,data[0]) for idx, data in enumerate(percentage) if data <=0.5]
        #     if len(idx_goodpoint_list)>0:
        #         idx_goodpoint_list = np.vstack(idx_goodpoint_list)
        #         goodpoint_index1 = list(idx_goodpoint_list[:,0])
        #         goodpoint_index1 = [int(item) for item in goodpoint_index1]
        #         X_goodpoint = X[goodpoint_index1]
        #         Y_goodpoint = Y[goodpoint_index1]
        #         Y_goodpoint_pred = post_mu[goodpoint_index1]
        #     MSE_MLE_outliers.append(mean_squared_error(Y,post_mu))
        #     MSE_GMM_outliers.append(mean_squared_error(Y_goodpoint,Y_goodpoint_pred))   
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
        #fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Conference\\AusIMM-2023\\Fig.4.png', bbox_inches='tight',dpi=300)
    
    
    
        # fig,axis = plt.subplots(2,3,figsize=(24,14),sharey=False,sharex=False);  #32 48 53 56 85 122  131
        # axis = axis.ravel()
        # Rhat_list= []
        # for i,j,m,c in zip([32,32,32,32,32,32],np.arange(0,6),[0.0,0.1,0.2,0.3,0.4,0.5],['y','c','b','r','b','g']):
        #     df3= df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_transfered'])
        #     X = np.array(df3['CuT_dh_transfered'])
        #     Y = np.array(df3['Fe_dh_transfered'])
        #     sigma2 = 1
        #     phi_x = np.vstack([X**0, X**1])
        #     p=m
        #     Nsamp = 20000
        #     logpost_outl = OutlierRegressionMixture(Y, phi_x, sigma2, V, p)
        #     sampler =  MHSampler(logpost_outl, GaussianProposal([1, 1]))
        #     chain_array = [ ]
        #     for n in range(4):
        #         theta0 = np.random.uniform(size=2)
        #         profile_timer(sampler.run, np.array(theta0), Nsamp)
        #         #print("chain.mean, chain.std =", sampler.chain().mean(), sampler.chain().std())
        #         #print("acceptance fraction =", sampler.accept_frac())
        #         chain_array.append(sampler.chain()[10001:,:])
        #     chain_array = np.array(chain_array)
        #     flatchain = chain_array.reshape(-1, chain_array.shape[-1])
        #     #traceplots(chain_array, xnames=['b', 'a'])
        #     rho_k, tau = autocorr(chain_array[1], 1000, plot=False)
    
        #     Rhat = gelman_rubin(chain_array)
        #     Rhat_list.append(Rhat)
        #     wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))
        #     percentage = []
        #     for x,y in zip(X,Y):
        #         resids = y - np.dot(flatchain.mean(axis=0), np.vstack([x**0, x**1]))
        #         S2_in, S2_out = 1, 1+100
        #         exp_in  = np.exp(-0.5*resids**2/S2_in)/np.sqrt(2*np.pi*S2_in)
        #         exp_out = np.exp(-0.5*resids**2/S2_out)/np.sqrt(2*np.pi*S2_out)
        #         percentage.append(p*exp_out/((1-p)*exp_in + p*exp_out))
        
        #     idx_badpoint_list = [(idx,data[0]) for idx, data in enumerate(percentage) if data >0.5]
        #     if len(idx_badpoint_list)>0:
        #         idx_badpoint_list = np.vstack(idx_badpoint_list)
        #         badpoint_index1 = list(idx_badpoint_list[:,0])
        #         badpoint_index1 = [int(item) for item in badpoint_index1]
        #         X_badpoint = X[badpoint_index1]
        #         Y_badpoint = Y[badpoint_index1]
                
        #     idx_goodpoint_list = [(idx,data[0]) for idx, data in enumerate(percentage) if data <=0.5]
        #     if len(idx_goodpoint_list)>0:
        #         idx_goodpoint_list = np.vstack(idx_goodpoint_list)
        #         goodpoint_index1 = list(idx_goodpoint_list[:,0])
        #         goodpoint_index1 = [int(item) for item in goodpoint_index1]
        #         X_goodpoint = X[goodpoint_index1]
        #         Y_goodpoint = Y[goodpoint_index1]
                
        #     if len(idx_badpoint_list)>0:
        #         axis[j].plot(X_badpoint, Y_badpoint, ls='None', color='r',marker='o', ms=5, label="Outliers")
        #         axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Non-outliers")
        #         axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='m', lw=2, label="MLE")
        #         flatchain = chain_array.reshape(-1, chain_array.shape[-1])
        #         func_samples = np.dot(flatchain[:,:2], phi_x)
        #         post_mu = np.mean(func_samples, axis=0)
        #         post_sig = np.std(func_samples, axis=0)
        #         axis[j].plot(X, post_mu, ls='--', lw=2, color=str(c), label='Posterior Mean of GMM' + '\n' + 'with' + '  ' + str(m) + '  ' + 'p')
        #         #axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GMM' + '\n' + '(95.5% confidence intervals)')
        #         axis[j].set_title('Block No.' + str(i+1),fontsize=18)
        #         axis[j].tick_params(axis='both', which='major', labelsize=18)
        #         axis[j].set_xlabel('Cu grade',fontsize=18)
        #         axis[j].set_ylabel('Fe grade',fontsize=18)
        #         axis[j].legend(loc='upper right',fontsize=16)
        #     else:
        #         axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Data")
        #         axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='m', lw=2, label="MLE")
        #         flatchain = chain_array.reshape(-1, chain_array.shape[-1])
        #         func_samples = np.dot(flatchain[:,:2], phi_x)
        #         post_mu = np.mean(func_samples, axis=0)
        #         post_sig = np.std(func_samples, axis=0)
        #         axis[j].plot(X, post_mu, ls='--', lw=2, color=str(c), label='Posterior Mean of GMM' + '\n' + 'with' + '  ' + str(m) + '  ' + 'p')
        #         #axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GMM' + '\n' + '(95.5% confidence intervals)')
        #         axis[j].set_title('Block No.' + str(i+1),fontsize=18)
        #         axis[j].tick_params(axis='both', which='major', labelsize=18)
        #         axis[j].set_xlabel('Cu grade',fontsize=18)
        #         axis[j].set_ylabel('Fe grade',fontsize=18)
        #         axis[j].legend(loc='upper right',fontsize=16)
        # fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Conference\\AusIMM-2023\\Fig.5.png', bbox_inches='tight',dpi=300)
    
    ##########extract all outliers in all blocks
        p=0.2
        outliers = []
        Nsamp = 2000
        Rhat_list= []
        df2_new=df2_new.reset_index(drop=True)
        for i in np.arange(0,len(blocks1)):
            df3= df2_new[df2_new['blocks']==i].sort_values(by=['CU_ppm'])
            X = np.array(df3['CU_ppm'])
            Y = np.array(df3['AU_ppm'])
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
            percentage = []
            for x,y in zip(X,Y):
                resids = y - np.dot(flatchain[:,0:2].mean(axis=0), np.vstack([x**0, x**1]))
                S2_in, S2_out = v11,v22
                exp_in  = np.exp(-0.5*resids**2/S2_in)/np.sqrt(2*np.pi*S2_in)
                exp_out = np.exp(-0.5*resids**2/S2_out)/np.sqrt(2*np.pi*S2_out)
                percentage.append(p*exp_out/((1-p)*exp_in + p*exp_out))
            idx_badpoint_list = [idx for idx, data in enumerate(percentage) if data >0.5]
            sub_df3 = df2_new[df2_new['blocks']==i].sort_values(by=['CU_ppm'])
            outliers.append(sub_df3.iloc[idx_badpoint_list])
        
        outliers1 = pd.concat(outliers)
        outliers1 = outliers1.reset_index(drop=True)
        
        df2_new_nonoutliers = pd.concat([df2_new,outliers1]).drop_duplicates(keep=False)
        df2_new_nonoutliers['data type'] = 'non-outliers'
        outliers1['data type'] = 'outliers'
        df2_new1 = pd.concat([df2_new_nonoutliers,outliers1])
        df2_new1 = df2_new1.reset_index(drop=True)
        outlier_number_list.append(len(outliers1))
        unique_lith.append(list(outliers1['LITH'].unique()))
        unique_alt.append(list(outliers1['AL_ALT'].unique()))
        lith.append(list(outliers1['LITH'].values))
        alt.append(list(outliers1['AL_ALT'].values))
        # fig = px.scatter_3d(outliers1, x="X_rotate",y="Y_rotate",z="Z_rotate",color="data type")
        # fig.update_traces(marker_size=3)
        # fig.update_layout(font=dict(size=22))
        # fig.update_layout(scene_aspectmode='data')
        # fig.show()    







        # p=0.2
        # Nsamp = 2000
        # import matplotlib.pyplot as plt
        # from scipy.stats import mode
        # # ax.set_xlim([0, 3])
        # fig,axis = plt.subplots(2,3,figsize=(22,14),sharey=False,sharex=False);  #32 48 53 56 85 122  131
        # axis = axis.ravel()
        # Rhat_list= []
        # MSE_MLE_nonoutliers = []
        # MSE_GMM_nonoutliers = []
        # MSE_MLE_outliers = []  
        # MSE_GMM_outliers = []
        # for i,j in zip([1,2,3,4,5,6],np.arange(0,6)):  # [2,43,60,79,95,145] [32,48,56,85,122,131]
        #     df3= df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_transfered'])
        #     X = np.array(df3['CuT_dh_transfered'])
        #     Y = np.array(df3['Fe_dh_transfered'])
        #     phi_x = np.vstack([X**0, X**1])
        #     logpost_outl = OutlierRegressionMixture(Y, phi_x, p)
        #     sampler =  MHSampler(logpost_outl, GaussianProposal([0.1,0.1,0.1,0.1]))
        #     chain_array = [ ]
        #     for n in range(4):
        #         np.random.seed(42)
        #         w_0 = np.random.uniform(0,1,size=2)
        #         v1_0 = np.array(invgamma.rvs(10,loc=0,scale = 0.5,random_state=0)).reshape(-1)
        #         v2_0 = np.array(invgamma.rvs(1,loc=0,scale = 1,random_state=0)).reshape(-1)
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
























