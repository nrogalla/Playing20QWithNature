import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform, correlation


# extracts stimulus from metadata 
def get_stimulus_data(stim_f):
    stimdata = pd.read_csv(stim_f)
    testdata= stimdata[stimdata['trial_type'] == 'test']

    temp = pd.DataFrame(columns=['stimulus', 'r1', 'r2'])
    temp[['stimulus', 'r1', 'r2']] = testdata['stimulus'].str.rsplit('_', expand= True)
    
    for t in temp.index:
        if temp['r2'][t] is None:
            testdata['stimulus'][t] = temp['stimulus'][t]
        else: 
            testdata['stimulus'][t] = temp['stimulus'][t]+temp['r1'][t]
    return testdata

# computes the RDM from fMRI betas, sorted by given function 
# returns the RDM and the sorted list of corresponding stimuli
def compute_rdm_per_subject(response_data, meta_data, sort_function = None):
    '''
    Parameters
    ----------
    response_data: numpy array with fMRI betas per trail
    meta_data: dataframe with stimulus column
    sort_function: function that determines how the RDM is to be sorted

    Returns
    -------
    rdm: Representational distance matrix over stimuli sorted by sort_function
    stimulus_list: list of stimuli sorted as in rdm
    '''
    
    if len(response_data) != len(meta_data):
        return 'Invalid input: response data and meta data must be of same length'
    
    #link meta_data to response_data and reduce dataset to needed columns
    df_response_data = pd.DataFrame(response_data)
    full_data = pd.merge(meta_data, df_response_data, left_index = True, right_index = True)
    
    reduced_data = full_data.drop(["Unnamed: 0", "session", "run", "subject_id", "trial_id", "trial_type"], axis = 1)
    
    # average data across stimuli
    df_averaged_response_data = reduced_data.groupby(['stimulus']).mean().reset_index()
    
    if sort_function is not None: 
        df_averaged_response_data = df_averaged_response_data.assign(sort = lambda df: sort_function(df['stimulus'].tolist()))
        #exclude data that does not strictly fall into the categories
        df_averaged_response_data = df_averaged_response_data[df_averaged_response_data['sort'] != -1]
        #sort values according to category distinction
        df_averaged_response_data = df_averaged_response_data.sort_values(['sort', 'stimulus'], ignore_index = True, ascending = True).drop('sort', axis='columns')
        
    stimulus_list = df_averaged_response_data['stimulus']
    df_sorted_data = df_averaged_response_data.drop(['stimulus'], axis = 1).to_numpy()
    
    # compute rdm    
    rdm_list = pdist(df_sorted_data, metric = 'correlation')
    rdm = squareform(rdm_list)

    return rdm, stimulus_list

def compute_rdm(response_data, meta_data, sort_function = None):

    if len(response_data) != len(meta_data):
        return 'Invalid input: response data and meta data must must be of same subject number'
    rdms = []
    for i in range(len(response_data)):
        rdm, stimuli_list = compute_rdm_per_subject(response_data[i], meta_data[i], sort_function)
        rdms.append(rdm)
    #average across subject RDMs
    return np.average(rdms, axis = 0), stimuli_list

# computes the RDM from fMRI betas, sorted by given function 
# returns the RDM and the sorted list of corresponding stimuli
def compute_rdm_session(response_data, meta_data, sort_function = None):
    '''
    Parameters
    ----------
    response_data: numpy array with fMRI betas per trail
    meta_data: dataframe with stimulus column
    sort_function: function that determines how the RDM is to be sorted

    Returns
    -------
    rdm: Representational distance matrix over stimuli sorted by sort_function
    df_sorted['stimulus']: list of stimuli sorted as in rdm
    '''
    if len(response_data) != len(meta_data):
        return 'Invalid input: response data and meta data must be of same length'
    df_response_data = pd.DataFrame(response_data)
    merged_data = pd.merge(meta_data, df_response_data, left_index = True, right_index = True)
    reduced_data = merged_data.drop(["Unnamed: 0", "run", "subject_id", "trial_id", "trial_type"], axis =1)
    session_rdms = []
    n_sessions = len(np.unique(np.array(reduced_data['session'])))

    #compute rdm per session and add to list of session_rdms
    for i in range(n_sessions):
        reduced_data_session = reduced_data[reduced_data['session']==i+1]
        if sort_function is not None: 
            df_averaged_response_data = reduced_data_session.assign(sort = lambda df: sort_function(df['stimulus'].tolist()))
            df_averaged_response_data = df_averaged_response_data[df_averaged_response_data['sort'] != -1]
            df_sorted = df_averaged_response_data.sort_values(['sort', 'stimulus'], ignore_index = True, ascending = True).drop('sort', axis='columns')
            stimulus_list = df_sorted['stimulus']
            df_sorted_data= df_sorted.drop(['stimulus', 'session'], axis = 1)
            df_sorted_data = df_sorted_data.to_numpy()
        else: 
            stimulus_list = reduced_data_session['stimulus']
            df_sorted_data = reduced_data_session.drop(['stimulus', 'session'], axis = 1).to_numpy()
            
        rdm_list = pdist(df_sorted_data, metric = 'correlation')
        rdm = squareform(rdm_list)
        session_rdms.append(rdm)

    #average over all session_rdms
    subject_rdm = np.average(session_rdms, axis = 0)

    return subject_rdm, stimulus_list

' a function for plotting the RDM '
def plot_rdm(rdm, percentile=False, rescale=False, lim=[0, 1], conditions=None, con_fontsize=16, cmap=None, title=None,
             title_fontsize=16):

    """
    Plot the RDM
    Parameters
    ----------
    rdm : array or list [n_cons, n_cons]
        A representational dissimilarity matrix.
    percentile : bool True or False. Default is False.
        Rescale the values in RDM or not by displaying the percentile.
    rescale : bool True or False. Default is False.
        Rescale the values in RDM or not.
        Here, the maximum-minimum method is used to rescale the values except for the
        values on the diagnal.
    lim : array or list [min, max]. Default is [0, 1].
        The corrs view lims.
    conditions : string-array or string-list. Default is None.
        The labels of the conditions for plotting.
        conditions should contain n_cons strings, If conditions=None, the labels of conditions will be invisible.
    con_fontsize : int or float. Default is 12.
        The fontsize of the labels of the conditions for plotting.
    cmap : matplotlib colormap. Default is None.
        The colormap for RDM.
        If cmap=None, the ccolormap will be 'jet'.
    title : string-array. Default is None.
        The title of the figure.
    title_fontsize : int or float. Default is 16.
        The fontsize of the title.
    """

    if len(np.shape(rdm)) != 2 or np.shape(rdm)[0] != np.shape(rdm)[1]:

        return "Invalid input!"

    # get the number of conditions
    cons = rdm.shape[0]

    crdm = copy.deepcopy(rdm)

    # if cons=2, the RDM cannot be plotted.
    if cons == 2:
        print("The shape of RDM cannot be 2*2.")

        return None
   
    # determine if it's a square
    a, b = np.shape(crdm)
    if a != b:
        return None

    if percentile == True:

        v = np.zeros([cons * cons, 2], dtype=float)
        for i in range(cons):
            for j in range(cons):
                v[i * cons + j, 0] = crdm[i, j]

        index = np.argsort(v[:, 0])
        m = 0
        
        for i in range(cons * cons):
            if i > 0:
                if v[index[i], 0] > v[index[i - 1], 0]:
                    m = m + 1
                v[index[i], 1] = m

        v[:, 0] = v[:, 1] * 100 / m

        for i in range(cons):
            for j in range(cons):
                crdm[i, j] = v[i * cons + j, 0]
        if cmap == None:
            plt.imshow(crdm, extent=(0, 1, 0, 1), cmap=plt.cm.jet, clim=(0, 100))
        else:
            plt.imshow(crdm, extent=(0, 1, 0, 1), cmap=cmap, clim=(0, 100))

    # rescale the RDM
    elif rescale == True:

        # flatten the RDM
        vrdm = np.reshape(rdm, [cons * cons])
        # array -> set -> list
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()

        # get max & min
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]

        # rescale
        if maxvalue != minvalue:

            for i in range(cons):
                for j in range(cons):

                    # not on the diagnal
                    if i != j:
                        crdm[i, j] = float((crdm[i, j] - minvalue) / (maxvalue - minvalue))

        # plot the RDM
        min = lim[0]
        max = lim[1]
        if cmap == None:
            plt.imshow(crdm, extent=(0, 1, 0, 1), cmap=plt.cm.jet, clim=(min, max))
        else:
            plt.imshow(crdm, extent=(0, 1, 0, 1), cmap=cmap, clim=(min, max))

    else:
        # plot the RDM
        min = lim[0]
        max = lim[1]
        if cmap == None:
            plt.imshow(crdm, extent=(0, 1, 0, 1), cmap=plt.cm.jet, clim=(min, max))
        else:
            plt.imshow(crdm, extent=(0, 1, 0, 1), cmap=cmap, clim=(min, max))

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    font = {'size': 18}

    if percentile == True:
        cb.set_label("Dissimilarity (percentile)", fontdict=font)
    elif rescale == True:
        cb.set_label("Dissimilarity (Rescaling)", fontdict=font)
    else:
        cb.set_label("Dissimilarity", fontdict=font)

    if conditions is not None and conditions.any() is not None:
        step = float(1 / cons)
        x = np.arange(0.5 * step, 1 + 0.5 * step, step)
        y = np.arange(1 - 0.5 * step, -0.5 * step, -step)
        plt.xticks(x, conditions, fontsize=con_fontsize, rotation=90, ha="right")
        plt.yticks(y, conditions, fontsize=con_fontsize)
    else:
        plt.axis("off")

    plt.title(title, fontsize=title_fontsize)

    plt.show()

    return 0


# sort function for animacy distinction
def get_value_animacy(stimulus_list): 
    distinction_list = []
    for stimulus in stimulus_list:
        #animals: mammals
        if stimulus in ('cow', 'beaver', 'chipmunk', 'hippopotamus', 'horse',  'monkey', 
                        'rabbit', 'boa', 'alligator', 'butterfly', 'chest1', 'dragonfly',  'iguana', 'starfish', 'wasp'):
                        #'bamboo', 'bean'):
            distinction_list.append(0)
        else:
            distinction_list.append(1)
    return distinction_list

# sort function for indoors distinction
def get_value_indoors(stimulus_list): 
    distinction_list = []
    for stimulus in stimulus_list:
        #indoors
        if stimulus in ('altar', 'bed', 'dough', 'candelabra', 'coatrack', 'blind', 'crayon', 
                           'drain', 'drawer','easel', 'grate', 'jam', 'jar', 'joystick', 'lasanga', 'microscope', 'mousetrap', 'pan',
                          'piano', 'quill', 'ribbon', 'shredder', 'spoon', 'television', 'typewriter', 'urinal', 'wallpaper', 'whip'):
            distinction_list.append(0)
        #not classifiable
        elif stimulus in ('chest1', 'pumpkin', 'pear', 'peach', 'pacifier', 'marshmallow', 'mango', 'lemonade', 'kimono', 'key', 'uniform',
                          'kazoo', 't-shirt', 'speaker', 'simcard', 'tamale', 'donut', 'cufflink', 'crank', 'cookie', 'clipboard', 'beer', 'watch',
                          'cheese', 'brownie', 'brace', 'bobslet', 'bean', 'hulahoop', 'guacamole', 'grape', 'fudge', 'earring', 'dress', 'wig'):
            distinction_list.append(-1)
        #outdoors
        else:
            distinction_list.append(1)
    return distinction_list

# sort function for size (compared to breadbox) distinction
def get_value_size(stimulus_list): 
    distinction_list = []
    for stimulus in stimulus_list:
        #animals: mammals
        if stimulus in ('banana', 'butterfly', 'grape', 'key', 'spoon', 'wasp', 'ashtray', 'bean', 'beer', 'brace', 'brownie', 'cheese',
                        'chipmunk', 'clipboard', 'cookie', 'crayon', 'cufflink', 'donut', 'dough', 'dragonfly', 'drain', 'dress', 'earring',
                        'footprint', 'fudge','grate', 'guacamole', 'headlamp', 'horseshoe', 'jam', 'joystick', 'kazoo', 'kimono',
                        'lasanga', 'lemonade', 'mango','marshmallow', 'mosquitonet', 'mousetrap', 'pacifier', 'peach', 'pear', 'quill', 
                        'ribbon', 'simcard', 'starfish', 't-shirt', 'tamale', 'uniform', 'watch', 'whip', 'wig'):
            distinction_list.append(0)
        else:
            distinction_list.append(1)
    return distinction_list

def get_value_man_made(stimulus_list): 
    distinction_list = []
    for stimulus in stimulus_list:
        
        if stimulus in ('cow', 'beaver', 'chipmunk', 'hippopotamus', 'horse',  'monkey', 
                        'rabbit', 'boa', 'alligator', 'butterfly', 'chest1', 'dragonfly',  'iguana', 'starfish', 'wasp',
                        'bamboo', 'bean', 'banana', 'peach', 'pear', 'grape', 'mango', 'pumpkin'):
            distinction_list.append(0)
        else:
            distinction_list.append(1)
    return distinction_list

def get_value_purpose(stimulus_list): 
    distinction_list = []
    for stimulus in stimulus_list:
        # entertainment / recreational
        if stimulus in ('beachball', 'beer', 'bobsled', 'crayon', 'ferriswheel', 'hulahoop', 'piano', 'seesaw'):
             distinction_list.append(0)  
        # decorative
        elif stimulus in ('candelabra', 'cufflink', 'earring', 'ribbon', 'wallpaper'): 
            distinction_list.append(1)
        # transportation
        elif stimulus in ('boat', 'bike', 'helicopter', 'hovercraft'): 
            distinction_list.append(2)
        # practical
        elif stimulus in ('ashtray', 'axe', 'bed', 'bench', 'blind', 'brace', 'crank', 'bulldozer', 'clipboard', 'drain', 'drawer', 'easel', 'grate', 'headlamp',
                          'jar', 'joystick', 'key', 'microscope', 'mosquitonet', 'mousetrap', 'pacifier', 'pan', 'shredder', 'simcard', 'speaker', 'spoon',
                          'streetlight', 'tent', 'umbrella', 'watch', 'whip'): 
            distinction_list.append(3)
        #food
        elif stimulus in ('banana', 'bean', 'grape', 'mango', 'peach', 'pear', 'pumpkin', 'brownie', 'cheese', 'cookie', 'donut', 'dough', 'fudge', 
                          'guacamole', 'jam', 'lasagna', 'marshmallow', 'tamale'): 
            distinction_list.append(4)
        # no
        #elif stimulus in ('altar', 'banana', 'beachball'): 
         #   distinction_list.append(0)
        else:
            distinction_list.append(-1)
    return distinction_list

def permutation_corr(v1, v2, method="spearman", iter=1000):

    """
    Conduct Permutation test for correlation coefficients

    Parameters
    ----------
    v1 : array
        Vector 1.
    v2 : array
        Vector 2.
    method : string 'spearman' or 'pearson' or 'kendall' or 'similarity' or 'distance'. Default is 'spearman'.
        The method to calculate the similarities.
        If method='spearman', calculate the Spearman Correlations. If method='pearson', calculate the Pearson
        Correlations. If methd='kendall', calculate the Kendall tau Correlations. If method='similarity', calculate the
        Cosine Similarities. If method='distance', calculate the Euclidean Distances.
    iter : int. Default is 1000.
        The times for iteration.

    Returns
    -------
    p : float
        The permutation test result, p-value.
    """

    if len(v1) != len(v2):

        return "Invalid input"

    # permutation test

    if method == "spearman":
        rtest = spearmanr(v1, v2, axis = None)[0]

        ni = 0

        for i in range(iter):
            shuffled_indices = np.random.permutation(len(v1))
            v1shuffle = v1.take(shuffled_indices, 0).take(shuffled_indices, 1)#np.random.permutation(v1)
            #v2shuffle = v2.take(shuffled_indices, 0).take(shuffled_indices, 1)
            rperm = spearmanr(v1shuffle, v2, axis = None)[0]#spearmanr(v1shuffle_array, v2_array)[0]
            if rperm > rtest:
                ni = ni + 1

    if method == "pearson":
        
        rtest = pearsonr(v1, v2)[0]

        ni = 0

        for i in range(iter):
            v1shuffle = np.random.permutation(v1)
            v2shuffle = np.random.permutation(v2)
            rperm = pearsonr(v1shuffle, v2shuffle)[0]

            if rperm>rtest:
                ni = ni + 1

    p = np.float64((ni+1)/(iter+1))

    return p


' a function for calculating the Spearman correlation coefficient between two RDMs '

def rdm_correlation_spearman(RDM1, RDM2, rescale=False, permutation=False, iter=1000):

    """
    Calculate the Spearman Correlation between two RDMs

    Parameters
    ----------
    RDM1 : array [ncons, ncons]
        The RDM 1.
        The shape of RDM1 must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    RDM2 : array [ncons, ncons].
        The RDM 2.
        The shape of RDM2 must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    rescale : bool True or False. Default is False.
        Rescale the values in RDM or not.
        Here, the maximum-minimum method is used to rescale the values except for the values on the diagonal.
    permutation : bool True or False. Default is False.
        Conduct permutation test or not.
    iter : int. Default is 1000.
        The times for iteration.

    Returns
    -------
    corr : array [r, p].
        The Spearman Correlation result.
        The shape of corr is [2], including a r-value and a p-value.
    """

    if len(np.shape(RDM1)) != 2 or len(np.shape(RDM2)) != 2 or np.shape(RDM1)[0] != np.shape(RDM1)[1] or \
            np.shape(RDM2)[0] != np.shape(RDM2)[1]:

        print("\nThe shapes of two RDMs should be [ncons, ncons]!\n")

        return "Invalid input!"

    # get number of conditions
    cons = np.shape(RDM1)[0]

    # calculate the number of value above the diagonal in RDM
    n = int(cons*(cons-1)/2)

    if rescale == True:

        # flatten the RDM1
        vrdm = np.reshape(RDM1, [cons*cons])
        # array -> set -> list
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()

        # get max & min
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]

        # rescale
        if maxvalue != minvalue:

            for i in range(cons):
                for j in range(cons):

                    # not on the diagnal
                    if i != j:
                        RDM1[i, j] = float((RDM1[i, j] - minvalue) / (maxvalue - minvalue))

        # flatten the RDM2
        vrdm = np.reshape(RDM2, [cons * cons])
        # array -> set -> list
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()

        # get max & min
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]

        # rescale
        if maxvalue != minvalue:

            for i in range(cons):
                for j in range(cons):

                    # not on the diagnal
                    if i != j:
                        RDM2[i, j] = float((RDM2[i, j] - minvalue) / (maxvalue - minvalue))

    # initialize two vectors to store the values above the diagnal of two RDMs
    v1 = np.zeros([n], dtype=np.float64)
    v2 = np.zeros([n], dtype=np.float64)
    # assignment
    nn = 0
    for i in range(cons-1):
        for j in range(cons-1-i):
            v1[nn] = RDM1[i, i+j+1]
            v2[nn] = RDM2[i, i+j+1]
            nn = nn + 1

    # calculate the Spearman Correlation
    rp = np.array(spearmanr(v1, v2))

    if permutation == True:

        rp[1] = permutation_corr(RDM1, RDM2, method="spearman", iter=iter)

    return rp


def get_rdm_0_1(stimuli_list, value_function):
    distinction = value_function(stimuli_list)
    unique, counts = np.unique(np.array(distinction), return_counts = True)
    dic = dict(zip(unique, counts))
    matrix_0_1 = np.zeros([len(distinction), len(distinction)], dtype=float)
    for i in range(len(distinction)): 
        for j in range(len(distinction)): 
            if distinction[i] == distinction[j]: 
                matrix_0_1[i][j] = 0
            else: 
                matrix_0_1[i][j] = 1
    return matrix_0_1


