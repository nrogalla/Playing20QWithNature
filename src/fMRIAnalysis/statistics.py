import numpy as np
import pandas as pd
import rsatoolbox
import rsatoolbox.rdm as rsr
import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from IPython.utils import io
import warnings
import statsmodels as sm
from decimal import Decimal
import src.fMRIAnalysis.data as data


def get_models(rdm, category_name_list, subset = None):
    """provides models for the category distinctions

    Args: 
        rdm (rsatoolbox.RDMs):  rdm 
        category_name_list: name of the category distinction 
        category_name_list: list of names for the category distinctions (same lenght as list of functions)
        subset: the values by which the subset selection is made from pattern_descriptors 

    Returns:
        model_rdms (list of rsatoolbox.RDMs): RDMs corresponding to the category distinctions
        models (list of rsatoolbox.model.ModelFixed): models corresponding to the category distinction
    """
    model_rdms = []
    for i in range(len(category_name_list)): 
        if subset == None: 
            model_rdms.append(rsr.get_categorical_rdm(np.array(data.get_values_distinction(rdm.pattern_descriptors['conds'], category_name_list[i])), category_name=category_name_list[i]))
        else: 
            rdm = rsr.get_categorical_rdm(np.array(data.get_values_distinction(rdm.pattern_descriptors['conds'], category_name_list[i])), category_name=category_name_list[i])
            subset_rdm = rdm.subset_pattern(by = category_name_list[i], value = subset)
            model_rdms.append(subset_rdm)
    models = []
    for i in range(len(model_rdms)): 
        models.append(rsatoolbox.model.ModelFixed(category_name_list[i], model_rdms[i]))
    return model_rdms, models

def normalize_rdms(rdms_list): 
    '''
    normalizes rdms by zscore

    Args: 
        rdms_list (list of rsatoolbox.RDMs)
    Returns
        list of rsatoolbox.RDMs: normalized
    '''
    normalized_rdms_list = []
    for rdm in rdms_list:
        normalized_rdms_list.append(stats.zscore(rdm.get_vectors()[0]))
    return normalized_rdms_list


def get_normalized_models(model_rdms, category_name_list):
    '''
    gets dataframe f normalised models with added constant for model fitting

    Args:
        model_rdms (rsatoolbox.rdm.RDMs): model rdms 
        category_name_list: list of distinctions
    Returns: 
        pd.DataFrame: normalized model rdms
    '''
    df = pd.DataFrame(columns = category_name_list)
    normalized_rdms = normalize_rdms(model_rdms)
    for i in range(len(normalized_rdms)):
        df[category_name_list[i]] = normalized_rdms[i]
    df = sm.tools.add_constant(df)
    return df


def get_subset_animacy(data_rdm): 
    '''
    reduces the rdm to only the animals part

    Args:
        data_rdm (rsatoolbox.rdm.RDMs): full rdm
    
    Returns:
        rsatoolbox.rdm.RDM: only animals
        list of rsatoolbox.model.ModelFixed: animate models
    '''
    an_rdm = rsr.get_categorical_rdm(np.array(data.get_values_distinction(data_rdm.pattern_descriptors['conds'], 'Subset Animals')), category_name='animacy')
    animate_rdm = an_rdm.subset_pattern(by = 'animacy', value = 0)
    animate_data_rdm = data_rdm.subset_pattern(by = 'index', value = animate_rdm.pattern_descriptors['index'])
    anim_category_name_list = ['Habitat', 'Carnivore', 'Herbivore', 'Omnivore', 'Mammal', 'Reptile', 'Invertebrate']
    anim_rdms, anim_models = get_models(animate_data_rdm, anim_category_name_list)
    animate_data_rdm.pattern_descriptors['index'] = anim_rdms[0].pattern_descriptors['index']
    return animate_data_rdm, anim_models, anim_category_name_list

def get_within_across(rdm, category_name):
    '''
    computes within and across category boundary correlations 

    Args:
        rdm (rsatoolbox.rdm.RDMs): rdm
        category_name: distinction
    
    Returns:
        float: within yes correlation
        float: within no correlation
        float: across yes-no correlation
    '''
    cat_rdm = rsr.get_categorical_rdm(np.array(data.get_values_distinction(rdm.pattern_descriptors['conds'], category_name)), category_name=category_name)
    yes_rdm = cat_rdm.subset_pattern(by = category_name, value = 0)
    no_rdm = cat_rdm.subset_pattern(by = category_name, value = 1)
    yes_data_rdm = rdm.subset_pattern(by = 'index', value = yes_rdm.pattern_descriptors['index'])
    no_data_rdm = rdm.subset_pattern(by = 'index', value = no_rdm.pattern_descriptors['index'])
    across_len = len(yes_rdm.pattern_descriptors['index'])
    across_animate_data = rdm.get_matrices()[0][:across_len][:,across_len:]

    within_yes= np.average(yes_data_rdm.dissimilarities[0]) 
    within_no = np.average(no_data_rdm.dissimilarities[0])
    across = np.average(across_animate_data) 
    return within_yes, within_no, across

def get_data_frame_within_across(RDM_corr, RDM_eucl, category_name_list):
    '''
    produces a result dataframe for the within/across correlations
    Args:
        RDM_corr (rsatoolbox.rdm.RDMs): with correlation distance
        RDM_eucl (rsatoolbox.rdm.RDMs): with euclidean distance
        category_name_list: list of category distinctions
    Returns
        pd.DataFrame
    '''
    df_within_across = pd.DataFrame(columns = ['Distinction', 'Within-Yes (Correlation)', 'Within-No (Correlation)', 'Across (Correlation)',
                                            'Within-Yes (Euclidean)', 'Within-No (Euclidean)', 'Across (Euclidean)'])
    df_within_across['Distinction'] = category_name_list
    for i in range(len(category_name_list)): 
        within_yes, within_no, across = get_within_across(RDM_corr, category_name_list[i])
        df_within_across['Within-Yes (Correlation)'][i] = within_yes
        df_within_across['Within-No (Correlation)'][i] = within_no
        df_within_across['Across (Correlation)'][i] = across

        within_yes, within_no, across = get_within_across(RDM_eucl, category_name_list[i])
        df_within_across['Within-Yes (Euclidean)'][i] = within_yes
        df_within_across['Within-No (Euclidean)'][i] = within_no
        df_within_across['Across (Euclidean)'][i] = across

    pd.set_option('display.max_colwidth', None)    
    return df_within_across

def get_overall_explained_variance(model_rdms, rdm, category_name_list):
    '''
    returns explained variance of fitting with all distinctions
    Args:
        model_rdms (rsatoolbox.rdm.RDMs): model rdms 
        rdm(rsatoolbox.rdm.RDMs): data
        category_name_list (list): list of distinctions
    Returns:
        nd.array: average score over scores of participant rdm to models scores
    '''
    reg_nnls = LinearRegression(positive=True)
    score = np.zeros(len(rdm))
    df = get_normalized_models(model_rdms, category_name_list)
    for i in range(len(rdm)):
        score[i]= reg_nnls.fit(df, rdm.get_vectors()[i]).score(df, rdm.get_vectors()[i])
    return np.average(score)

def get_unique_explained_variance( rdm, category_name_list):
    model_rdms, models = get_models(rdm, category_name_list)
    constant = rsr.get_categorical_rdm(np.ones(len(model_rdms[0].to_dict()['pattern_descriptors']['index'])), category_name='constant')
    constant.dissimilarities = constant.dissimilarities +1
    model_constant= rsatoolbox.model.ModelFixed('constant', constant)
    
    c_model = [model_constant] + models
    fit, r_v, p= permutation_test_nnls_explained_variance(c_model, rdm, N=1000)

    df_explained_variance = pd.DataFrame(columns = ['Distinction', 'Explained Variance', 'P-value'])
    df_explained_variance['Distinction'] = category_name_list
    df_explained_variance['Explained Variance'] = fit[1:]
    df_explained_variance['P-value'] = p[1:]
    return df_explained_variance


    
def get_permutation_results_dataframe(category_name_list, corr, p): 
    '''
    returns permutation results in a dataframe

    Args: 
        category_name_list (list): list of distinctions
        corr (list): correlation for each distinction
        p (list): p-value for each distinction
    
    Returns: 
        pf.DataFrame
    '''
    df_res = pd.DataFrame(columns = ['Model', 'Correlation', 'P-Value'])
    df_res['Model'] = category_name_list
    df_res['Correlation'] = corr
    df_res['P-Value'] = p
    return df_res

def eval_permutation_pattern(models, data, theta=None, method='spearman', N=1000):
    """performs permutation of patterns to get a sampling distribution

    Args:
        models(rsatoolbox.model.Model or list): models to be evaluated
        data(rsatoolbox.rdm.RDMs): data to evaluate on
        theta(numpy.ndarray): parameter vector for the models
        method(string): comparison method to use
        N(int): number of samples

    Returns:
        numpy.ndarray: vector of evaluations
    """
    
    models, evaluations, theta, _ = \
        rsatoolbox.util.inference_util.input_check_model(models, theta, None, N)
    
    corr_test = np.zeros(len(models))
    for j, mod in enumerate(models):
                
        rdm_pred = mod.predict_rdm(theta=theta[j])
        
        corr_test[j] = np.mean(rsatoolbox.rdm.compare(rdm_pred, data,
                                            method))

    counts = np.zeros(len(corr_test))
    
    for i in tqdm.trange(N):
        with io.capture_output() as captured:
            rdm_p = rsatoolbox.rdm.rdms.permute_rdms(data)
        
        for j, mod in enumerate(models):
            rdm_pred = mod.predict_rdm(theta=theta[j])
           
            evaluations[i, j] = np.mean(rsatoolbox.rdm.compare(rdm_pred, rdm_p,
                                                method))
            
            if evaluations[i, j] >= corr_test[j]:
                counts[j] = counts[j] + 1
        
    p = np.float64((counts+1)/(N+1))
    return evaluations, corr_test, p

def permutation_test_nnls_explained_variance(models, data, theta=None,  N=1000):
    """performs a permutation test of the explained variance of glm with non-negative linear regression fit
    Args:
        models(rsatoolbox.model.Model or list): models to be evaluated
        data(rsatoolbox.rdm.RDMs): data to evaluate on
        theta(numpy.ndarray): parameter vector for the models
        N(int): number of samples

    Returns:
        numpy.ndarray: vector of tested fit
        numpy.ndarray: vector of explained variances for permutation
        numpy.ndarray: vector of p-values for each model

    """
    models, evaluations, theta, _ = \
        rsatoolbox.util.inference_util.input_check_model(models, theta, None, N)
    
    # compute the explained variances for the models that are supposed to be tested
    fit_test = np.zeros((len(models), len(data)))
    reg_nnls = LinearRegression(positive=True)
    df = pd.DataFrame(columns= [mod.name for mod in models])  
    
    for j, mod in enumerate(models):
        rdm_pred = mod.predict_rdm(theta=theta[j])    
        vectors = rdm_pred.get_vectors()[0] 
        if (len(np.unique(vectors[0])) != 1):
            df[mod.name] = stats.zscore(vectors)
        else: 
            df[mod.name] = vectors
       
    for k in range(len(df.columns)):
        col = df.columns[k]
        for i in range(len((data))):
            reg_nnls = LinearRegression(positive=True)
            d = data.get_vectors()[i]
            
            red_score = reg_nnls.fit(df[df.columns[df.columns != col]],d).score(df[df.columns[df.columns != col]], d)
            
            fit_test[k][i] = reg_nnls.fit(df, d).score(df, d) - red_score
        
    fit_test = np.average(fit_test, axis = 1)
    # establish distribution and count times explained variance is higher or equal to test case
    counts = np.zeros(len(fit_test))
    for i in tqdm.trange(N):
        df = pd.DataFrame(columns= [mod.name for mod in models])  
        for j, mod in enumerate(models):
            rdm_pred = mod.predict_rdm(theta=theta[j])    
            vectors = rdm_pred.get_vectors()[0] 
            if (len(np.unique(vectors[0])) != 1):
                df[mod.name] = stats.zscore(vectors)
            else: 
                df[mod.name] = vectors
        

        with io.capture_output() as captured:
            rdm_p_vectors = rsatoolbox.rdm.rdms.permute_rdms(data).get_vectors()
        eval = np.zeros(shape = len(df.columns))

        for k in range(len(df.columns)):
            #model to be tested
            col = df.columns[k]
            mod_e_var = np.zeros((len(rdm_p_vectors)))
            for m in range(len(rdm_p_vectors)):
                red_score = reg_nnls.fit(df[df.columns[df.columns != col]],rdm_p_vectors[m]).score(df[df.columns[df.columns != col]], rdm_p_vectors[m])
                #full model - reduced model variance
                
                mod_e_var[m] = reg_nnls.fit(df, rdm_p_vectors[m]).score(df, rdm_p_vectors[m]) - red_score
            eval[k] = np.average(mod_e_var)
            if eval[k] >= fit_test[k]:
                counts[k] = counts[k] + 1

        evaluations[i]= eval
    # p-value as number of times explained variance is higher or equal to the test case    
    p = np.float64((counts+1)/(N+1))
    return fit_test, evaluations, p

#functions for plotting statistics
            
def _plot_voxel_selection(df, xlabel = None, ylabel = None ):
    sns.set(rc={"figure.figsize":(20, 8)})
    
    sns.set_theme() # to make style changable from defaults use this line of code befor using set_style
    sns.set_style("white")
    ax = sns.lineplot(df.T, palette = sns.color_palette(palette='tab20')[:14], dashes = False)
    l = np.arange(0, 82, step=2)
    ax.set_xticks(l)
    if xlabel is not None: 
        ax.set_xlabel(xlabel)
    if ylabel is not None: 
        ax.set_ylabel(ylabel)

def plot_correlation_to_nc(df_corr, df_p, category_name_list):
    '''
    plots correlation of model rdms with data rdm against noise ceiling cutoffs
    with horizontal bars below the plot indicating the significance

    Args: 
        df_corr (pd.DataFrame): dataframe of correlations
        df_p(pd.DataFrame): dataframe with p-values
        category_name_list (list): list of category distinctions 
    '''
    sns.set(rc={"figure.figsize":(20, 8)})
    sns.set_theme() # to make style changable from defaults use this line of code befor using set_style
    sns.set_style("white")
    ax = sns.lineplot(df_corr.T, palette = sns.color_palette(palette='tab20')[:14], dashes = False)
    l = np.arange(0, 70, step=2)
    ax.set_xticks(l)
    ax.set_ylabel('correlation')
    ax.set_xlabel('noise ceiling cut-off')
    ax.axvline(x= 61, ymin = 0, ymax = 1, color = 'grey', ls = '--')

    xmin = -1
    xmax= 100
    y = -0.09
    colors = sns.color_palette(palette='tab20')
    c = 0
    sig = False
    for cat in category_name_list: 
        for i in range(len(df_p.T[cat] )):
            if df_p.T[cat][i] < 0.05: 
                if xmin < 0:
                    sig = True
                    xmin = float(df_p.T[cat].index[i])
                    #ensures that significance will also be visible for points
                    xmax = float(df_p.T[cat].index[i])+0.001
                else:
                    xmax = float(df_p.T[cat].index[i])
            else: 
                
                if xmin >= 0:
                    trans = ax.get_xaxis_transform()
                    ax.plot([xmin * 100,xmax * 100],[y,y], color=colors[c], transform=trans, clip_on=False)
                    xmin = -1
        if xmin >= 0:
            trans = ax.get_xaxis_transform()
            ax.plot([xmin * 100,xmax * 100],[y,y], color=colors[c], transform=trans, clip_on=False)
        if sig == True:
            y -= 0.01
        c += 1
        xmin = -1 
        sig = False
    
def plot_correlation_average_activation(response_data, meta_data, category_name_list, number_participants):
    '''
    plots correlation against number of voxels included chosen based on visual responsiveness
    (not used for final version of this analysis, but included for completeness as used in earlier stages)

    Args:
        response_data (list): list of numpy arrays containing the response data of all subjects, list has length of number of subjects, response data has length of number of voxels
        meta_data (list): list of DataFrames of Metadata per subject, list has length of number of subjects
        category_name_list (list): list of categories to plot rdms for
        number_participant (int): number of participants to include in the evaluation, used to either include or exclude participant 3
    '''
    warnings.filterwarnings("ignore")
    df = pd.DataFrame(columns = ['Model'])
    df['Model'] = category_name_list
    n_v = 0
    while n_v < 4145:
        RDM_corr = rsr.calc_rdm(data.get_dataset_subset(response_data[:number_participants], meta_data[:number_participants], n_v, None, None), method='correlation', descriptor='conds')
        model_rdms, models = get_models(RDM_corr, category_name_list)
        results = rsatoolbox.inference.eval_fixed(models,RDM_corr, method='spearman').evaluations[0]
        df[str(n_v)] = np.average(results, axis = 1)
        n_v += int(5)

    df.set_index('Model', inplace = True)
    _plot_voxel_selection(df, 'Number of voxels', 'Correlation')    


def plot_correlation_noise_ceiling(response_data, meta_data, noise_ceil, category_name_list, number_participants, method):
    '''
    plots correlation against number of voxels included chosen based on noise ceiling cutoff

    Args:
        response_data (list): list of numpy arrays containing the response data of all subjects, list has length of number of subjects, response data has length of number of voxels
        meta_data (list): list of DataFrames of Metadata per subject, list has length of number of subjects
        category_name_list (list): list of categories to plot rdms for
        number_participant (int): number of participants to include in the evaluation, used to either include or exclude participant 3
    '''
    df = pd.DataFrame(columns = ['Model'])
    df['Model'] = category_name_list
    cutoff = Decimal('0.0')
    while cutoff <= 0.6:
        RDM_corr = rsr.calc_rdm(data.get_dataset_subset(response_data[:number_participants], meta_data[:number_participants], None, noise_ceil[:number_participants], cutoff), 
                                method=method, descriptor='conds')
        model_rdms, models = get_models(RDM_corr, category_name_list)
        results = rsatoolbox.inference.eval_fixed(models,RDM_corr, method='spearman').evaluations[0]
        df[str(cutoff)] = np.average(results, axis = 1)
        cutoff += Decimal('0.01')
    df.set_index('Model', inplace = True)
    _plot_voxel_selection(df, 'Number of voxels', 'Correlation')  
 
def plot_correlation_noise_ceiling_mean(response_data, meta_data, noise_ceil, category_name_list, number_participants, method):
    '''
    plots correlation against number of voxels included chosen based on noise ceiling cutoff
    with mean_RDM computed from participant RDM for correlation (since this is not representing the right proceduce it is shown here
        but was employed in earlier parts of the analysis in an extra method instead of being incorporated as an option into the actual one)

    Args:
        response_data (list): list of numpy arrays containing the response data of all subjects, list has length of number of subjects, response data has length of number of voxels
        meta_data (list): list of DataFrames of Metadata per subject, list has length of number of subjects
        category_name_list (list): list of categories to plot rdms for
        number_participant (int): number of participants to include in the evaluation, used to either include or exclude participant 3
    '''
    df= pd.DataFrame(columns = ['Model'])
    df['Model'] = category_name_list
    cutoff = Decimal('0.0')
    while cutoff <= 0.7:
        rdm = rsr.calc_rdm(data.get_dataset_subset(response_data[:number_participants], meta_data[:number_participants], None, noise_ceil[:number_participants], cutoff), method=method, descriptor='conds')
        mean_rdm = rsatoolbox.rdm.rdms.RDMs.mean(rdm)
        model_rdms, models = get_models(mean_rdm, category_name_list)
        results = rsatoolbox.inference.eval_fixed(models,mean_rdm, method='spearman')
        df[str(cutoff)] = results.evaluations[0]
        cutoff += Decimal('0.01')
    df.set_index('Model', inplace = True)
    _plot_voxel_selection(df, 'Number of voxels', 'Correlation') 

def plot_corr_and_p(response_data, meta_data, noise_ceil, category_name_list, number_participants, method):
    '''
    plots correlation against number of voxels included chosen based on noise ceiling cutoff
    wtih correlations averaged from participant correlations

    Args:
        response_data (list): list of numpy arrays containing the response data of all subjects, list has length of number of subjects, response data has length of number of voxels
        meta_data (list): list of DataFrames of Metadata per subject, list has length of number of subjects
        noise_ceil (list): list of numpy arrays containing noise ceilings data of all subjects, list has length of number of subjects, noise ceiling has length of number of voxels
        category_name_list (list): list of categories to plot rdms for
        number_participant (int): number of participants to include in the evaluation, used to either include or exclude participant 3
        method (string): method to use as distance metric
    '''
    df_c_p = pd.DataFrame(columns = ['Model'])
    df_c_p['Model'] = category_name_list
    df_c_corr = pd.DataFrame(columns = ['Model'])
    df_c_corr['Model'] = category_name_list

    cutoff = Decimal('0.0')
    while cutoff <= 0.7:
        RDM_corr = rsr.calc_rdm(data.get_dataset_subset(response_data[:number_participants], meta_data[:number_participants], noise_ceil = noise_ceil[:number_participants], cut_off = cutoff), method=method, descriptor='conds')
        
        model_rdms, models = get_models(RDM_corr, category_name_list)
        res, corr, p= eval_permutation_pattern(models, RDM_corr, method = 'spearman')
        df_c_p[str(cutoff)] = p
        df_c_corr[str(cutoff)]= corr
        cutoff += Decimal('0.01')
    df_c_p.set_index('Model', inplace = True)
    df_c_corr.set_index('Model', inplace = True)
    plot_correlation_to_nc(df_c_corr, df_c_p, category_name_list)


def plot_corr_and_p_thesis_pearson(response_data, meta_data, noise_ceil, category_name_list, number_participants, method):
    '''
    replicates plot as depicted in the thesis, 
    mistake: correlation based on mean_rdm, p-value on mean correlation of individual rdms
    
    Args:
        response_data (list): list of numpy arrays containing the response data of all subjects, list has length of number of subjects, response data has length of number of voxels
        meta_data (list): list of DataFrames of Metadata per subject, list has length of number of subjects
        noise_ceil (list): list of numpy arrays containing noise ceilings data of all subjects, list has length of number of subjects, noise ceiling has length of number of voxels
        category_name_list (list): list of categories to plot rdms for
        number_participant (int): number of participants to include in the evaluation, used to either include or exclude participant 3
        method (string): method to use as distance metric
    '''
    df_c_p = pd.DataFrame(columns = ['Model'])
    df_c_p['Model'] = category_name_list
    df_c_corr = pd.DataFrame(columns = ['Model'])
    df_c_corr['Model'] = category_name_list

    cutoff = Decimal('0.0')
    while cutoff <= 0.7:
        rdm = rsr.calc_rdm(data.get_dataset_subset(response_data[:number_participants], meta_data[:number_participants], noise_ceil = noise_ceil[:number_participants], cut_off = cutoff), method=method, descriptor='conds')
        
        mean_rdm = rsatoolbox.rdm.rdms.RDMs.mean(rdm)
        model_rdms, models = get_models(mean_rdm, category_name_list)
        results = rsatoolbox.inference.eval_fixed(models,mean_rdm, method='spearman')
        df_c_corr[str(cutoff)] = results.evaluations[0]
        model_rdms, models = get_models(rdm, category_name_list)
        res, corr, p= eval_permutation_pattern(models, rdm, method = 'spearman')
        df_c_p[str(cutoff)] = p
        cutoff += Decimal('0.01')
    df_c_p.set_index('Model', inplace = True)
    df_c_corr.set_index('Model', inplace = True)
    plot_correlation_to_nc(df_c_corr, df_c_p, category_name_list)

def plot_corr_and_p_thesis_euclidean(normalized_response_data, response_data, meta_data, noise_ceil, category_name_list, number_participants, method):
    '''
    replicates plot as depicted in the thesis, 
    mistake: correlation based on unnormalized, p-value on normalized
    
    Args:
        response_data (list): list of numpy arrays containing the response data of all subjects, list has length of number of subjects, response data has length of number of voxels
        meta_data (list): list of DataFrames of Metadata per subject, list has length of number of subjects
        noise_ceil (list): list of numpy arrays containing noise ceilings data of all subjects, list has length of number of subjects, noise ceiling has length of number of voxels
        category_name_list (list): list of categories to plot rdms for
        number_participant (int): number of participants to include in the evaluation, used to either include or exclude participant 3
        method (string): method to use as distance metric
    '''
    df_c_p = pd.DataFrame(columns = ['Model'])
    df_c_p['Model'] = category_name_list
    df_c_corr = pd.DataFrame(columns = ['Model'])
    df_c_corr['Model'] = category_name_list

    cutoff = Decimal('0.0')
    while cutoff <= 0.7:
        rdm = rsr.calc_rdm(data.get_dataset_subset(response_data[:number_participants], meta_data[:number_participants], noise_ceil = noise_ceil[:number_participants], cut_off = cutoff), method=method, descriptor='conds')
        rdm_norm = rsr.calc_rdm(data.get_dataset_subset(normalized_response_data[:number_participants], meta_data[:number_participants], noise_ceil = noise_ceil[:number_participants], cut_off = cutoff), method=method, descriptor='conds')
        
        model_rdms, models = get_models(rdm, category_name_list)
        results = rsatoolbox.inference.eval_fixed(models, rdm, method='spearman').evaluations[0]
        df_c_corr[str(cutoff)] = np.average(results, axis = 1)
        model_rdms, models = get_models(rdm_norm, category_name_list)
        res, corr, p= eval_permutation_pattern(models, rdm_norm, method = 'spearman')
        df_c_p[str(cutoff)] = p
        cutoff += Decimal('0.01')
    df_c_p.set_index('Model', inplace = True)
    df_c_corr.set_index('Model', inplace = True)
    plot_correlation_to_nc(df_c_corr, df_c_p, category_name_list)

def plot_p_to_nc(response_data, meta_data, noise_ceil, category_name_list, number_participants, method):
    '''
    plots p-value against number of voxels included chosen based on noise ceiling cutoff
   

    Args:
        response_data (list): list of numpy arrays containing the response data of all subjects, list has length of number of subjects, response data has length of number of voxels
        meta_data (list): list of DataFrames of Metadata per subject, list has length of number of subjects
        noise_ceil (list): list of numpy arrays containing noise ceilings data of all subjects, list has length of number of subjects, noise ceiling has length of number of voxels
        category_name_list (list): list of categories to plot rdms for
        number_participant (int): number of participants to include in the evaluation, used to either include or exclude participant 3
        method (string): method to use as distance metric
    '''
    df = pd.DataFrame(columns = ['Model'])
    df['Model'] = category_name_list
    cutoff = Decimal('0.0')
    while cutoff <= 0.7:
        rdm = rsr.calc_rdm(data.get_dataset_subset(response_data[:number_participants], meta_data[:number_participants], noise_ceil = noise_ceil[:number_participants], cut_off = cutoff), method=method, descriptor='conds')
        model_rdms, models = get_models(rdm, category_name_list)
        res, corr, p= eval_permutation_pattern(models, rdm, method = 'spearman')
        df[str(cutoff)] = p
        cutoff += Decimal('0.01')
    df.set_index('Model', inplace = True)
    _plot_voxel_selection(df, xlabel ='Noise ceiling cutoff', ylabel = 'P-value')