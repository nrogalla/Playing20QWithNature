import numpy as np
import pandas as pd
import rsatoolbox
import rsatoolbox.data as rsd
import rsatoolbox.rdm as rsr
import tqdm
from scipy import stats
import scipy

# extracts stimulus from metadata 
def get_stimulus_data(stim_f):
    stimdata = pd.read_csv(stim_f)
    testdata= stimdata[stimdata['trial_type'] == 'test']

    temp = pd.DataFrame(columns=['stimulus', 'r1', 'r2'])
    temp[['stimulus', 'r1', 'r2']] = testdata['stimulus'].str.rsplit('_', expand= True)
    
    for t in temp.index:
         with pd.option_context('mode.chained_assignment', None):
            if temp['r2'][t] is None:
                testdata['stimulus'][t] = temp['stimulus'][t]
            else: 
                testdata['stimulus'][t] = temp['stimulus'][t]+temp['r1'][t]
    return testdata

def select_voxels_av_activation(response_data_subj, number_of_voxels):
    average_activation_per_voxel = np.average(response_data_subj, axis = 0)
    ind = np.argpartition(average_activation_per_voxel, -number_of_voxels)[-number_of_voxels:]
    sorted_ind = ind[np.argsort(ind)]
    return sorted_ind

def select_voxels_noise_cutoff(noise_ceil_sub, cutoff):
    noise_ceil_sub = noise_ceil_sub.reshape(np.shape(noise_ceil_sub)[0],)
    ind = np.where(noise_ceil_sub>cutoff)[0]
    return ind

def get_dataset_subset(response_data, meta_data, number_of_voxels, noise_ceil, cut_off): 
    data = []
    nSubj = len(response_data)
    for i in np.arange(nSubj):
        des = {'subj': i+1}
        response_data_subj = response_data[i]
        meta_data_subj = meta_data[i]
        df_response_data_subj = pd.DataFrame(response_data_subj)
        if noise_ceil is not None and cut_off is not None: 
            sorted_ind = select_voxels_noise_cutoff(noise_ceil[i], cut_off)
        else: 
            sorted_ind = select_voxels_av_activation(response_data_subj, number_of_voxels)
        full_data = pd.merge(meta_data_subj, df_response_data_subj, left_index = True, right_index = True).drop(["Unnamed: 0", "session", "run", "subject_id", "trial_id", "trial_type"], axis = 1)
        
        stimulus_list_subj_session = full_data['stimulus']
        
        response_data_subj_sess = full_data.drop(['stimulus'], axis = 1).to_numpy()
        
        chn_des = {'voxels': np.array([x for x in df_response_data_subj.columns])}
        obs_des = {'conds': np.array([str(x) for x in stimulus_list_subj_session])} 
        data_obj = rsd.Dataset(measurements=response_data_subj_sess,
                                descriptors=des,
                                obs_descriptors=obs_des,
                                channel_descriptors=chn_des)
        data_subset = data_obj.subset_channel(by = 'voxels', value = sorted_ind )
        data_av, stim_list_average, _ = rsatoolbox.data.average_dataset_by(data_subset, 'conds') 
        
        obs_des = {'conds': np.array([str(x) for x in stim_list_average])} 
        
        data_obj_av = rsd.Dataset(measurements=data_av,
                                descriptors=des,
                                obs_descriptors=obs_des,
                                channel_descriptors=data_subset.channel_descriptors)
        data.append(data_obj_av)
        
    return data

def sort_rdm(rdm, sort_function):
   
    conds_df = pd.DataFrame(columns = ['stimulus'])
    conds_df['stimulus'] = rdm.pattern_descriptors['conds']
    conds_df = conds_df.assign(sort = lambda df: sort_function(df['stimulus'].tolist()))
    #exclude data that does not strictly fall into the categories
    conds_df = conds_df[conds_df['sort'] != -1]
    #sort values according to category distinction
    sorted_conds_df = conds_df.sort_values(['sort', 'stimulus'], ascending = True).drop('sort', axis='columns')
    sorted_ind = sorted_conds_df.index
    rdm.reorder(sorted_ind)
    return rdm

def get_models(mean_RDM_corr, functions, category_name_list, subset = None):
    if len(functions) != len(category_name_list):
        raise ValueError('number of supplied functions should equal length of category_name_list')
    model_rdms = []
    for i in range(len(functions)): 
        if subset == None: 
            model_rdms.append(rsr.get_categorical_rdm(np.array(functions[i](mean_RDM_corr.pattern_descriptors['conds'])), category_name=category_name_list[i]))
        else: 
            rdm = rsr.get_categorical_rdm(np.array(functions[i](mean_RDM_corr.pattern_descriptors['conds'])), category_name=category_name_list[i])
            subset_rdm = rdm.subset_pattern(by = category_name_list[i], value = subset)
            model_rdms.append(subset_rdm)
    models = []
    for i in range(len(model_rdms)): 
        models.append(rsatoolbox.model.ModelFixed(category_name_list[i], model_rdms[i]))
    return model_rdms, models

# sort function for animacy distinction
def get_value_animacy_pictures(stimulus_list): 
    
    distinction_list = []
    for stimulus in stimulus_list:
        if stimulus in ('cow', 'beaver', 'chipmunk', 'hippopotamus', 'horse',  'monkey', 
                        'rabbit', 'alligator', 'butterfly', 'iguana', 'starfish', 'wasp', 'dragonfly', 'chest1', 'bamboo'):
                        #'bamboo', 'bean'):
            distinction_list.append(0)
        else:
            distinction_list.append(1)
    return distinction_list

def get_value_animals_pictures(stimulus_list): 
    
    distinction_list = []
    for stimulus in stimulus_list:
        if stimulus in ('cow', 'beaver', 'chipmunk', 'hippopotamus', 'horse',  'monkey', 
                        'rabbit', 'alligator', 'butterfly', 'iguana', 'starfish', 'wasp', 'dragonfly'):
                        #'bamboo', 'bean'):
            distinction_list.append(0)
        else:
            distinction_list.append(1)
    return distinction_list

def normalize_rdms(rdms_list): 
    normalized_rdms_list = []
    for rdm in rdms_list:
        normalized_rdms_list.append(stats.zscore(rdm.get_vectors()[0]))
    return normalized_rdms_list

# sort function for indoors distinction
def get_value_indoors_pictures(stimulus_list): 
    distinction_list = []
    for stimulus in stimulus_list:
        #indoors
        if stimulus in ('altar', 'ashtray', 'banana', 'beachball', 'bed', 'beer', 'blind', 'boa', 'bobsled', 'brace', 'brownie', 'candelabra', 'cheese', 'chest1', 
                        'clipboard', 'coatrack', 'cookie', 'crank', 'crayon', 'cufflink', 'donut', 'dough', 'drawer', 'dress', 'earring', 'fudge', 'guacamole', 'horseshoe',
                        'jar', 'joystick', 'kazoo', 'key', 'kimono', 'lasagna', 'lemonade', 'mango', 'microscope', 'mosquitonet', 'mousetrap', 'pacifier', 'pan', 
                        'piano', 'quill', 'ribbon', 'shredder', 'simcard', 'speaker', 'spoon', 'tamale', 'television', 't-shirt', 'typewriter', 'urinal', 
                        'wallpaper', 'watch', 'wig'):
            distinction_list.append(0)
        else:
            distinction_list.append(1)
    return distinction_list

# sort function for size (compared to breadbox) distinction
def get_value_size_pictures(stimulus_list): 
    distinction_list = []
    for stimulus in stimulus_list:
        'smaller'
        if stimulus in ('banana', 'boa', 'butterfly', 'grape', 'key', 'spoon', 'wasp', 'ashtray', 'bean', 'beer', 'brownie', 'dress', 'cheese',
                        'chipmunk', 'clipboard', 'cookie', 'crayon', 'crank', 'cufflink', 'donut', 'dough', 'dragonfly', 'earring',
                        'footprint', 'fudge', 'guacamole', 'headlamp', 'horseshoe', 'jam', 'joystick', 'kazoo',
                        'lasagna', 'lemonade', 'mango','marshmallow', 'mosquitonet', 'mousetrap', 'pacifier', 'peach', 'pear', 'quill', 
                        'ribbon', 'simcard', 'speaker', 'starfish', 't-shirt', 'tamale', 'uniform', 'watch', 'whip', 'wig'):
            distinction_list.append(0)
        else:
            distinction_list.append(1)
    return distinction_list

def get_value_man_made_chatGPT(stimulus_list): 
    distinction_list = []
    for stimulus in stimulus_list:
        #natural
        if stimulus in ('pumpkin', 'bamboo', 'beaver', 'nest', 'bean', 'cheese', 'pear', 'monkey', 'wasp', 'rabbit', 
                        'cow', 'starfish', 'horse', 'banana', 'mango', 'grape', 'chipmunk', 'alligator', 
                         'iguana', 'butterfly', 'hippopotamus', 'dragonfly', 'stalagamite', 
                         'peach', 'chest1'):
            distinction_list.append(0)
        else:
            distinction_list.append(1)
    return distinction_list

def get_value_entertainment_chatGPT(stimulus_list): 
    distinction_list = []
    for stimulus in stimulus_list:
        if stimulus in ('pumpkin', 'boa', 'seesaw', 'ferriswheel', 'kazoo', 'hulahoop', 'ribbon',
                         'television', 'crayon', 'piano', 'beachball', 'speaker', 'joystick', 'beer', 'boat'):
             distinction_list.append(0)  
        
        else:
            distinction_list.append(1)
    return distinction_list

def get_value_transportation_chatGPT(stimulus_list): 
    distinction_list = []
    for stimulus in stimulus_list:
        if stimulus in ('bulldozer', 'bobsled', 'bike', 'boat', 'helicopter', 'hovercraft'):
             distinction_list.append(0)  
        
        else:
            distinction_list.append(1)
    return distinction_list

def get_value_food_chatGPT(stimulus_list): 
    distinction_list = []
    for stimulus in stimulus_list:
        if stimulus in ('pumpkin', 'jam', 'bean', 'cheese', 'pear', 'banana', 'donut', 'lasagna', 'cow', 'cookie', 'peach', 'lemonade', 
                        'fudge', 'marshmallow', 'brownie', 'grape', 'tamale', 'guacamole'):
             distinction_list.append(0)  
        
        else:
            distinction_list.append(1)
    return distinction_list

def get_value_metal_pictures(stimulus_list): 
    distinction_list = []
    for stimulus in stimulus_list:
       
        if stimulus in ('bulldozer', 'cufflink', 'grate', 'watch', 'key', 'microscope', 'television', 'bike', 'headlamp', 
                        'candelabra',  'piano', 'clipboard', 'speaker', 'crank', 'brace', 'spoon', 'axe', 'ashtray', 'typewriter', 'horseshoe',
                          'pan', 'joystick', 'boat', 'bobsled', 'earring', 'hovercraft', 'seesaw', 'bench'):
             distinction_list.append(0)  
        
        else:
            distinction_list.append(1)
    return distinction_list

def get_value_plastic_pictures(stimulus_list): 
    distinction_list = []
    for stimulus in stimulus_list:
        
        if stimulus in ('beachball', 'blind', 'boa', 'brace', 'crank', 'earring', 'headlamp', 'hovercraft', 'hulahoop',
                         'joystick', 'kazoo', 'microscope', 'pacifier', 'piano', 'ribbon', 'pacifier', 'shredder', 'simcard', 
                         'television', 'typewriter',  'wig'):
             distinction_list.append(0)  
        
        else:
            distinction_list.append(1)
    return distinction_list

def get_value_natural_material_pictures(stimulus_list): 
    distinction_list = []
    for stimulus in stimulus_list:
        
        if stimulus in ('pumpkin', 'bamboo', 'beaver', 'nest', 'bean', 'pear', 'banana', 'rabbit', 'cow', 'starfish', 
                        'mango', 'ribbon', 'monkey', 'wasp', 'grape', 'chipmunk', 'alligator', 'boa', 'iguana', 'butterfly', 'hippopotamus', 
                        'altar', 'bed', 'bench', 'clipboard', 'drawer', 'easel', 'guacamole', 'speaker', 'stalagmite'):
             distinction_list.append(0)  
        
        else:
            distinction_list.append(1)
    return distinction_list

def get_value_kitchen_chatGPT(stimulus_list): 
    distinction_list = []
    for stimulus in stimulus_list:
        
        if stimulus in ('pumpkin', 'donut', 'jam', 'bean', 'cheese', 'pear', 'banana', 'lasagna', 'cow', 'cookie', 'peach', 'lemonade', 
                        'fudge', 'marshmallow', 'brownie', 'grape', 'tamale', 'guacamole', 'jar', 'spoon', 'pan'):
             distinction_list.append(0)  
        else:
            distinction_list.append(1)
    return distinction_list

def get_value_livingroom_chatGPT(stimulus_list): 
    distinction_list = []
    for stimulus in stimulus_list:
        
        if stimulus in ('key', 'drawer','television', 'jar','piano', 'lemonade','speaker', 'candelabra', 'wallpaper'):
             distinction_list.append(0)  
        
        else:
            distinction_list.append(1)
    return distinction_list

def get_value_bedroom_chatGPT(stimulus_list): 
    distinction_list = []
    for stimulus in stimulus_list:
        
        if stimulus in ('cufflink', 'watch', 'key', 'drawer', 'television', 't-shirt', 'uniform', 
                         'dress', 'earring', 'bed', 'mosquitonet', 'wallpaper', 'speaker'):
             distinction_list.append(0)  
        
        else:
            distinction_list.append(1)
    return distinction_list

def get_value_electricity_chatGPT(stimulus_list): 
    distinction_list = []
    for stimulus in stimulus_list:
        
        if stimulus in ('shredder', 'television', 'headlamp', 'speaker', 'joystick', 'helicopter', 'ferriswheel', 'streetlight', 'hovercraft'):
             distinction_list.append(0)  
        
        else:
            distinction_list.append(1)
    return distinction_list

def get_value_habitat(stimulus_list):
    distinction_list = []
    for stimulus in stimulus_list:
        
        if stimulus in ('beaver', 'starfish', 'alligator'):
             distinction_list.append(0)  
        elif stimulus in ('cow', 'chipmunk', 'hippopotamus', 'horse', 'monkey', 'rabbit', 'butterfly', 'iguana', 'wasp', 'dragonfly'):
            distinction_list.append(1)
        else: 
            distinction_list.append(-1)
    return distinction_list

def get_value_carnivore_chatGPT(stimulus_list):
    distinction_list = []
    for stimulus in stimulus_list:
        
        if stimulus in ( 'alligator', 'dragonfly', 'starfish', 'wasp'):
            distinction_list.append(0)  
        elif stimulus in ('cow', 'beaver', 'chipmunk', 'hippopotamus', 'horse', 'monkey', 'rabbit', 'butterfly', 'iguana'):
            distinction_list.append(1)
        else:
            distinction_list.append(-1)
    return distinction_list

def get_value_herbivore_chatGPT(stimulus_list):
    distinction_list = []
    for stimulus in stimulus_list:
        
        if stimulus in ('cow', 'beaver', 'chipmunk', 'hippopotamus', 'horse', 'rabbit', 'butterfly', 'iguana'):
            distinction_list.append(0)  
        elif stimulus in ('alligator', 'monkey', 'starfish', 'wasp', 'dragonfly'):
            distinction_list.append(1)
        else:
            distinction_list.append(-1)
    return distinction_list

def get_value_omnivore_chatGPT(stimulus_list):
    distinction_list = []
    for stimulus in stimulus_list:
        
        if stimulus in ('monkey'):
            distinction_list.append(0)  
        elif stimulus in ('cow', 'beaver', 'chipmunk', 'hippopotamus', 'horse', 'rabbit', 'butterfly', 'iguana',
                           'alligator', 'dragonfly', 'starfish', 'wasp'):
            distinction_list.append(1)
        else:
            distinction_list.append(-1)
    return distinction_list

def get_value_mammal(stimulus_list):
    distinction_list = []
    for stimulus in stimulus_list:
        
        if stimulus in ('monkey','cow', 'beaver', 'chipmunk', 'hippopotamus', 'horse', 'rabbit' ):
            distinction_list.append(0)  
        elif stimulus in ('butterfly', 'iguana', 'alligator', 'dragonfly', 'starfish', 'wasp'):
            distinction_list.append(1)
        else:
            distinction_list.append(-1)
    return distinction_list

def get_value_mammal_all(stimulus_list):
    distinction_list = []
    for stimulus in stimulus_list:
        
        if stimulus in ('monkey','cow', 'beaver', 'chipmunk', 'hippopotamus', 'horse', 'rabbit' ):
            distinction_list.append(0)  
        else:
            distinction_list.append(1)
    return distinction_list

def get_value_reptile(stimulus_list):
    distinction_list = []
    for stimulus in stimulus_list:
        
        if stimulus in ('alligator','iguana'):
            distinction_list.append(0)  
        elif stimulus in ('monkey','cow', 'beaver', 'chipmunk', 'hippopotamus', 'horse', 'rabbit' , 'butterfly', 'dragonfly', 'starfish', 'wasp'):
            distinction_list.append(1)
        else:
            distinction_list.append(-1)
    return distinction_list

def get_value_reptile_all(stimulus_list):
    distinction_list = []
    for stimulus in stimulus_list:
        
        if stimulus in ('alligator','iguana'):
            distinction_list.append(0)  
        else:
            distinction_list.append(1)
    return distinction_list

def get_value_invertebrate(stimulus_list):
    distinction_list = []
    for stimulus in stimulus_list:
        
        if stimulus in ('butterfly', 'dragonfly', 'starfish', 'wasp'):
            distinction_list.append(0)  
        elif stimulus in ('monkey','cow', 'beaver', 'chipmunk', 'hippopotamus', 'horse', 'rabbit' , 'alligator','iguana'):
            distinction_list.append(1)
        else:
            distinction_list.append(-1)
    return distinction_list
def get_value_invertebrate_all(stimulus_list):
    distinction_list = []
    for stimulus in stimulus_list:
        
        if stimulus in ('butterfly', 'dragonfly', 'starfish', 'wasp'):
            distinction_list.append(0)  
        else:
            distinction_list.append(1)
    return distinction_list

# adapted from rsatoolbox bootstrap pattern to work for model fit bootstrapping
def bootstrap_nnls(models, data, theta=None, method='cosine', N=1000,
                           pattern_descriptor='index', rdm_descriptor='index',
                           boot_noise_ceil=False):
    """evaluates a models on data
    performs bootstrapping over patterns to get a sampling distribution

    Args:
        models(rsatoolbox.model.Model or list): models to be evaluated
        data(rsatoolbox.rdm.RDMs): data to evaluate on
        theta(numpy.ndarray): parameter vector for the models
        method(string): comparison method to use
        N(int): number of samples
        pattern_descriptor(string): descriptor to group patterns for bootstrap
        rdm_descriptor(string): descriptor to group patterns for noise
            ceiling calculation

    Returns:
        numpy.ndarray: vector of evaluations

    """
    models, evaluations, theta, _ = \
        rsatoolbox.util.inference_util.input_check_model(models, theta, None, N)
    noise_min = []
    noise_max = []
    for i in tqdm.trange(N):
        sample, pattern_idx = \
            rsatoolbox.inference. bootstrap_sample_pattern(data, pattern_descriptor)
        df = pd.DataFrame(columns= [mod.name for mod in models])
        if len(np.unique(pattern_idx)) >= 3:
           
            for j, mod in enumerate(models):
                
                rdm_pred = mod.predict_rdm(theta=theta[j])
                rdm_pred = rdm_pred.subsample_pattern(pattern_descriptor,
                                                      pattern_idx)
                
                vectors, y, non_nan_mask = rsatoolbox.util.rdm_utils._parse_nan_vectors(rdm_pred.get_vectors(), sample.get_vectors())
                
                if (len(np.unique(vectors[0])) != 1):
                    df[mod.name] = stats.zscore(vectors[0])
                else: 
                    df[mod.name] = vectors[0]
                
            evaluations[i]= scipy.optimize.nnls(df, y[0])[0]
            if boot_noise_ceil:
                noise_min_sample, noise_max_sample = rsatoolbox.inference.noise_ceiling.boot_noise_ceiling(
                    sample, method=method, rdm_descriptor=rdm_descriptor)
                noise_min.append(noise_min_sample)
                noise_max.append(noise_max_sample)
        else:
            evaluations[i, :] = np.nan
            noise_min.append(np.nan)
            noise_max.append(np.nan)
    if boot_noise_ceil:
        eval_ok = np.isfinite(evaluations[:, 0])
        noise_ceil = np.array([noise_min, noise_max])
        variances = np.cov(np.concatenate([evaluations[eval_ok, :].T,
                                           noise_ceil[:, eval_ok]]))
    else:
        eval_ok = np.isfinite(evaluations[:, 0])
        noise_ceil = np.array(rsatoolbox.inference.noise_ceiling.boot_noise_ceiling(
            data, method=method, rdm_descriptor=rdm_descriptor))
        variances = np.cov(evaluations[eval_ok, :].T)
    dof = data.n_cond - 1
    result = rsatoolbox.inference.Result(models, evaluations, method=method,
                    cv_method='bootstrap_pattern_nnls', noise_ceiling=noise_ceil,
                    variances=variances, dof=dof, n_rdm=None,
                    n_pattern=data.n_cond)
    result.n_rdm = data.n_rdm
    return result