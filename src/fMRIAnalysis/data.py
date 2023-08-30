import numpy as np
import pandas as pd
import rsatoolbox
import rsatoolbox.data as rsd


# extracts stimulus from metadata 
def get_stimulus_data(meta_data_csv):
    """Extracts stimulus from Meta_data

    Args:
        meta_data_csv (csv): Metadata csv file with columns 'trial_type' and 'stimulus' (containing file names of stimuli)

    Returns:
        pd.DataFrame: DataFrame with stimulus column
    """
    stimdata = pd.read_csv(meta_data_csv)
    testdata= stimdata[stimdata['trial_type'] == 'test']

    # extracts stimulus name from file name
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
    """Extracts indices of highest activation voxels based on average activation

    Args:
        response_data_subj (ndarray): numpy array containing the response data of one subject
        number_of_voxels (int): number of highest activation voxels to extract

    Returns:
        ndarray: indices of highest activation voxels sorted numerically
    """
    
    average_activation_per_voxel = np.average(response_data_subj, axis = 0)
    if number_of_voxels >= len(average_activation_per_voxel):
        ind = np.arange(0,len(average_activation_per_voxel),1)
    else:
        ind = np.argpartition(average_activation_per_voxel, -number_of_voxels)[-number_of_voxels:]
    sorted_ind = ind[np.argsort(ind)]
    return sorted_ind

def select_voxels_noise_cutoff(noise_ceil_sub, cutoff):
    """Extracts indices of highest activation voxels based on noise ceiling

    Args:
        noise_ceil_sub (ndarray): numpy array containing the per-voxel noise ceiling data of one subject
        cutoff (float): noise ceiling cutoff

    Returns:
        indices of highest activation voxels sorted numerically
    """
    noise_ceil_sub = noise_ceil_sub.reshape(np.shape(noise_ceil_sub)[0],)
    ind = np.where(noise_ceil_sub>cutoff)[0]
    return ind

def get_dataset_subset(response_data, meta_data, number_of_voxels = 500, noise_ceil = None, cut_off = None): 
    """Builds rsatoolbox Dataset

    Args:
        response_data (list): list of numpy arrays containing the response data of all subjects, list has length of number of subjects, response data has length of number of voxels
        meta_data (list): list of DataFrames of Metadata per subject, list has length of number of subjects
        number_of_voxels (int): number of voxels to extract
        noise_ceiling (list): list of noise_ceilings per voxel per subject, list has length of number of subjects, if supplied, voxel selection will be done by noise ceiling
        cut_off (float): noise_ceiling cut_off value, only relevant and necessary if noise ceiling was supplied

    Returns:
        List: list of rsatoolbox Datasets per subject, list has length of number of subjects
    """
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

def get_values_distinction(stimulus_list, distinction_name):
    """provides list of 0 and 1 values corresponding to the animacy of the stimulus in stimulus_list

    Args: 
        stimulus_list (list): list of stimuli names

    Returns:
        list: list of 0 and 1 values corresponding to the category distinction
    """
    distinction_list = []
    if distinction_name in ['Habitat', 'Carnivore', 'Herbivore', 'Omnivore', 'Mammal', 'Reptile', 'Invertebrate']:
        for stimulus in stimulus_list:
            category_lists = get_category_lists_animals(distinction_name)
            if stimulus in category_lists[0]:
                distinction_list.append(0)  
            elif stimulus in category_lists[1]:
                distinction_list.append(1)  
            else:
                distinction_list.append(-1)
    else: 
        for stimulus in stimulus_list:
            
            if stimulus in get_category_list(distinction_name):
                distinction_list.append(0)  
            else:
                distinction_list.append(1)
    return distinction_list

def get_assignment_dataframe(stimuli_list, category_name_list):
    """provides dataframe with category assignments
    Args:
        stimuli_list (list) : list of stimuli
        category_name_list (list): list of categories to provide distinctions for

    Returns:
        numpy.ndarray: vector of evaluations
    """
    df_assignments = pd.DataFrame(columns = ['Distinction', 'Yes', 'No'])
    df_assignments['Distinction'] = category_name_list
    for j in range(len(category_name_list)):
        numeric_cat = get_values_distinction(stimuli_list, category_name_list[j])
        yes_list = []
        no_list = []
        for i in range(len(stimuli_list)): 

            if numeric_cat[i] == 0:
                yes_list.append(stimuli_list[i])
            else:
                no_list.append(stimuli_list[i])
    
        df_assignments['Yes'][j] = yes_list
        df_assignments['No'][j] = no_list
    pd.set_option('display.max_colwidth', None)
    return df_assignments  


def get_category_list(distinction_name):
    """provides list of 'yes' category of available distinctions
    Args:
        distinction_name (string): name of the distinction

    Returns:
        list: list of yes category objects
    """
    if distinction_name == 'Animacy':
        return ['cow', 'beaver', 'chipmunk', 'hippopotamus', 'horse',  'monkey', 
                    'rabbit', 'alligator', 'butterfly', 'iguana', 'starfish', 'wasp', 'dragonfly', 'chest1', 'bamboo']
    elif distinction_name == 'Indoors':
        return ['altar', 'ashtray', 'banana', 'beachball', 'bed', 'beer', 'blind', 'boa', 'bobsled', 'brace', 'brownie', 'candelabra', 'cheese', 'chest1', 
                        'clipboard', 'coatrack', 'cookie', 'crank', 'crayon', 'cufflink', 'donut', 'dough', 'drawer', 'dress', 'earring', 'fudge', 'guacamole', 'horseshoe',
                        'jar', 'joystick', 'kazoo', 'key', 'kimono', 'lasagna', 'lemonade', 'mango', 'microscope', 'mosquitonet', 'mousetrap', 'pacifier', 'pan', 
                        'piano', 'quill', 'ribbon', 'shredder', 'simcard', 'speaker', 'spoon', 'tamale', 'television', 't-shirt', 'typewriter', 'urinal', 
                        'wallpaper', 'watch', 'wig']
    elif distinction_name == 'Size':
        return ['banana', 'boa', 'butterfly', 'grape', 'key', 'spoon', 'wasp', 'ashtray', 'bean', 'beer', 'brownie', 'dress', 'cheese',
                    'chipmunk', 'clipboard', 'cookie', 'crayon', 'crank', 'cufflink', 'donut', 'dough', 'dragonfly', 'earring',
                    'footprint', 'fudge', 'guacamole', 'headlamp', 'horseshoe', 'jam', 'joystick', 'kazoo',
                    'lasagna', 'lemonade', 'mango','marshmallow', 'mosquitonet', 'mousetrap', 'pacifier', 'peach', 'pear', 'quill', 
                    'ribbon', 'simcard', 'speaker', 'starfish', 't-shirt', 'tamale', 'uniform', 'watch', 'whip', 'wig']
    elif distinction_name == 'Man-Made':
        return ['pumpkin', 'bamboo', 'beaver', 'nest', 'bean', 'cheese', 'pear', 'monkey', 'wasp', 'rabbit', 
                        'cow', 'starfish', 'horse', 'banana', 'mango', 'grape', 'chipmunk', 'alligator', 
                         'iguana', 'butterfly', 'hippopotamus', 'dragonfly', 'stalagamite', 
                         'peach', 'chest1']
    elif distinction_name == 'Entertainment':
        return ['pumpkin', 'boa', 'seesaw', 'ferriswheel', 'kazoo', 'hulahoop', 'ribbon',
                         'television', 'crayon', 'piano', 'beachball', 'speaker', 'joystick', 'beer', 'boat']
    elif distinction_name == 'Transportation':
        return ['bulldozer', 'bobsled', 'bike', 'boat', 'helicopter', 'hovercraft']
    elif distinction_name == 'Food':
        return ['pumpkin', 'jam', 'bean', 'cheese', 'pear', 'banana', 'donut', 'lasagna', 'cow', 'cookie', 'peach', 'lemonade', 
                        'fudge', 'marshmallow', 'brownie', 'grape', 'tamale', 'guacamole']
    elif distinction_name == 'Metal':
        return ['bulldozer', 'cufflink', 'grate', 'watch', 'key', 'microscope', 'television', 'bike', 'headlamp', 
                        'candelabra',  'piano', 'clipboard', 'speaker', 'crank', 'brace', 'spoon', 'axe', 'ashtray', 'typewriter', 'horseshoe',
                          'pan', 'joystick', 'boat', 'bobsled', 'earring', 'hovercraft', 'seesaw', 'bench']
    elif distinction_name == 'Plastic':
        return ['beachball', 'blind', 'boa', 'brace', 'crank', 'earring', 'headlamp', 'hovercraft', 'hulahoop',
                         'joystick', 'kazoo', 'microscope', 'pacifier', 'piano', 'ribbon', 'pacifier', 'shredder', 'simcard', 
                         'television', 'typewriter',  'wig']
    elif distinction_name == 'Natural Material':
        return ['pumpkin', 'bamboo', 'beaver', 'nest', 'bean', 'pear', 'banana', 'rabbit', 'cow', 'starfish', 
                        'mango', 'ribbon', 'monkey', 'wasp', 'grape', 'chipmunk', 'alligator', 'boa', 'iguana', 'butterfly', 'hippopotamus', 
                        'altar', 'bed', 'bench', 'clipboard', 'drawer', 'easel', 'guacamole', 'speaker', 'stalagmite']
    elif distinction_name == 'Kitchen':
        return ['pumpkin', 'donut', 'jam', 'bean', 'cheese', 'pear', 'banana', 'lasagna', 'cow', 'cookie', 'peach', 'lemonade', 
                        'fudge', 'marshmallow', 'brownie', 'grape', 'tamale', 'guacamole', 'jar', 'spoon', 'pan']
    elif distinction_name == 'Living room':
        return ['key', 'drawer','television', 'jar','piano', 'lemonade','speaker', 'candelabra', 'wallpaper']
    elif distinction_name == 'Bedroom':
        return ['cufflink', 'watch', 'key', 'drawer', 'television', 't-shirt', 'uniform', 
                         'dress', 'earring', 'bed', 'mosquitonet', 'wallpaper', 'speaker']
    elif distinction_name == 'Electricity':
        return ['shredder', 'television', 'headlamp', 'speaker', 'joystick', 'helicopter', 'ferriswheel', 'streetlight', 'hovercraft', 'watch']
    elif distinction_name == 'Subset Animals':
        return ['cow', 'beaver', 'chipmunk', 'hippopotamus', 'horse',  'monkey', 
                        'rabbit', 'alligator', 'butterfly', 'iguana', 'starfish', 'wasp', 'dragonfly']
    else: 
        raise ValueError('Distinction not available')
    
def get_category_lists_animals(distinction_name):
    """provides list of 'yes' category of available animal distinctions
    Args:
        distinction_name (string): name of the distinction

    Returns:
        list: list of yes category objects
    """
    if distinction_name == 'Habitat':
        return [['beaver', 'starfish', 'alligator'],['cow', 'chipmunk', 'hippopotamus', 'horse', 'monkey', 'rabbit', 'butterfly', 'iguana', 'wasp', 'dragonfly']]
    elif distinction_name == 'Carnivore':
        return [['alligator', 'dragonfly', 'starfish', 'wasp'], ['cow', 'beaver', 'chipmunk', 'hippopotamus', 'horse', 'monkey', 'rabbit', 'butterfly', 'iguana']]
    elif distinction_name == 'Herbivore':
        return [['cow', 'beaver', 'chipmunk', 'hippopotamus', 'horse', 'rabbit', 'butterfly', 'iguana'], ['alligator', 'monkey', 'starfish', 'wasp', 'dragonfly']]
    elif distinction_name == 'Omnivore':
        return [['monkey'], ['cow', 'beaver', 'chipmunk', 'hippopotamus', 'horse', 'rabbit', 'butterfly', 'iguana',
                        'alligator', 'dragonfly', 'starfish', 'wasp']]
    elif distinction_name == 'Mammal':
        return [['monkey','cow', 'beaver', 'chipmunk', 'hippopotamus', 'horse', 'rabbit'],['butterfly', 'iguana', 'alligator', 'dragonfly', 'starfish', 'wasp']]
    elif distinction_name == 'Reptile':
        return [['alligator','iguana'],['monkey','cow', 'beaver', 'chipmunk', 'hippopotamus', 'horse', 'rabbit' , 'butterfly', 'dragonfly', 'starfish', 'wasp']]
    elif distinction_name == 'Invertebrate':
        return [['butterfly', 'dragonfly', 'starfish', 'wasp'], ['monkey','cow', 'beaver', 'chipmunk', 'hippopotamus', 'horse', 'rabbit' , 'alligator','iguana']]


