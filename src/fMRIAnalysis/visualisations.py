import numpy as np
import pandas as pd
import rsatoolbox
import rsatoolbox.rdm as rsr
import os
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import scipy.spatial.distance as ssd
import sklearn
import warnings
import src.fMRIAnalysis.data as data



def get_icons():
    """returns icons

    Returns:
        dictionary: dictionary of icons with {'name', 'image'}, with image being the icon and name being the stimulus name
    """
    directory = r"..\data\betas_csv_testset\images"
    icons = defaultdict(list)

    # iterate over files in directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            # resizing images to the same size
            (width, height) = (100, 100)
            im_resized = Image.open(f).resize((width, height))
    
            split_filename = str(filename).rsplit('_')
            if len(split_filename) == 2:
                icons['name'].append(split_filename[0])
            else: 
                icons['name'].append(split_filename[0]+split_filename[1])
            icons['image'].append(rsatoolbox.vis.Icon(im_resized))
    return icons

def get_icons_color(category_name):
    """returns markers

    Args: 
        category_name (str): name of the category distinction 

    Returns:
        dictionary: dictionary of icons with {'name', 'image'}, with image being the marker colored by function and name being the stimulus name
    """
    directory = r"..\data\betas_csv_testset\images"
    icons = defaultdict(list)
    # iterate over files directory
    colors = plt.get_cmap('Accent').colors
    markers = list(matplotlib.markers.MarkerStyle('').markers.keys())
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            
            split_filename = str(filename).rsplit('_')
            name = ''
            if len(split_filename) == 2:
                name = split_filename[0]
                icons['name'].append(name)
            else: 
                name = split_filename[0]+split_filename[1]
                icons['name'].append(name)
            
            icons['image'].append(rsatoolbox.vis.Icon(marker=".",marker_front = False,
                                       color=colors[data.get_values_distinction([name], category_name)[0]]))

    return icons

def sort_rdm(rdm, category_name_list):
    """returns rdms sorted according to a category distinction
    Args: 
        rdm (rsatoolbox.RDMs): rdm to sort
        category_name_list: name of the category distinction 

    Returns:
        rsatoolbox.RDM: sorted according to sort_function
    """
    conds_df = pd.DataFrame(columns = ['stimulus'])
    conds_df['stimulus'] = rdm.pattern_descriptors['conds']
    conds_df = conds_df.assign(sort = lambda df: data.get_values_distinction(df['stimulus'].tolist(), category_name_list))
    #exclude data that does not strictly fall into the categories
    conds_df = conds_df[conds_df['sort'] != -1]
    #sort values according to category distinction
    sorted_conds_df = conds_df.sort_values(['sort', 'stimulus'], ascending = True).drop('sort', axis='columns')
    sorted_ind = sorted_conds_df.index
    rdm.reorder(sorted_ind)
    return rdm

def label_rdm(fig, cat_1, cat_2, ideal_rdm, size = 10, condition_labels = False): 
    """adds distinction labels to figure of rdms sorted according to a category distinction
    Args: 
        fig (Figure): figure to annotate
        cat_1 (string): name of the left side of the category distinction
        cat_2 (string): na of the right side of the category distinction
        size (int): size of the font
        condition_labels (bool): if True, annotation will be moved to make space for labels

    Returns:
        rsatoolbox.RDM: sorted according to sort_function
    """
    length = 0
    for i in range(len(ideal_rdm)):
        if(ideal_rdm[i][0] == 0):
            length+=1
    length = length#/len(ideal_rdm)
    cond_adjust = 0
    if condition_labels == True: 
        cond_adjust = - 15
        for i in range(1, 16): 
            fig.axes.text(x= -i, y = length-0.5, s = 'l', rotation = 'vertical', ha = 'center',  size = size, va = 'center_baseline')
    # only print both labels on upper left corner if they won't overlap
    if (length -5 > 3):
        fig.axes.text(x= cond_adjust-1, y = length - 5, s = cat_1, rotation = 'vertical', ha = 'right', size = size)
    fig.axes.text(x= cond_adjust-1, y = length-0.5, s = 'l', rotation = 'vertical', ha = 'center',  size = size, va = 'center_baseline')
    fig.axes.text(x= cond_adjust-1, y = length + 5, s = cat_2, rotation = 'vertical', ha = 'right', va = 'top',  size = size)
    
    fig.axes.text(x= length - 5 , y = -0.7 , s = cat_1, va = 'bottom', ha = 'right',  size = size)
    fig.axes.text(x= length -0.5, y = -0.5, s = 'l', size = size, ha = 'center')
    fig.axes.text(x= length + 5 , y = -0.7 , s = cat_2, va = 'bottom', ha = 'left',  size = size)


# adapted from rsatoolbox.vis.model_map.map_model_comparison to only show intermodal RDM
def plot_intermodel_distance_matrix(models, rdms_data, metric, category_name_list):
    '''
    plots RDM with distances between models

    Args: 
        models(rsatoolbox.model.Model or list): models 
        rdms_data(rsatoolbox.rdm.RDMs): data
        metric: distance metric for comparisons
        category_name_list: list of distinctions
    
    '''
    n_models = len(models)
    n_dissim = int(models[0].n_cond * (models[0].n_cond - 1) / 2)
    modelRDMs = np.empty((n_models, n_dissim))
    for idx, model_i in enumerate(models):
        if rdms_data is not None:
            theta = model_i.fit(rdms_data)
            modelRDMs[idx, :] = model_i.predict(theta)
        else:
            modelRDMs[idx, :] = model_i.predict()

    modelRDMs = modelRDMs - modelRDMs.mean(axis=1, keepdims=True)
    modelRDMs /= np.sqrt(np.einsum('ij,ij->i', modelRDMs, modelRDMs))[:, None]
    
    intermodelDists = ssd.squareform(
        ssd.pdist(modelRDMs, metric=metric))
    rdm_dists = np.zeros((n_models, n_models))
 
    rdm_dists[:, :] = intermodelDists

    plt.imshow(rdm_dists, cmap='Greys')
    plt.colorbar()
    plt.xticks(ticks = np.arange(0,14, 1),labels = category_name_list, rotation = 90)
    plt.yticks(ticks = np.arange(0,14, 1),labels = category_name_list)
    plt.show()



def plot_mds_images(data_rdm, icon_size):
    '''
    plots MDS with images

    Args:
        data_rdm (rsatoolbox.rdm.RDMs): data rdm
        icon_size (float > 0): scaling of the icons size
    '''
    icon_rdm = rsatoolbox.rdm.RDMs(
                dissimilarities=data_rdm.dissimilarities,
                dissimilarity_measure=data_rdm.dissimilarity_measure,
                rdm_descriptors=data_rdm.rdm_descriptors,
                pattern_descriptors=get_icons()
            )
    fig = rsatoolbox.vis.show_MDS(
        icon_rdm,
        pattern_descriptor='image',
        icon_size = icon_size
        )
    plt.box(on=None)
    # neccessary because axis limits in previous function cuts of outermost valies
    ax = plt.gca()  
    ax.autoscale() 
    fig.set_figheight(15)
    fig.set_figwidth(15)


def plot_mds_distinctions(data_rdm, category_name_list):
    '''
    plots MDS with color indicating distinction

    Args:
        data_rdm (rsatoolbox.rdm.RDMs): data rdm
        category_name_list (list): list of distinctions
    '''
    warnings.filterwarnings("ignore")
    size = 3
    plot_shape = (int(np.sqrt(len(category_name_list)))+1, int(np.sqrt(len(category_name_list)))+1)
    plt.figure(figsize=(plot_shape[0]*size, size* plot_shape[1]))
    axes = []
    mds_em = sklearn.manifold.MDS(n_components=2,
                random_state=1,#seed,
                dissimilarity='precomputed')
    for i in range(len(category_name_list)): 
        rdm = rsatoolbox.rdm.RDMs(
                    dissimilarities=data_rdm.dissimilarities,
                    dissimilarity_measure="correlation",
                    pattern_descriptors= get_icons_color(category_name_list[i])
                )
        coord = mds_em.fit_transform(rdm.get_matrices()[0])
        
        group = data.get_values_distinction(rdm.pattern_descriptors['name'], category_name_list[i])

        labels = ['yes', 'no']
        
        ax = plt.subplot(plot_shape[0], plot_shape[1],i+1)
        axes.append(ax)
        ax.set_title(category_name_list[i])
        for g in np.unique(group):
            i = np.where(group == g)
            ax.scatter(coord[:,0][i], coord[:,1][i],
                c = plt.get_cmap('Paired').colors[g*4], label = labels[g])
        plt.box(on=None)
        plt.legend()

    for ax in axes: 
        ax.tick_params(axis='both', which='both', bottom=False, top=False,
                    right=False, left=False, labelbottom=False, labeltop=False,
                    labelleft=False, labelright=False)

    plt.show()


def get_images_along_distinction(data_rdm, distinction_name):
    '''
    plots images along a distinction, seperating into 'yes' and 'no' categories

    Args: 
        data_rdm(rsatoolbox.rdm.RDMs): rdm 
        distinction_name (string): name of the distinction to be plotted
    '''
    icons = get_icons()
    category = data.get_values_distinction(icons['name'], distinction_name) 
    y_spacing = 8
    temp_x_left = 3
    temp_y_left = 0
    temp_x_right = 24
    temp_y_right = 0
    coords = np.ones((1,100,2))
    for i in range(len(icons['name'])): 
        if category[i] == 0:
            coords[0, i, 0] = temp_x_left
            coords[0,i,1] = temp_y_left
            if temp_x_left < 21:
                temp_x_left += 2
            else: 
                temp_x_left = 3
                temp_y_left += y_spacing
        else: 
            coords[0, i, 0] = temp_x_right
            coords[0, i, 1] = temp_y_right
            if temp_x_right < 42:
                temp_x_right += 2
            else: 
                temp_x_right = 24
                temp_y_right += y_spacing
    
    icon_rdm = rsatoolbox.rdm.RDMs(
                dissimilarities=data_rdm.dissimilarities,
                dissimilarity_measure=data_rdm.dissimilarity_measure,
                rdm_descriptors=data_rdm.rdm_descriptors,
                pattern_descriptors=get_icons()
            )
    fig = rsatoolbox.vis.show_scatter(
        icon_rdm,
        coords = coords,
        pattern_descriptor = 'image',
        icon_size=0.4
    )
    fig.set_figheight(5)
    fig.set_figwidth(20)
    plt.box(on=None)
    plt.title('   ' + distinction_name, fontsize = 15)
    plt.text( 10.6,44,'Yes', fontsize = 12)
    plt.text( 31.6,44,'No',fontsize = 12)
    plt.plot([22.5,22.5], [-20,40], color = 'black')

def get_stimuli_images(data_rdm):
    """plots all stimuli images

    Args:
        data_rdm(rsatoolbox.rdm.RDMs): RDM 

    """
    coords = np.ones((1,100,2))
    start = 0
    stop = 10
    for i in range(10):
        coords[0][start:stop,0] = np.arange(0,10,1) *10
        coords[0][start:stop,1] = np.zeros((1,10))+ (i*10)
        start = stop
        stop += 10
    
    icon_rdm = rsatoolbox.rdm.RDMs(
                dissimilarities=data_rdm.dissimilarities,
                dissimilarity_measure=data_rdm.dissimilarity_measure,
                rdm_descriptors=data_rdm.rdm_descriptors,
                pattern_descriptors=get_icons()
            )
    fig = rsatoolbox.vis.show_scatter(
            icon_rdm,
            coords = coords,
            pattern_descriptor = 'image',
            icon_size=0.55
        )
    fig.set_figheight(10)
    fig.set_figwidth(10)
    plt.box(on=None)


def plot_all_model_rdms(data_rdm, category_name_list):
    '''
    plots data rdm sorted according to the given distinctions

    Args: 
        data_rdm(rsatoolbox.rdm.RDMs): data rdm
        category_name_list (list): list of categories to plot rdms for
    '''
    sorted_rdms = []
    for i in range(len(category_name_list)):
        sorted_rdms.append(rsatoolbox.rdm.rank_transform(sort_rdm(data_rdm, category_name_list[i])))
    rdms = rsatoolbox.rdm.concat(sorted_rdms)
    rdms.rdm_descriptors = {'label': category_name_list}

    fig, ax, ret_val = rsatoolbox.vis.show_rdm(rdms, cmap = 'viridis', figsize = (15,15), rdm_descriptor = 'label', show_colorbar='figure')
    i = 0
    
    square_len = np.ceil(np.sqrt(len(rdms)))
    row = 0
    col = 0
    for rdm in sorted_rdms:
        categorical_rdm = rsr.get_categorical_rdm(np.array(data.get_values_distinction(rdm.pattern_descriptors['conds'], category_name_list[i])))
        label_rdm(ax[row][col], 'yes', 'no', categorical_rdm.get_matrices()[0], 7)
        
        i += 1
        if col < square_len-1:
            col += 1
        else:
            col = 0
            row += 1