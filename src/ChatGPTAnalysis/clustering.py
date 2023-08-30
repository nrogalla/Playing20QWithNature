from scipy.spatial.distance import correlation
from absl import logging
from sklearn.cluster import KMeans
import re
import numpy as np
import src.ChatGPTAnalysis.extendedDendrogram as extendedDendrogram
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman' 
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import seaborn as sns
import sklearn

sns.set_theme() 
sns.set_style("white")


def extract_question(text):
      """
      Extracts question from a line of text while excluding questions that are not part of the game '20 questions'

      Args:
        text (string) : line 

      Returns:
        string : line reduced to relevant question, or none if no question was determined
      """
      line = text[9:]
      line = line.split(": ")[-1]
      line = line.split("! ")[-1]
      line = line.split (". ")[-1]
      if all(quest not in line for quest in ("correct", "would you","let me know", "tell me", "10", "reveal", "What is", "Is it one of these two", "What was the object")): 
        return line
      else:
        return None

def compute_similarity(a, b):
  return correlation(a,b)

def embed(model, input):
  return model.encode(input)

def extract_questions(text): 
  """
      Extracts questions from a transcript of the game '20 Questions'

      Args:
        text (text) : list of lines in the transcript 

      Returns:
        string : list of questions asked during the game
  """
  all_questions_with_duplicates = []

  logging.set_verbosity(logging.ERROR)
  for sub in text:
    if "ChatGPT" in sub and "?" in sub:
      question = extract_question(sub)
      if question is not None:
        all_questions_with_duplicates.append(question)
  return all_questions_with_duplicates

def extract_questions_by_position(position, text):
  """
      Extracts questions by position in a transcript of the game '20 Questions', 
      e.g. position = 1 returns all questions asked as the first question in the game

      Args:
        position (list of ints): (list of) positions ot extract questions for
        text (text) : list of lines in the transcript 

      Returns:
        string : list of questions at position asked during the game
  """
  pos_count = None
  all_questions_with_duplicates_position = []
  logging.set_verbosity(logging.ERROR)
  for sub in text:
    # reset position count for every beginning of a game
    if "Object:" in sub: 
      pos_count = 1
    if "ChatGPT" in sub and "?" in sub:
      if pos_count in position:
        question = extract_question(sub)
        if question is not None:
          all_questions_with_duplicates_position.append(question)
        else: 
          # question did not belong to the main part of the game, therefore position has to be adjusted
          pos_count -=1 
      pos_count += 1
  return all_questions_with_duplicates_position

def cluster(n_clusters, data):
    """
      performs KMeans clustering

      Args:
        n_clusters (int): number of clusters to be used for Kmeans
        data (list): list of question embeddings
      Returns: 
        KMeans: KMeans object
        numpy.ndarray: int labels of clusters
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state = 1, max_iter = 5000)
    kmeans.fit(data)
    Z = kmeans.predict(data)
    return kmeans, Z

def get_mean_cluster_size(R):
    """
      determies the mean cluster size in a dendrogram

      Args:
        R (dictionary): dictionary as returned by scipy.cluster.hierarchy.dendrogram
      Returns: 
        int: mean cluster size rounded to the next integer
    """
    counts = []
    for r in R['ivl']: 
        dig = re.findall(r'\d+', r)
        if dig == []:
            counts.append(1)
        else:
            counts.append(int(dig[0]))
    return round(np.mean(counts))


def get_truncated_dendrogram(Z, truncate_labels, figsize, labels, truncate_mode = None, p = 100, truncate_threshold = 0, color_threshold =0.7, min_clus_limit = False):
  '''
    plots a truncated fully labeled dendrogram

    Args:
      Z (scipy.cluster.hierarchy.linkage): hierarchical linkage to be displayed
      truncate_labels (list): list of labels for truncated dendrogram
      figsize (float, float): (width, height) of the plot
      truncate_mode (string): None, 'lastp', 'level' or 'threshold', for further information see extendedDendrogram
      truncate_threshold (float): only for truncate_mode 'threshold', else None
      color_threshold (float): threshold below which dendrogram will be colored distincively
      labels (list): list of labels of untruncated dendrogram
      p (int): for 'threshold' truncation: max cluster size, for 'lastp' number of non-singleton clusters, for 'level', number of levels
      min_clus_limit (bool): specifies if cluster of sizes below mean should be excluded (only available for 'threshold' truncation
    
  '''
  plt.figure(figsize=figsize)
  if truncate_mode != None:
    R = extendedDendrogram.dendrogram(Z, leaf_font_size = 14, color_threshold= color_threshold, truncate_mode = truncate_mode,truncate_threshold=truncate_threshold, labels = labels, no_plot=True, p=p) 

    # create a label dictionary
    leaves_with_label = R['leaves'][:len(truncate_labels)]
    temp = {leaves_with_label[ii]: truncate_labels[ii] for ii in range(len(leaves_with_label))}


    def llf(xx, count):
        if xx in leaves_with_label:
          return "{}".format(temp[xx]) + " (" + count + ")"
        #in case not enough labels have been supplied
        else:
          return "other"
          
    if min_clus_limit == True:
      min_clus_size = get_mean_cluster_size(R)
    else:
      min_clus_size = 0
    extendedDendrogram.dendrogram(Z, orientation = 'left', above_threshold_color = 'black', leaf_font_size = 14, leaf_label_func=llf, truncate_mode = truncate_mode,  color_threshold = color_threshold, truncate_threshold=truncate_threshold, labels = labels, p = p, min_clus_size = min_clus_size)
  else:
    extendedDendrogram.dendrogram(Z, orientation = 'left', above_threshold_color = 'black', leaf_font_size = 14, truncate_mode = truncate_mode,  color_threshold = color_threshold,  labels = labels)
  plt.show()

def visualise_KMeans(c, labels, questions, question_embeddings, d_r_method):
  '''
    visualises KMeans by performing dimensionality reduction

    Args:
      c (numpy.ndarray): int label of cluster per question
      labels (list): labels for clusters, lenght of the list must correspond to the number of clusters
      questions (list): list of questions corresponding to embeddings
      question_embeddings (numpy.ndarray): array of all embeddings corresponding to the questions
      d_r_method (string): method used for dimensionality reduction, implemented are MDS and tSNE

  '''
  if d_r_method == 'MDS':
    reduced_emb= sklearn.manifold.MDS(max_iter = 100000, random_state = 1).fit_transform(question_embeddings)
  elif d_r_method == 'tSNE':
    reduced_emb= sklearn.manifold.TSNE(n_iter = 100000, random_state = 1).fit_transform(question_embeddings)
  else: 
    raise ValueError('dimensionality reduction method not implemented')
  df_embeddings = pd.DataFrame(reduced_emb)
  df_embeddings = df_embeddings.assign(text = questions)
  df_embeddings = df_embeddings.assign(Cluster_number= c)

  sizes = []
  for i in range(len(np.unique(df_embeddings['Cluster_number']))):
      sizes.append(len(list(df_embeddings[df_embeddings['Cluster_number'] == i]['Cluster_number'])))
  df_clus_size = pd.DataFrame(columns = ['Cluster', 'Size'])
  df_clus_size['Cluster'] = ['Outdoors / Nature', "Functional Object / Tool", "Animal Classification", "Physical Characteristics and Object Features", "Entertainment and Recreational Object Usage", "Habitat and Geographic Distribution", "Kitchen and Household Item", "Plant and Botanical Characteristics",
  "Indoor Object Location", 'Material Composition', "Physical Interaction", "Aquatic Environment", "Electronic Device Characteristics", "Natural vs Man-Made Distinction", "Animal Classification and Traits", "Domesticated and Pet Animals", 
  "Animal Classification and Characteristics", "Size Comparison", "Transportation", "Living Organism"]
  df_clus_size['Size'] = sizes

  labels_with_size = []
  for i in range(len(labels)):
      labels_with_size.append(labels[i] + ' ('+ str(int(df_clus_size[df_clus_size['Cluster'] == labels[i]]['Size'])) + ')')
  set_labels = list(set(labels_with_size))

  colors = ["yellowgreen","red","darkgray","burlywood", "darkseagreen", "firebrick", "darkturquoise",  "orchid",  "pink",  "olive",  "darkviolet", "salmon",  "orange", 
          "gold",   "darkblue", "black", "mediumvioletred", "steelblue",   "rosybrown",   "hotpink"]

  dcm = {set_labels[i]:colors[i] for i in range(0, len(set_labels))}
  df_embeddings = df_embeddings.assign(Cluster= labels_with_size)

  fig = px.scatter(df_embeddings, x = 0, y = 1, color='Cluster', color_discrete_map = dcm, width = 1000, height = 700,labels={'color': 'Cluster'},  hover_data = ['text'] )
  fig.update_layout(
      font_family="Times New Roman",
      title_font_family="Times New Roman",
      font_size = 16
  )
  fig.update_xaxes(visible=False)
  fig.update_yaxes(visible=False)

  fig.show()


def get_labels_kMeans(c):
  '''
    provides labels for kMeans
    Args:
      c (numpy.ndarray): int label of cluster per question

    Returns:
      list: list of labels of KMeans clusters
  '''
  labels = []
  for i in c:
    if i == 0:
      labels.append('Outdoors / Nature')
    elif i == 1:
      labels.append("Functional Object / Tool")
    elif i == 2:
      labels.append("Animal Classification")
    elif i == 3:
      labels.append("Physical Characteristics and Object Features")
    elif i == 4:
      labels.append("Entertainment and Recreational Object Usage")
    elif i == 5:
      labels.append("Habitat and Geographic Distribution")
    elif i == 6:
      labels.append("Kitchen and Household Item")
    elif i == 7:
      labels.append("Plant and Botanical Characteristics")
    elif i == 8:
      labels.append("Indoor Object Location")
    elif i == 9:
      labels.append('Material Composition')
    elif i == 10:
      labels.append("Physical Interaction")
    elif i == 11:
      labels.append("Aquatic Environment")
    elif i == 12:
      labels.append("Electronic Device Characteristics")
    elif i == 13:
      labels.append("Natural vs Man-Made Distinction")
    elif i == 14:
      labels.append("Animal Classification and Traits")
    elif i == 15:
      labels.append("Domesticated and Pet Animals")
    elif i == 16:
      labels.append("Animal Classification and Characteristics")
    elif i == 17:
      labels.append("Size Comparison")
    elif i == 18:
      labels.append("Transportation")
    elif i == 19:
      labels.append("Living Organism")
    else: 
      labels.append("other")
  return labels

def plot_Elbow_chart(question_embeddings):
  '''
    plots heuristic Elbow chart for cluster number selection in KMeans

    Args:
      question_embeddings(numpy.array): array of question embeddings 
  '''
  wcss = []

  for k in range(1, 40):
      kmeans = KMeans(n_clusters=k, n_init = 'auto', random_state=42)
      kmeans.fit(question_embeddings)
      wcss.append(kmeans.inertia_)

  wcss = pd.DataFrame(wcss, columns=['Value'])
  wcss.index += 1

  sns.set_theme() 
  sns.set_style("whitegrid")
  ax = sns.lineplot(wcss)

  ax.set_ylabel('WCCS')
  ax.set_xlabel('Number of Clusters')

  ax.get_legend().set_visible(False)