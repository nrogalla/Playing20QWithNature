from scipy.spatial.distance import correlation
from absl import logging
from sklearn.cluster import KMeans

def extract_question(text):
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

def extract_questions(text): 
  all_questions_with_duplicates = []

  logging.set_verbosity(logging.ERROR)
  for sub in text:
    #compute embeddings for ChatGpt questions
    if "ChatGPT" in sub and "?" in sub:
      question = extract_question(sub)
      if question is not None:
        all_questions_with_duplicates.append(question)
  return all_questions_with_duplicates

def extract_questions_by_position(position, text):

  pos_count = None
  all_questions_with_duplicates_position = []
  logging.set_verbosity(logging.ERROR)
  for sub in text:
    if "Object:" in sub: 
      pos_count = 1
    #compute embeddings for ChatGpt questions
    if "ChatGPT" in sub and "?" in sub:
      if pos_count in position:
        question = extract_question(sub)
        if question is not None:
          all_questions_with_duplicates_position.append(question)
        else: 
          pos_count -=1 
      pos_count += 1
  return all_questions_with_duplicates_position

def cluster(n_clusters, data):
    kmeans = KMeans(n_clusters=n_clusters, random_state = 1)
    kmeans.fit(data)
    Z = kmeans.predict(data)
    return kmeans, Z
