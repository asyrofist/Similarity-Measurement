import nltk
import pandas as pd
import statistics
import numpy as np
import tensorflow.compat.v1 as tf
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tabulate import tabulate 
tf.disable_v2_behavior()

stemming = PorterStemmer()
stops = set(stopwords.words("english"))
lem = WordNetLemmatizer()

# fungsi ini untuk mengecek data preprocessing
def preprocessing(index0):
  x1 = pd.ExcelFile(index0)
  dfs = {sh:x1.parse(sh) for sh in x1.sheet_names}
  return dfs

# fungsi ini digunakan untuk mengecek data secara keseluruhan dataset tertentu
def fulldataset(index0, index1):
  x1 = pd.ExcelFile(index0)
  dfs = {sh:x1.parse(sh) for sh in x1.sheet_names}[index1]
  return dfs

# fungsi ini digunakan untuk melihat data excel yang spesifik
def dataset(index0, index1, index2, index3):
  x1 = pd.ExcelFile(index0)
  dfs = {sh:pd.x1.parse(sh) for sh in pd.x1.sheet_names}[index1][index2][index3]
  return dfs

# cleaning text
def apply_cleaning_function_to_list(X):
    cleaned_X = []
    for element in X:
        cleaned_X.append(clean_text(element))
    return cleaned_X

def clean_text(raw_text):
    """This function works on a raw text string, and:
        1) changes to lower case
        2) tokenizes (breaks down into words
        3) removes punctuation and non-word text
        4) finds word stems
        5) removes stop words
        6) rejoins meaningful stem words"""
    
    # Convert to lower case
    text = raw_text.lower()
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Keep only words (removes punctuation + numbers)
    # use .isalnum to keep also numbers
    token_words = [w for w in tokens if w.isalpha()]
    
    # # Stemming
    # stemmed_words = [stemming.stem(w) for w in token_words]

    # Lemmatization
    lemma_words = [lem.lemmatize(w) for w in token_words]
    
    # Remove stop words
    meaningful_words = [w for w in lemma_words if not w in stops]
    

    # Rejoin meaningful stemmed words
    joined_words = ( " ".join(meaningful_words))
    
    # Return cleaned data
    return joined_words

# lavenstein
def similarity_levenshtein(s1, s2):
    if len(s1) < len(s2):
        return similarity_levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

# Cosine Similarity 
def similarity_cosine(data1, data2):
    # tokenization 
    X_list = word_tokenize(data1) 
    Y_list = word_tokenize(data2) 

    # sw contains the list of stopwords 
    sw = stopwords.words('english') 
    l1 =[];l2 =[] 

    # remove stop words from string 
    X_set = {w for w in X_list if not w in sw} 
    Y_set = {w for w in Y_list if not w in sw} 

    # form a set containing keywords of both strings 
    rvector = X_set.union(Y_set) 
    for w in rvector: 
      if w in X_set: l1.append(1) # create a vector 
      else: l1.append(0) 
      if w in Y_set: l2.append(1) 
      else: l2.append(0) 
    c = 0

    # cosine formula 
    for i in range(len(rvector)): 
        c+= l1[i]*l2[i] 
    cosine = c / float((sum(l1)*sum(l2))**0.5) 
    return cosine 


# jaccard similarity
def similarity_jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


# term frequency - inverse document frequency
hasil_tfidf = []
def tfidf(data1, id_requirement):
  vect = TfidfVectorizer()
  tfidf_matrix = vect.fit_transform(data1)
  df = pd.DataFrame(tfidf_matrix.toarray(), index=id_requirement,  columns = vect.get_feature_names())
  hasil_tfidf.append(tfidf_matrix.toarray())
  print(tabulate(df, headers = 'keys', tablefmt = 'psql'))

# vsm similarity
hasil_vsm = []
def vsm_similarity(data, id_requirement):
  from sklearn.metrics.pairwise import cosine_similarity
  vect = TfidfVectorizer()
  tfidf_matrix = vect.fit_transform(data)
  matrix_tfidf = tfidf_matrix.toarray()
  vsm = cosine_similarity(matrix_tfidf[0:], matrix_tfidf)
  hasil_vsm.append(vsm)
  df = pd.DataFrame(vsm, index=id_requirement,  columns = id_requirement)
  print(tabulate(df, headers = 'keys', tablefmt = 'psql'))

# visualisasi
def visualisasi(data, id_requirement):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    pca = PCA(n_components=len(data))
    my_pca = pca.fit_transform(data)
    plt.scatter(my_pca[:,0], my_pca[:,1])

    for i, word in enumerate(id_requirement):
        plt.annotate(word, xy=(my_pca[i,0], my_pca[i,1]))
        plt.xlabel('widht')
        plt.ylabel('height')

    plt.title('Visualisasi')
    plt.show()

def hitungstat(data):
    maximum = max(data)
    minimum = min(data)
    mean = statistics.mean(data)
    harmonicmean = statistics.harmonic_mean(data) 
    pvariance = statistics.pvariance(data)
    variance = statistics.variance(data)
    median = statistics.median(data) 
    median_group = statistics.median_grouped(data)
    lowmedian = statistics.median_low(data)
    highmedian = statistics.median_high(data)
    standar_deviasi = statistics.stdev(data)
    # modus = statistics.mode(data)
    df = pd.DataFrame(data)
    dataBaru = maximum, minimum, mean, harmonicmean, pvariance, median, median_group, lowmedian, highmedian, standar_deviasi
    df = pd.DataFrame(dataBaru, index=['maximum', 'minimum', 'mean', 'harmonic mean', 'pvariance', 'median', 'median group', 'lowmedian', 'highmedian', 'standar deviasi'], columns=['variable value'])
    df.to_csv('hitungstat.csv', sep='\t')
    return df

def npstat(data):
  maximum = np.max(data)
  minimum = np.min(data)
  mean = np.mean(data)
  variance = np.var(data)
  median = np.median(data)
  standar_deviasi = np.std(data)
  dataBaru = maximum, minimum, mean, variance, median, standar_deviasi
  df = pd.DataFrame(dataBaru, index=['maximum', 'minimum', 'mean', 'variance', 'median', 'standar deviasi'], columns=['variable value'])
  df.to_csv('npstat.csv', sep='\t')
  return df

def jumlahWupalmer(data):
    clean = [x for x in data if x != None]
    jumlah = sum(clean)
    return jumlah

# gensim
kataset = []
kata = []
kata_split = []
def stringwords(text):
  # string, words
  string_sentences = ' '.join(text)
  kata_split.append(string_sentences)
  words = string_sentences.split(' ')
  kata.append(words)
  print("cleaned text\n", text)
  print("string sentence\n",string_sentences)
  # set words
  words1 = set(words)
  print ("set words\n", words)
  kataset.append(words1)

int2word = {}
word2int = {}
def wordint(text):
  vocab_size = len(text)
  print("panjang vocab\n",vocab_size)
  for i,word in enumerate(text):
    # word, int
    word2int[word] = i
    int2word[i] = word
    print("word2int\n", word2int)
    print("int2word\n",int2word)

sentences = []
def splitting(text):
  raw_sentences = text
  for sentence in raw_sentences:
    sentences.append(sentence.split())
  print("raw sentences\n", raw_sentences)
  print("splitting sentences\n",sentences)

def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

x_train = [] 
y_train = [] 
def traintest(text):
  for data_word in text:
      x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))
      y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))
  print("X Train", x_train)
  print("\n")
  print("y Train", y_train)

def arraytraintest(x, y):
  x_train = np.asarray(x)
  y_train = np.asarray(y)
  print("X Train", x_train)
  print("\n")
  print("y Train", y_train)

def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def find_closest(word_index, vectors):
    min_dist = 10000 # to act like positive infinity
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index

# word2vec
# regex function
def regextext(data):
  for i in range(len(sentences)):
    sentences[i] = [word.lower() for word in sentences[i] if re.match('^[a-zA-Z]+', word)]
    return sentences

# define a function that computes cosine similarity between two words
def cosine_similarity(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)

# glove
# define a function that converts word into embedded vector
def vector_converter(word):
    idx = glove.dictionary[word]
    return glove.word_vectors[idx]
# define a function that computes cosine similarity between two words
def cosine_similarity(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)

# sentence modeling
# define function to compute weighted vector representation of sentence
# parameter 'n' means number of words to be accounted when computing weighted average
def sent_PCA(sentence, n = 2):
    pca = PCA(n_components = n)
    pca.fit(np.array(sentence).transpose())
    variance = np.array(pca.explained_variance_ratio_)
    words = []
    for _ in range(n):
        idx = np.argmax(variance)
        words.append(np.amax(variance) * sentence[idx])
        variance[idx] = 0
    return np.sum(words, axis = 0)

# define a function that computes cosine similarity between two words
def cosine_similarity(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)

def label_cluster(data, label, kolom):
  print("hasli perbandingan label clustering{}\n".format(index1))
  a = pd.DataFrame(data, index=label, columns=kolom)
  print(tabulate(a, headers = 'keys', tablefmt = 'psql'))   
  
def prediksi_cluster(data, label, kolom):
  print("hasli perbandingan prediksi clustering{}\n".format(index1))
  a = pd.DataFrame(data, index=label, columns=kolom)
  print(tabulate(a, headers = 'keys', tablefmt = 'psql'))   