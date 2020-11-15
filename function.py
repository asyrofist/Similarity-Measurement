import nltk
import math
import pandas as pd
import statistics
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
import collections

# download nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

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
  return df

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
  return df

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
  return df

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

# variable tfidf
def l2_normalizer(vec):
    denom = np.sum([el**2 for el in vec])
    return [(el / math.sqrt(denom)) for el in vec]

def build_lexicon(corpus):
    lexicon = set()
    for doc in corpus:
        lexicon.update([word for word in doc.split()])
    return lexicon

def freq(term, document):
  return document.split().count(term)

def numDocsContaining(word, doclist):
    doccount = 0
    for doc in doclist:
        if freq(word, doc) > 0:
            doccount +=1
    return doccount 

def idf(word, doclist):
    n_samples = len(doclist)
    df = numDocsContaining(word, doclist)
    return np.log(n_samples / 1+df)

def build_idf_matrix(idf_vector):
    idf_mat = np.zeros((len(idf_vector), len(idf_vector)))
    np.fill_diagonal(idf_mat, idf_vector)
    return idf_mat  
  
  
def pmi_measurement(text1, text2):
    stopwords_ = set(stopwords.words('english'))
    words1 = [word.lower() for word in text1.split() if len(word) > 2 and word not in stopwords_]
    words2 = [word.lower() for word in text2.split() if len(word) > 2 and word not in stopwords_]

    finder = BigramCollocationFinder.from_words(words1+words2)
    bgm = BigramAssocMeasures()
    score = bgm.mi_like
    collocations = {'_'.join(bigram): pmi for bigram, pmi in finder.score_ngrams(score)}
    return collocations

def pmi_jumlah(text1, text2):
    stopwords_ = set(stopwords.words('english'))
    words1 = [word.lower() for word in text1.split() if len(word) > 2 and word not in stopwords_]
    words2 = [word.lower() for word in text2.split() if len(word) > 2 and word not in stopwords_]
    finder = BigramCollocationFinder.from_words(words1+words2)
    bgm = BigramAssocMeasures()
    score = bgm.mi_like
    total_pmi = sum([math.log(pmi) for bigram, pmi in finder.score_ngrams(score)])
    return total_pmi

def co_occurrence(sentences, window_size):
    d = collections.defaultdict(int)
    vocab = set()
    for text in sentences:
        # preprocessing (use tokenizer instead)
        text = text.lower().split()
        # iterate over sentences
        for i in range(len(text)):
            token = text[i]
            vocab.add(token)  # add to vocab
            next_token = text[i+1 : i+1+window_size]
            for t in next_token:
                key = tuple( sorted([t, token]) )
                d[key] += 1

    # formulate the dictionary into dataframe
    vocab = sorted(vocab) # sort vocab
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16), index=vocab, columns=vocab)
    for key, value in d.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    return df

