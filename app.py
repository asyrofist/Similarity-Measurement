# import
from function import nltk, re, math, string, st, alt, train_test_split
from function import classification_report, accuracy_score, precision_score, recall_score
from function import neighbors, tree, svm, GaussianNB, RandomForestClassifier, CountVectorizer, PCA
from function import preprocessing, fulldataset, apply_cleaning_function_to_list, pd, np, sent_PCA
from function import similarity_cosine, similarity_levenshtein, similarity_jaccard, tfidf, hasil_tfidf, TfidfVectorizer
from function import l2_normalizer, build_lexicon, freq, numDocsContaining, idf, build_idf_matrix, pmi_measurement, pmi_jumlah, co_occurrence
from function import KMeans, adjusted_rand_score, TruncatedSVD, TfidfVectorizer, word_tokenize 
from function import spatial, Pool, Word2Vec, distance, TaggedDocument, Doc2Vec, cosine_similarity
from function import LabelEncoder, plt, sns, confusion_matrix, ProfileReport, st_profile_report

from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def traintestANN(x, y, rasio,  input1, input2, input3, 
                 aktivasi1, aktivasi2, aktivasi3, 
                 layer1, layer2, layer3, learning_rate, 
                 verbose_value, batch_value, epoch_value):
    # Split the data for training and testing
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=rasio)

    # Build the model
    model = Sequential()
    model.add(Dense(input1, input_shape=(4,), activation= aktivasi1, name= layer1))
    model.add(Dense(input2, activation= aktivasi2, name= layer2))
    model.add(Dense(input3, activation= aktivasi3, name= layer3))

    # Adam optimizer with learning rate of 0.001
    optimizer = Adam(lr= learning_rate)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Train the model
    model.fit(train_x, train_y, verbose= verbose_value, batch_size= batch_value, epochs= epoch_value)  
    results = model.evaluate(test_x, test_y)

st.write("""
# Requirement Dependency Measurements
Berikut ini algoritma yang digunakan untuk pengukuran kebergantungan antar kebutuhan
""")
st.set_option('deprecation.showPyplotGlobalUse', False)

#file upload
index0 = st.file_uploader("Choose a file") 
if index0 is not None:
    st.sidebar.header('Dataset Parameter')
    x1 = pd.ExcelFile(index0)
    index1 = st.sidebar.selectbox( 'What Dataset you choose?', x1.sheet_names)
    st.subheader('Dataset parameters')
    st.write(fulldataset(index0, index1))
   
    # Nilai Pembanding
    st.sidebar.subheader('Measurement Parameter')
    similaritas     = st.sidebar.checkbox("Similarity & Classification")
    ontology        = st.sidebar.checkbox("Ontology Construction")
    extraction      = st.sidebar.checkbox("Requirement Extraction")
    occurance       = st.sidebar.checkbox("Term Co-Occurance")
        
    #co-occurance 
    if occurance:
       st.header("First Co-occurance")
       text_to_clean = list(fulldataset(index0, index1)['Requirement Statement'])
       cleaned_text = apply_cleaning_function_to_list(text_to_clean)
               
       #pmi measurement
       st.subheader("PMI Measurement Parameter")
       id_requirement = fulldataset(index0, index1)['ID']
       a2 = []
       for angka in range(0, len(cleaned_text)):
          a1 = [pmi_measurement(cleaned_text[angka], cleaned_text[num]) for num in range(0, len(cleaned_text))]
          a2.append(a1)
       tabel_pmi = pd.DataFrame(a2, index= id_requirement, columns= id_requirement)
       st.dataframe(tabel_pmi)
        
       #pmi jumlah 
       st.subheader("PMI Sum Parameter")
       a4 = []
       for angka in range(0, len(cleaned_text)):
          a3 = [pmi_jumlah(cleaned_text[angka], cleaned_text[num]) for num in range(0, len(cleaned_text))]
          a4.append(a3)
       df_jumlahpmi = pd.DataFrame(a4, index= id_requirement, columns= id_requirement)
       st.dataframe(df_jumlahpmi)
       
       st.subheader("Document Profiling")
       profile1 = st.checkbox("Profile PMI parameter")

       # feature collection
       st.subheader('Feature  parameters')
       desc_pmi = df_jumlahpmi.describe()
       opsi_pmi = st.multiselect('What Feature PMI do you remove?',['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],['count'])
       desc_pmi = desc_pmi.drop(opsi_pmi, axis=0)
       desc_pmi = desc_pmi.T
       st.write(desc_pmi)
                
       # second order
       st.header("Second Co-occurance")
       text_to_clean = list(fulldataset(index0, index1)['Requirement Statement'])
       cleaned_text = apply_cleaning_function_to_list(text_to_clean)
       
       st.subheader('VSM parameters')
       vect = TfidfVectorizer()
       tfidf_matrix = vect.fit_transform(cleaned_text)
       matrix_tfidf = tfidf_matrix.toarray()
       vsm = cosine_similarity(matrix_tfidf[0:], matrix_tfidf)
       id_requirement = fulldataset(index0, index1)['ID']
       df_vsm = pd.DataFrame(vsm, index=id_requirement,  columns = id_requirement)
       st.dataframe(df_vsm)
    
       st.subheader("Document Profiling")
       profile2 = st.checkbox("Profile VSM parameter")
       
       # feature collection
       st.subheader('Feature  parameters')
       desc_vsm = df_vsm.describe()
       opsi_vsm = st.multiselect('What Feature VSM do you remove?',['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],['count', 'min', '25%', 'max'])
       desc_vsm = desc_vsm.drop(opsi_vsm, axis=0)
       desc_vsm = desc_vsm.T
       st.write(desc_vsm)
                        
       # third order
       st.header('Thrd Co-occurance')
       st.sidebar.subheader("Model Parameter SVD")
       feature_value = st.sidebar.slider('Berapa Max Feature Model?', 0, 10, 1000)
       iterasi_value = st.sidebar.slider('Berapa Dimension Model?', 0, 200, 100)
       random_value = st.sidebar.slider('Berapa Random Model?', 0, 300, 122)

       # SVD represent documents and terms in vectors 
       st.subheader('SVD parameters')        
       vectorizer = TfidfVectorizer(stop_words='english', max_features= feature_value, max_df = 0.5, smooth_idf= True)
       X = vectorizer.fit_transform(cleaned_text)
       fitur_id = vectorizer.get_feature_names()
       svd_model = TruncatedSVD(n_components= (X.shape[0]), algorithm='randomized', n_iter=iterasi_value, random_state=random_value)
       svd_model.fit(X)
       jumlah_kata = svd_model.components_
       df_svd = pd.DataFrame(jumlah_kata, index= id_requirement, columns= fitur_id)
       st.dataframe(df_svd)
        
       st.subheader("Document Profiling")
       profile3 = st.checkbox("Profile SVD parameter")


       # feature collection
       st.subheader('Feature  parameters')
       desc_svd = df_svd.describe()
       opsi_svd = st.multiselect('What Feature SVD do you remove?',['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],['count'])
       desc_svd = desc_svd.drop(opsi_svd, axis=0)
       desc_svd = desc_svd.T
       st.write(desc_svd)
    
       # Document Profile
       if profile1:
          pr = ProfileReport(df_jumlahpmi, explorative=True)
          st_profile_report(pr)
       elif profile2:
          pr = ProfileReport(df_vsm, explorative=True)
          st_profile_report(pr)
       elif profile3:
          pr = ProfileReport(df_svd, explorative=True)
          st_profile_report(pr)
       
    # Requirement Extraction
    elif extraction:
       text_to_clean = list(fulldataset(index0, index1)['Requirement Statement'])
       cleaned_text = apply_cleaning_function_to_list(text_to_clean)
       id_requirement = fulldataset(index0, index1)['ID']
 
       # LSA
       st.sidebar.subheader("Model Parameter")
       feature_value = st.sidebar.slider('Berapa Max Feature Model?', 0, 10, 1000)
       iterasi_value = st.sidebar.slider('Berapa Dimension Model?', 0, 200, 100)
       random_value = st.sidebar.slider('Berapa Random Model?', 0, 300, 122)

       # SVD represent documents and terms in vectors 
       st.subheader('LSA parameters')        
       vectorizer = TfidfVectorizer(stop_words='english', max_features= feature_value, max_df = 0.5, smooth_idf= True)
       X = vectorizer.fit_transform(cleaned_text)
       fitur_id = vectorizer.get_feature_names()
       svd_model = TruncatedSVD(n_components= (X.shape[0]), algorithm='randomized', n_iter=iterasi_value, random_state=random_value)
       svd_model.fit(X)
       jumlah_kata = svd_model.components_
       tabel_lsa = pd.DataFrame(jumlah_kata, index= id_requirement, columns= fitur_id)
       st.dataframe(tabel_lsa)
   
       # kMeans
       st.subheader('KMeans parameters')
       true_k = (X.shape[0])
       # true_k = (X.shape[1]-1)
       model = KMeans(n_clusters=true_k, init='k-means++', max_iter=iterasi_value, n_init=1)
       model.fit(jumlah_kata)
       order_centroids = model.cluster_centers_.argsort()[:, ::-1]
       terms = vectorizer.get_feature_names()
       tabel_kmeans = pd.DataFrame(order_centroids, index= id_requirement, columns= terms)
       st.dataframe(tabel_kmeans)
        
       # cosine
       st.subheader('Cosine parameters') 
       hasil_cosine = cosine_similarity(order_centroids[0:], order_centroids)
       id_term = [("term {}".format(num)) for num in range(0, (X.shape[1]-1))]
       df_cos = pd.DataFrame(hasil_cosine, index=id_requirement, columns=id_requirement)
       st.dataframe(df_cos)
        
       # feature collection
       st.subheader('Feature  parameters')
       options = st.multiselect('What Feature do you remove?',['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],['count', 'max'])
       desc_cos = df_cos.describe()
       desc_cos = desc_cos.drop(options, axis=0)
       desc_cos = desc_cos.T
        
       # visusalisasi
       fig, ax = plt.subplots()
       sns.heatmap(desc_cos, annot=True, ax=ax)
       st.pyplot() 
        
       # Document Profile
       st.subheader("Document Profiling")
       profile = st.checkbox("Profile parameter")
       if profile:
          pr = ProfileReport(df_cos, explorative=True)
          st_profile_report(pr)
        
    # Ontology Construction
    elif ontology:
       text_to_clean = list(fulldataset(index0, index1)['Requirement Statement'])
       cleaned_text = apply_cleaning_function_to_list(text_to_clean)
       
       # document bag of words
       count_vector = CountVectorizer(cleaned_text)
       count_vector.fit(cleaned_text)
       doc_array = count_vector.transform(cleaned_text).toarray()            
       doc_feature = count_vector.get_feature_names()
       st.subheader('BOW parameters')
       id_requirement = fulldataset(index0, index1)['ID']
       bow_matrix = pd.DataFrame(doc_array, index= id_requirement, columns= doc_feature)
       st.dataframe(bow_matrix)
                
       # tfidf          
       doc_term_matrix_l2 = []
       # document l2 normalizaer
       for vec in doc_array:
           doc_term_matrix_l2.append(l2_normalizer(vec))

       # vocabulary & idf matrix 
       vocabulary = build_lexicon(cleaned_text)
       mydoclist = cleaned_text
       my_idf_vector = [idf(word, mydoclist) for word in vocabulary]
       my_idf_matrix = build_idf_matrix(my_idf_vector)

       doc_term_matrix_tfidf = []
       #performing tf-idf matrix multiplication
       for tf_vector in doc_array:
           doc_term_matrix_tfidf.append(np.dot(tf_vector, my_idf_matrix))
        
       doc_term_matrix_tfidf_l2 = []
       #normalizing
       for tf_vector in doc_term_matrix_tfidf:
            doc_term_matrix_tfidf_l2.append(l2_normalizer(tf_vector))
            
       hasil_tfidf = np.matrix(doc_term_matrix_tfidf_l2)
       st.subheader('TFIDF parameters')
       tfidf_matrix = pd.DataFrame(hasil_tfidf, index= id_requirement, columns= doc_feature)
       st.dataframe(tfidf_matrix)
                
       #doc2vec
       st.subheader('doc2vec parameters')
       sentences = [word_tokenize(num) for num in cleaned_text]
       for i in range(len(sentences)):
            sentences[i] = TaggedDocument(words = sentences[i], tags = ['sent{}'.format(i)])    # converting each sentence into a TaggedDocument
       st.sidebar.subheader("Model Parameter")
       size_value = st.sidebar.slider('Berapa Size Model?', 0, 200, len(doc_feature))
       iterasi_value = st.sidebar.slider('Berapa Iterasi Model?', 0, 100, 10)
       window_value = st.sidebar.slider('Berapa Window Model?', 0, 10, 3)
       dimension_value = st.sidebar.slider('Berapa Dimension Model', 0, 10, 1)
        
       model = Doc2Vec(documents = sentences, dm = dimension_value, size = size_value, window = window_value, min_count = 1, iter = iterasi_value, workers = Pool()._processes)
       model.init_sims(replace = True)
#        model = Doc2Vec.load('doc2vec_model')
       nilai_vektor = [model.infer_vector("sent{}".format(num)) for num in range(0, len(cleaned_text))]
       id_requirement = fulldataset(index0, index1)['ID']
       df_vektor = pd.DataFrame(nilai_vektor, index=id_requirement, columns= ['vektor {}'.format(num) for num in range(0, size_value)])
       st.dataframe(df_vektor)
            
       # Kmeans
       st.subheader('Kmeans parameters')
       true_k = len(nilai_vektor)
       model = KMeans(n_clusters=true_k, init='k-means++', max_iter=iterasi_value, n_init=1)
       model.fit(nilai_vektor)
       order_centroids = model.cluster_centers_.argsort()[:, ::-1]
       id_requirement = fulldataset(index0, index1)['ID']
       df_kmeans = pd.DataFrame(order_centroids, index= id_requirement, columns= ['vektor {}'.format(num) for num in range(0, size_value)])
       st.dataframe(df_kmeans)
       
       # feature collection
       st.subheader('Feature  parameters')
       options = st.multiselect('What Feature do you remove?',['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],['count'])
       desc_kmeans = df_kmeans.describe()
       desc_kmeans = desc_kmeans.drop(options, axis=0)
       desc_kmeans = desc_kmeans.T
        
       # Visualisasi
       fig, ax = plt.subplots()
       sns.heatmap(desc_kmeans, annot=True, ax=ax)
       st.pyplot()
       
       # Document Profile
       st.subheader("Document Profiling")
       profile = st.checkbox("Profile parameter")
       if profile:
          pr = ProfileReport(df_kmeans, explorative=True)
          st_profile_report(pr)
       
    # similarity
    elif similaritas:
      text_to_clean = list(fulldataset(index0, index1)['Requirement Statement'])
      cleaned_text = apply_cleaning_function_to_list(text_to_clean)
                                
      # variable parameter
      st.sidebar.header('Training Parameters')
      hasil = st.sidebar.selectbox('What Similarity Measurement?', ['cosine', 'levenshtein', 'jaccard', 'tfidf', 'vsm', 'doc2vec', 'sentencemodel'])
        
      # cosine
      if hasil == 'cosine':
            st.subheader('Similarity cosine parameters')
            hasil_cosine = []
            for angka in range(0, len(cleaned_text)):
                a = [similarity_cosine(cleaned_text[angka], cleaned_text[num]) for num in range(0, len(cleaned_text))]
                hasil_cosine.append(a)
            id_requirement = fulldataset(index0, index1)['ID']
            df_cos = pd.DataFrame(hasil_cosine, index= id_requirement, columns= id_requirement)
            st.write(df_cos)
                       
            #feature description
            desc_cos = df_cos.describe()
            st.write(desc_cos)

            # feature collection
            st.subheader('Feature  parameters')
            options = st.multiselect('What Feature do you remove?',['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],['count', 'min', '25%', 'max'])
            desc_cos = desc_cos.drop(options, axis=0)
            desc_cos = desc_cos.T
            hasil = desc_cos
            fig, ax = plt.subplots()
            sns.heatmap(hasil, annot=True, ax=ax)
            st.pyplot()
            
            # Document Profile
            st.subheader("Document Profiling")
            profile = st.checkbox("Profile parameter")
            if profile:
              pr = ProfileReport(df_cos, explorative=True)
              pr.to_file("profile_report.html")
              st_profile_report(pr)

            
      # levenshtein
      elif hasil == 'levenshtein':
            st.subheader('Similarity levenshtein parameters')
            hasil_levenshtein = []
            for angka in range(0, len(cleaned_text)):
                b = [similarity_levenshtein(cleaned_text[angka], cleaned_text[num]) for num in range(0, len(cleaned_text))]
                hasil_levenshtein.append(b)
            id_requirement = fulldataset(index0, index1)['ID']
            df_lev = pd.DataFrame(hasil_levenshtein, index= id_requirement, columns= id_requirement)
            st.dataframe(df_lev)
                        
            #feature description
            desc_lev = df_lev.describe()
            st.write(desc_lev)

            # feature collection
            st.subheader('Feature  parameters')
            options = st.multiselect('What Feature do you remove?',['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],['count'])
            desc_lev = desc_lev.drop(options, axis=0)
            desc_lev = desc_lev.T
            hasil = desc_lev
            fig, ax = plt.subplots()
            sns.heatmap(hasil, annot=True, ax=ax)
            st.pyplot()

            # Document Profile
            st.subheader("Document Profiling")
            profile = st.checkbox("Profile parameter")
            if profile:
              pr = ProfileReport(df_lev, explorative=True)
              st_profile_report(pr)
                        
      # jaccard
      elif hasil == 'jaccard':
            st.subheader('Similarity jaccard parameters')
            hasil_jaccard = []
            for angka in range(0, len(cleaned_text)):
                b = [similarity_jaccard(cleaned_text[angka], cleaned_text[num]) for num in range(0, len(cleaned_text))]
                hasil_jaccard.append(b)
            id_requirement = fulldataset(index0, index1)['ID']
            df_jaccard = pd.DataFrame(hasil_jaccard, index= id_requirement, columns= id_requirement)
            hasil = hasil_jaccard
            st.dataframe(df_jaccard)
            
            #feature description
            desc_jaccard = df_jaccard.describe()
            st.write(desc_jaccard)

            # feature collection
            st.subheader('Feature  parameters')
            options = st.multiselect('What Feature do you remove?',['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],['count'])
            desc_jaccard = desc_jaccard.drop(options, axis=0)
            desc_jaccard = desc_jaccard.T
            hasil = desc_jaccard
            fig, ax = plt.subplots(figsize=(10,10))
            sns.heatmap(hasil, annot=True, ax=ax)
            st.pyplot()
            
            # Document Profile
            st.subheader("Document Profiling")
            profile = st.checkbox("Profile parameter")
            if profile:
              pr = ProfileReport(df_jaccard, explorative=True)
              st_profile_report(pr)

      # tfidf
      elif hasil == 'tfidf':
            st.subheader('Similarity tfidf parameters')
            vect = TfidfVectorizer()
            tfidf_matrix = vect.fit_transform(cleaned_text)
            id_requirement = fulldataset(index0, index1)['ID']
            df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), index=id_requirement,  columns = vect.get_feature_names())
            st.dataframe(df_tfidf)
                        
            #feature description
            desc_tfidf = df_tfidf.describe()
            st.write(desc_tfidf)

            # feature collection
            st.subheader('Feature  parameters')
            options = st.multiselect('What Feature do you remove?',['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],['count'])
            desc_tfidf = desc_tfidf.drop(options, axis=0)
            desc_tfidf = desc_tfidf.T
            hasil = desc_tfidf
            fig, ax = plt.subplots()
            sns.heatmap(hasil, annot=True, ax=ax)
            st.pyplot()
            
            # Document Profile
            st.title("Document Profiling")
            profile = st.checkbox("Profile parameter")
            if profile:
              pr = ProfileReport(df_tfidf, explorative=True)
              pr.to_file("profile_report.html")
              st_profile_report(pr)

            
      # vsm
      elif hasil == 'vsm':
            st.subheader('Similarity vsm parameters')
            vect = TfidfVectorizer()
            tfidf_matrix = vect.fit_transform(cleaned_text)
            matrix_tfidf = tfidf_matrix.toarray()
            vsm = cosine_similarity(matrix_tfidf[0:], matrix_tfidf)
            id_requirement = fulldataset(index0, index1)['ID']
            df_vsm = pd.DataFrame(vsm, index=id_requirement,  columns = id_requirement)
            hasil = vsm
            st.dataframe(df_vsm)
            
            #feature description
            desc_vsm = df_vsm.describe()
            st.write(desc_vsm)

            # feature collection
            st.subheader('Feature  parameters')
            options = st.multiselect('What Feature do you remove?',['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],['count'])
            desc_vsm = desc_vsm.drop(options, axis=0)
            desc_vsm = desc_vsm.T
            hasil = desc_vsm
            fig, ax = plt.subplots()
            sns.heatmap(hasil, annot=True, ax=ax)
            st.pyplot()
            
            # Document Profile
            st.subheader("Document Profiling")
            profile = st.checkbox("Profile parameter")
            if profile:
              pr = ProfileReport(df_vsm, explorative=True)
              st_profile_report(pr)


      # doc2vec
      elif hasil == 'doc2vec':
            st.subheader('Similarity doc2vec parameters')
            sentences = [word_tokenize(num) for num in cleaned_text]
            for i in range(len(sentences)):
                sentences[i] = TaggedDocument(words = sentences[i], tags = ['sent{}'.format(i)])    # converting each sentence into a TaggedDocument
            st.sidebar.subheader('Doc2Vec Parameter')
            vocabulary = build_lexicon(cleaned_text)
            dimension_value = st.sidebar.slider('Berapa Dimension Model', 0, 10, 1)
            size_value = st.sidebar.slider('Berapa Size Model?', 0, 200, len(vocabulary))
            window_value = st.sidebar.slider('Berapa Window Model?', 0, 10, 3)
            iterasi_value = st.sidebar.slider('Berapa Iterasi Model?', 0, 100, 10)
            
            model = Doc2Vec(documents = sentences, dm = dimension_value, size = size_value, window = window_value, min_count = 1, iter = iterasi_value, workers = Pool()._processes)
            model.init_sims(replace = True)
            nilai_vektor = [model.infer_vector("sent{}".format(num)) for num in range(0, len(cleaned_text))]
            id_requirement = fulldataset(index0, index1)['ID']
            df_vektor = pd.DataFrame(nilai_vektor, index=id_requirement, columns= ['vektor {}'.format(num) for num in range(0, size_value)])
            st.dataframe(df_vektor)
                        
            #feature description
            desc_vektor = df_vektor.describe()
            st.write(desc_vektor)

            # feature collection
            st.subheader('Feature  parameters')
            options = st.multiselect('What Feature do you remove?',['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],['count'])
            desc_vektor = desc_vektor.drop(options, axis=0)
            desc_vektor = desc_vektor.T
            hasil = desc_vektor
            fig, ax = plt.subplots()
            sns.heatmap(hasil, annot=True, ax=ax)
            st.pyplot()
            
            # Document Profile
            st.subheader("Document Profiling")
            profile = st.checkbox("Profile parameter")
            if profile:
              pr = ProfileReport(df_vektor, explorative=True)
              st_profile_report(pr)

      # sentencemodel
      elif hasil == 'sentencemodel':
            st.subheader('Similarity sentencemodel parameters')
            threshold = 5
            for i in range(len(cleaned_text)):
                if len(cleaned_text[i]) < 5:
                    cleaned_text[i] = None
            cleaned_text = [sentence for sentence in cleaned_text if sentence is not None]
            vocabulary = build_lexicon(cleaned_text)
            dimension_value = st.sidebar.slider('Berapa Dimension Model', 0, 10, 1)
            size_value = st.sidebar.slider('Berapa Size Model?', 0, 200, len(vocabulary))
            mode_value = st.sidebar.selectbox('What Mode?', [0, 1])
            window_value = st.sidebar.slider('Berapa Window Model?', 0, 10, 3)
            iterasi_value = st.sidebar.slider('Berapa Iterasi Model?', 0, 100, 10)
            
            model = Word2Vec(sentences = cleaned_text, size = size_value, sg = mode_value, window = window_value, min_count = 1, iter = iterasi_value, workers = Pool()._processes)
            model.init_sims(replace = True)
            for i in range(len(cleaned_text)):
                cleaned_text[i] = [model[word] for word in cleaned_text[i]]
            sent_vectorized = [sent_PCA(sentence) for sentence in cleaned_text]
            hasil_sentencemodel = []
            for angka in range(0, len(cleaned_text)):
              a = [distance.euclidean(sent_vectorized[angka], sent_vectorized[num]) for num in range(0, len(cleaned_text))]
              hasil_sentencemodel.append(a)
            id_requirement = fulldataset(index0, index1)['ID']
            df_sentmodel = pd.DataFrame(hasil_sentencemodel, index=id_requirement, columns=id_requirement)
            st.dataframe(df_sentmodel)
                        
            #feature description
            desc_sentmodel = df_sentmodel.describe()
            st.write(desc_sentmodel)

            # feature collection
            st.subheader('Feature  parameters')
            options = st.multiselect('What Feature do you remove?',['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],['count'])
            desc_sentmodel = desc_sentmodel.drop(options, axis=0)
            desc_sentmodel = desc_sentmodel.T
            hasil = desc_sentmodel
            fig, ax = plt.subplots(figsize=(10,10))
            sns.heatmap(hasil, annot=True, ax=ax)
            st.pyplot()
            
            # Document Profile
            st.subheader("Document Profiling")
            profile = st.checkbox("Profile parameter")
            if profile:
              pr = ProfileReport(df_sentmodel, explorative=True)
              st_profile_report(pr)

      
      st.subheader("Train Test Parameter")
      traintest = st.checkbox("Train/Test Data")
      if traintest:
          # variable training testing
          kalimat         = fulldataset(index0, index1)['kalimat']
          le_Y            = LabelEncoder()
          label_kalimat   = le_Y.fit_transform(kalimat)
          size            = st.sidebar.slider('test_size', 0.1, 0.6, 0.3)

          X_train, X_test, y_train, y_test = train_test_split(hasil, label_kalimat, test_size=size,random_state=109) # 70% training and 30% test
          st.subheader('User Train Test parameters')
          traintest = pd.DataFrame([y_train, y_test], index=['TRAIN', 'TEST'])
          st.write(traintest)      

          # classification
          st.sidebar.header('Classification Parameters')
          SVM = st.sidebar.button('Support Vector Machine')
          RFC = st.sidebar.button('Random Forest Classifier')
          KNN = st.sidebar.button('K Nearest Neighbor')
          GNB = st.sidebar.button('Gaussian Naive Bias')
          DT  = st.sidebar.button('Decission Tree')
          ANN  = st.sidebar.checkbox('Artificial Neural Network')
          Profile  = st.checkbox('Document Profilling')

          # support vector machine
          if SVM:
              supportvectormachine = svm.SVC(decision_function_shape='ovo')
              supportvectormachine.fit(X_train, y_train)
              y_pred = supportvectormachine.predict(X_test)
              st.subheader('Classification based on SVM')
              st.write(classification_report(y_test, y_pred))
              st.subheader("cetak Prediksi setiap classifier")
              akurasi = accuracy_score(y_test, y_pred) 
              presisi = precision_score(y_test, y_pred, average='macro') 
              rekal = recall_score(y_test, y_pred, average='macro') 
              results = confusion_matrix(y_test, y_pred)
              fig, ax = plt.subplots()
              sns.heatmap(results, annot=True, ax=ax)
              st.pyplot()
              chart_data = pd.DataFrame([akurasi, presisi, rekal], index=['akurasi', 'presisi', 'rekal'])
              st.bar_chart(chart_data)

          # random forest classifier
          elif RFC:
              RFC = RandomForestClassifier()
              RFC.fit(X_train, y_train)
              y_pred = RFC.predict(X_test)
              st.subheader('Classification based on RFC')
              st.write(classification_report(y_test, y_pred))
              st.subheader("cetak Prediksi setiap classifier")
              akurasi = accuracy_score(y_test, y_pred) 
              presisi = precision_score(y_test, y_pred, average='macro') 
              rekal = recall_score(y_test, y_pred, average='macro') 
              results = confusion_matrix(y_test, y_pred)
              fig, ax = plt.subplots()
              sns.heatmap(results, annot=True, ax=ax)
              st.pyplot()
              chart_data = pd.DataFrame([akurasi, presisi, rekal], index=['akurasi', 'presisi', 'rekal'])
              st.bar_chart(chart_data)

          # K-Nearset Neighbor
          elif KNN:
              kNN = neighbors.KNeighborsClassifier(n_neighbors = 10, weights='distance')
              kNN.fit(X_train, y_train)
              y_pred = kNN.predict(X_test)
              st.subheader('Classification based on KNN')
              st.write(classification_report(y_test, y_pred))
              st.subheader("cetak Prediksi setiap classifier")
              akurasi = accuracy_score(y_test, y_pred) 
              presisi = precision_score(y_test, y_pred, average='macro') 
              rekal = recall_score(y_test, y_pred, average='macro') 
              results = confusion_matrix(y_test, y_pred)
              fig, ax = plt.subplots(figsize=(10,10))
              sns.heatmap(results, annot=True, ax=ax)
              st.pyplot()
              chart_data = pd.DataFrame([akurasi, presisi, rekal], index=['akurasi', 'presisi', 'rekal'])
              st.bar_chart(chart_data)

          # Gaussian Naive Bias
          elif GNB:
              GNB = GaussianNB()
              GNB.fit(X_train, y_train) 
              y_pred = GNB.predict(X_test)
              st.subheader('Classification based on GNB')
              st.write(classification_report(y_test, y_pred))
              st.subheader("cetak Prediksi setiap classifier")
              akurasi = accuracy_score(y_test, y_pred) 
              presisi = precision_score(y_test, y_pred, average='macro') 
              rekal = recall_score(y_test, y_pred, average='macro') 
              results = confusion_matrix(y_test, y_pred)
              fig, ax = plt.subplots()
              sns.heatmap(results, annot=True, ax=ax)
              st.pyplot()
              chart_data = pd.DataFrame([akurasi, presisi, rekal], index=['akurasi', 'presisi', 'rekal'])
              st.bar_chart(chart_data)

          # Decission Tree
          elif DT:
              DT = tree.DecisionTreeClassifier()
              DT.fit(X_train, y_train)
              y_pred = DT.predict(X_test)
              st.subheader('Classification based on DT')
              st.write(classification_report(y_test, y_pred))
              st.subheader("cetak Prediksi setiap classifier")
              akurasi = accuracy_score(y_test, y_pred) 
              presisi = precision_score(y_test, y_pred, average='macro') 
              rekal = recall_score(y_test, y_pred, average='macro') 
              results = confusion_matrix(y_test, y_pred)
              # visual
              fig, ax = plt.subplots()
              sns.heatmap(results, annot=True, ax=ax)
              st.pyplot()
              chart_data = pd.DataFrame([akurasi, presisi, rekal], index=['akurasi', 'presisi', 'rekal'])
              st.bar_chart(chart_data)
                
          elif ANN:
            x = hasil
            y_ = label_kalimat.reshape(-1, 1) # Convert data to a single column

            # One Hot encode the class labels
            encoder = OneHotEncoder(sparse=False)
            y = encoder.fit_transform(y_)
            
            # Test on unseen data
            st.sidebar.header('ANN Parameters')            
            rasio = size
            input1 = st.sidebar.slider('input1?', 0, 100, 10)
            input2 = st.sidebar.slider('input2?', 0, 100, 10)
            input3 = st.sidebar.slider('input3?', 0, 10, 3)
            aktivasi1 = st.sidebar.selectbox('aktivasi1?', ['relu', 'softmax'])
            aktivasi2 = st.sidebar.selectbox('aktivasi2?', ['relu', 'softmax'])
            aktivasi3 = st.sidebar.selectbox('aktivasi3?', ['softmax', 'relu'])
            layer1 = st.sidebar.selectbox('layer1?', ['fc1', 'fc2', 'output'])
            layer2 = st.sidebar.selectbox('layer2?', ['fc2', 'fc1', 'output'])
            layer3 = st.sidebar.selectbox('layer3?', ['output', 'fc1', 'fc2'])
            learning_rate = 0.001
            verbose_value = st.sidebar.slider('verbose size?', 0, 5, 2)
            batch_value = st.sidebar.slider('batch size?', 0, 10, 5)
            epoch_value = st.sidebar.slider('epoch size?', 0, 1000, 200)

            traintestANN(x, y, rasio,  input1, input2, input3, 
                         aktivasi1, aktivasi2, aktivasi3, 
                         layer1, layer2, layer3, learning_rate, 
                         verbose_value, batch_value, epoch_value)
            results_ = confusion_matrix(text_x, test_y)
            # visual
            fig, ax = plt.subplots()
            sns.heatmap(results_, annot=True, ax=ax)
            st.pyplot()
            
          # Document Profile
          elif Profile:
              pr = ProfileReport(hasil, explorative=True)
              st.title("Document Profiling")
              st.write(hasil)
              st_profile_report(pr)        
