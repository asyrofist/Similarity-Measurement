from function import re, math, string, ff, st, alt
from function import word_tokenize, train_test_split
from function import classification_report, accuracy_score, precision_score, recall_score
from function import neighbors, tree, svm, GaussianNB, RandomForestClassifier, CountVectorizer, PCA
from function import preprocessing, fulldataset, apply_cleaning_function_to_list, pd, np, sent_PCA
from function import similarity_cosine, similarity_levenshtein, similarity_jaccard, tfidf, hasil_tfidf, TfidfVectorizer
from function import l2_normalizer, build_lexicon, freq, numDocsContaining, idf, build_idf_matrix, pmi_measurement, pmi_jumlah, co_occurrence
from function import KMeans, adjusted_rand_score, TruncatedSVD, TfidfVectorizer
from function import spatial, Pool, Word2Vec, distance, TaggedDocument, Doc2Vec, cosine_similarity

from sklearn.preprocessing import LabelEncoder

st.write("""
# Requirement Dependency Measurements
Berikut ini algoritma yang digunakan untuk pengukuran kebergantungan antar kebutuhan
""")

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
       tabel_jumlahpmi = pd.DataFrame(a4, index= id_requirement, columns= id_requirement)
       st.dataframe(tabel_jumlahpmi)
       
       #fitur pmi
       st.subheader("Feature PMI Parameter")
       desc_pmi = tabel_jumlahpmi.describe()
       st.dataframe(desc_pmi)
        
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
       
       #fitur vsm
       st.subheader("Feature VSM Parameter")
       desc_vsm = df_vsm.describe()
       st.dataframe(desc_vsm)
    
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
       cos = pd.DataFrame(hasil_cosine, index=id_requirement, columns=id_requirement)
       st.dataframe(cos)
       
       # Visualisasi
       st.subheader('Feature Parameters')
       st.dataframe(cos.describe())
       fig = ff.create_distplot(hasil_cosine, id_requirement)
       st.plotly_chart(fig, use_container_width=True)
       
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
       model = Doc2Vec.load('doc2vec_model')
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
    
       # Visualisasi
       st.subheader('Feature parameters')
       st.dataframe(df_kmeans.describe())
       st.line_chart(df_kmeans.describe())
    
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
            st.write(desc_cos)
            
      # levenshtein
      elif hasil == 'levenshtein':
            st.subheader('Similarity levenshtein parameters')
            hasil_levenshtein = []
            for angka in range(0, len(cleaned_text)):
                b = [similarity_levenshtein(cleaned_text[angka], cleaned_text[num]) for num in range(0, len(cleaned_text))]
                hasil_levenshtein.append(b)
            id_requirement = fulldataset(index0, index1)['ID']
            df_lev = pd.DataFrame(hasil_levenshtein, index= id_requirement, columns= id_requirement)
            hasil = hasil_levenshtein
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
            st.write(desc_lev)
            
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
            st.write(desc_jaccard)

      # tfidf
      elif hasil == 'tfidf':
            st.subheader('Similarity tfidf parameters')
            vect = TfidfVectorizer()
            tfidf_matrix = vect.fit_transform(cleaned_text)
            id_requirement = fulldataset(index0, index1)['ID']
            df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), index=id_requirement,  columns = vect.get_feature_names())
            hasil = tfidf_matrix.toarray()
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
            st.write(desc_tfidf)

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
            st.write(desc_vsm)

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
            model = Doc2Vec.load('doc2vec_model')
            nilai_vektor = [model.infer_vector("sent{}".format(num)) for num in range(0, len(cleaned_text))]
            id_requirement = fulldataset(index0, index1)['ID']
            df_vektor = pd.DataFrame(nilai_vektor, index=id_requirement, columns= ['vektor {}'.format(num) for num in range(0, size_value)])
            hasil = nilai_vektor
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
            st.write(desc_vektor)

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
            hasil = hasil_sentencemodel
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
            st.write(desc_sentmodel)
      
      # variable training testing
      kalimat = fulldataset(index0, index1)['kalimat']
      le_Y = LabelEncoder()
      label_kalimat = le_Y.fit_transform(kalimat)
#       label_statement = fulldataset(index0, index1)['label']
      size            = st.sidebar.slider('test_size', 0.1, 0.6, 0.3)
      
      # classification
      st.sidebar.header('Classification Parameters')
      SVM = st.sidebar.button('Support Vector Machine')
      RFC = st.sidebar.button('Random Forest Classifier')
      KNN = st.sidebar.button('K Nearest Neighbor')
      GNB = st.sidebar.button('Gaussian Naive Bias')
      DT  = st.sidebar.button('Decission Tree')
      
      X_train, X_test, y_train, y_test = train_test_split(hasil, label_kalimat, test_size=size,random_state=109) # 70% training and 30% test
      st.subheader('User Train Test parameters')
      traintest = pd.DataFrame([y_train, y_test], index=['TRAIN', 'TEST'])
      st.write(traintest)      
        
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
          chart_data = pd.DataFrame([akurasi, presisi, rekal], index=['akurasi', 'presisi', 'rekal'])
          st.bar_chart(chart_data)
