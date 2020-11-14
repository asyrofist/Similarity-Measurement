import re
import streamlit as st
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score 
from sklearn import neighbors, tree, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from function import preprocessing, fulldataset, apply_cleaning_function_to_list, pd, np, sent_PCA
from function import similarity_cosine, similarity_levenshtein, similarity_jaccard, tfidf, hasil_tfidf, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from multiprocessing import Pool
from scipy import spatial
from scipy.spatial import distance
from gensim.models import Word2Vec
from multiprocessing import Pool
from scipy import spatial


st.write("""
# Similarity & Classiifcation Measurements
Berikut ini algoritma yang digunakan untuk pengukuran similaritas dan klasifikasi
""")

#file upload
index0 = st.file_uploader("Choose a file") 
if index0 is not None:
    x1 = pd.ExcelFile(index0)
    index1 = st.sidebar.selectbox( 'What Dataset you choose?', x1.sheet_names)
    st.sidebar.write('You choose',index1)
    st.subheader('Dataset parameters')
    st.write(fulldataset(index0, index1))

    # Nilai Pembanding
    st.sidebar.header('Measurement Parameter')
    similaritas = st.sidebar.checkbox("Similarity & Classification")
    ontology = st.sidebar.checkbox("Ontology Construction")
    
    # Ontology Construction
    if ontology:
       text_to_clean = list(fulldataset(index0, index1)['Requirement Statement'])
       cleaned_text = apply_cleaning_function_to_list(text_to_clean)
       
       # document bag of words
       doc_bow = []
       def bagofwords(text):
           count_vector = CountVectorizer(text)
           count_vector.fit(text)
           doc_array = count_vector.transform(text).toarray()
           doc_bow.append(doc_array)
            
       # document tfidf
       st.subheader('Similarity cosine parameters')
       id_requirement = fulldataset(index0, index1)['ID']
       bow_matrix = pd.DataFrame(doc_bow, index= id_requirement, columns=count_vector.get_feature_names())
       st.dataframe(bow_matrix)  
        
        
    
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
        hasil = hasil_cosine
        st.sidebar.write('anda memilih: cosine')
        st.dataframe(df_cos)
      
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
        st.sidebar.write('anda memilih: levenshtein')
        st.dataframe(df_lev)
      
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
        st.sidebar.write('anda memilih: jaccard')
        st.dataframe(df_jaccard)

      # tfidf
      elif hasil == 'tfidf':
        st.subheader('Similarity tfidf parameters')
        vect = TfidfVectorizer()
        tfidf_matrix = vect.fit_transform(cleaned_text)
        id_requirement = fulldataset(index0, index1)['ID']
        df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), index=id_requirement,  columns = vect.get_feature_names())
        hasil = tfidf_matrix.toarray()
        st.sidebar.write('anda memilih: tfidf')
        st.dataframe(df_tfidf)

      # vsm
      elif hasil == 'vsm':
        st.subheader('Similarity tfidf parameters')
        vect = TfidfVectorizer()
        tfidf_matrix = vect.fit_transform(cleaned_text)
        matrix_tfidf = tfidf_matrix.toarray()
        vsm = cosine_similarity(matrix_tfidf[0:], matrix_tfidf)
        id_requirement = fulldataset(index0, index1)['ID']
        df_vsm = pd.DataFrame(vsm, index=id_requirement,  columns = id_requirement)
        hasil = vsm
        st.sidebar.write('anda memilih: vsm')
        st.dataframe(df_vsm)

      # doc2vec
      elif hasil == 'doc2vec':
        st.subheader('Similarity doc2vec parameters')
        sentences = [word_tokenize(num) for num in cleaned_text]
        for i in range(len(sentences)):
            sentences[i] = TaggedDocument(words = sentences[i], tags = ['sent{}'.format(i)])    # converting each sentence into a TaggedDocument
        model = Doc2Vec(documents = sentences, dm = 1, size = 100, window = 3, min_count = 1, iter = 10, workers = Pool()._processes)
        model.init_sims(replace = True)
        model.save('doc2vec_model')
        model = Doc2Vec.load('doc2vec_model')
        nilai_vektor = [model.infer_vector("sent{}".format(num)) for num in range(0, len(cleaned_text))]
        id_requirement = fulldataset(index0, index1)['ID']
        df_vektor = pd.DataFrame(nilai_vektor, index=id_requirement, columns= ['vektor {}'.format(num) for num in range(0, 100)])
        hasil = nilai_vektor
        st.sidebar.write('anda memilih: doc2vec')
        st.dataframe(df_vektor)

      # sentencemodel
      elif hasil == 'sentencemodel':
        st.subheader('Similarity sentencemodel parameters')
        threshold = 5
        for i in range(len(cleaned_text)):
            if len(cleaned_text[i]) < 5:
                cleaned_text[i] = None
        cleaned_text = [sentence for sentence in cleaned_text if sentence is not None]
        model = Word2Vec(sentences = cleaned_text, size = 100, sg = 1, window = 3, min_count = 1, iter = 10, workers = Pool()._processes)
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
        st.sidebar.write('anda memilih: Sentence Model')
        st.dataframe(df_sentmodel)
      
      # variable training testing
      label_statement = fulldataset(index0, index1)['label']
      size = st.sidebar.slider('test_size', 0.1, 0.6, 0.3)
      X_train, X_test, y_train, y_test = train_test_split(hasil, label_statement, test_size=size,random_state=109) # 70% training and 30% test
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
