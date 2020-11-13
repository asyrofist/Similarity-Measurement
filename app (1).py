import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score 
from sklearn import neighbors, tree, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from function import preprocessing, fulldataset, apply_cleaning_function_to_list, pd, np, similarity_cosine, similarity_levenshtein, similarity_jaccard
from tabulate import tabulate 

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

    # similarity
    if similaritas:
      text_to_clean = list(fulldataset(index0, index1)['Requirement Statement'])
      cleaned_text = apply_cleaning_function_to_list(text_to_clean)
      
      # variable parameter
      st.sidebar.header('Training Parameters')
      hasil = st.sidebar.selectbox('What Similarity Measurement?', ['cosine', 'levenshtein', 'jaccard'])
      
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