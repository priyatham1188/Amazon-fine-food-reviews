import nltk
import re
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from operator import itemgetter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing.text import Tokenizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def task1(path):
    




    df = pd.read_csv(r"C:\Users\ikpri\Downloads\Reviews.csv\Reviews.csv")
    print(df.shape)

    data = df
    # We have reviews with rating from 1 to 5 where rating = 3 is considered as neutral and now we are removing those so that we can have either positive or negative reviews in the dataset

    data = data[data['Score']!=3]
    print(data.shape)

    def scoring(x):
        if(x<3):
            return 0
        else:
            return 1
    
    score = data['Score']
    temp_score = score.map(scoring)
    data['Score']= temp_score

    print(data.head(50))

    """x = 'Score'
    fig, ax = plt.subplots()
    fig.suptitle(x, fontsize=12)
    data[x].reset_index().groupby(x).count().sort_values(by= 
       "index").plot(kind="barh", legend=False, 
        ax=ax).grid(axis='x')
    plt.show()"""


    data_updated = data.drop_duplicates(subset=('UserId','ProfileName','Time','Text'))
    data_updated.shape
    #Removing the data where helpfulnessNumerator is more than helpfulnesss denominator as it's incorrect
    final = data_updated[data_updated['HelpfulnessNumerator']<=data_updated['HelpfulnessDenominator']]
    final.shape

    positive_count=0
    negative_count=0
    scores_updated=[]
    for row in final.itertuples():
        if row[7] == 1 and positive_count < 25000:
            scores_updated.append(row)
            positive_count+=1
        elif row[7] == 0 and negative_count < 25000:
            scores_updated.append(row)
            negative_count+=1
    print(len(scores_updated))
    data_final = pd.DataFrame(scores_updated)

    X_final = data_final['Text']
    y_final = data_final['Score']

    lst_stopwords = nltk.corpus.stopwords.words("english")

    def text_preprocessing(text,stem,stopwords_list=None):
        text.lower()
        htmlr = re.compile('<.*?>')
        text = re.sub(htmlr, ' ', text)        
        text = re.sub(r'[?|!|\'|"|#]',r'',text)
        text = re.sub(r'[.|,|)|(|\|/]',r' ',text)
    
        text_lst = text.split()
    
        if stopwords_list is not None:
            text_lst = [word for word in text_lst if word not in stopwords_list ]
    
        if stem == True:
            snow_stem = nltk.stem.SnowballStemmer('english')
            text_lst = [snow_stem.stem(word) for word in text_lst]
        
        text = " ".join(text_lst)
        return text

    data_final["text_clean"] = X_final.apply(lambda x: text_preprocessing(x, True, lst_stopwords))

    X_final = data_final["text_clean"]

    X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final, test_size=0.2, random_state=42
        )


    def lstm(X_train, X_test, y_train, y_test):
    
        #Tokenizer builds a lexicon of words found in reviews, as well as an index depending on how often they appear. 'num words' will assist you keep as many words as possible depending on their frequency.
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(X_train)

        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)
        # we pad sequences and make all the reviews of same dimension 
        max_review_length = 500
        X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
        X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    
        # The embedding creates the num_words size of matrix and will allocate 1 * 32 matrix for each word and thus repeats in the loop and makes 500*32 of vector dimension or params 
        embedding_vecor_length = 32
        model = Sequential()
        model.add(Embedding(5000, embedding_vecor_length, input_length=max_review_length))

        # Here I've 2LSTM layers where one carries the output to its second layer through return sequences and it has 100 params/units and 50 units respectively
        model.add(LSTM(100,return_sequences=True)),
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        model.fit(X_train, y_train, epochs=5)
        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy" , (scores[1]*100))


    lstm(X_train, X_test, y_train, y_test)

    def Naive_bayes(X_train, X_test, y_train, y_test):

        #Naive bayes on Bow

        Count_Vect = CountVectorizer(max_features = 5000)
        final_X_train = Count_Vect.fit_transform(X_train)
        final_X_test = Count_Vect.transform(X_test)
        param_grid= {'alpha':[0.1,1,10,100]}
        optimal_param = optimal_params(MultinomialNB(),param_grid,final_X_train,y_train)
        clf = MultinomialNB(alpha = optimal_param['alpha'])
        clf.fit(final_X_train,y_train)
        pred = clf.predict(final_X_test)
        acc1 = accuracy_score(y_test, pred) * 100
        f11 = f1_score(y_test, pred) * 100
        print('\nAccuracy='  ,acc1)
        print('F1-Score=' ,f11)
        cm = confusion_matrix(y_test,pred)
        print("\n",cm)


    def optimal_params(clf,param_grid,X_train,y_train):
    
        grid = GridSearchCV(estimator = clf,param_grid=param_grid ,cv = 5,n_jobs = -1)
        grid.fit(X_train, y_train)
        print("best param = ", grid.best_params_)
        print("Accuracy on train data = ", grid.best_score_*100)
        
        a = grid.best_params_
        
        return a
    


    Naive_bayes(X_train, X_test, y_train, y_test)
    
    def RF(X_train, X_test, y_train, y_test):
        Count_Vect = CountVectorizer(max_features = 5000)
        final_X_train = Count_Vect.fit_transform(X_train)
        final_X_test = Count_Vect.transform(X_test)

        param_grid= {'n_estimators':[10,50,100],'max_depth':[10,50,100,150]}
        optimal_param = optimal_params(RandomForestClassifier(),param_grid,final_X_train,y_train)
        clf =RandomForestClassifier(n_estimators=optimal_param['n_estimators'],max_depth=optimal_param['max_depth'])
        clf.fit(final_X_train,y_train)
        pred = clf.predict(final_X_test)
        acc1 = accuracy_score(y_test, pred) * 100
        f11 = f1_score(y_test, pred) * 100
        print('\nAccuracy='  ,acc1)
        print('F1-Score=' ,f11)
        cm = confusion_matrix(y_test,pred)
        print("\n",cm)


    RF(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    # Imports
    task1('../data/Reviews.csv')
    
    
    

    

    

    



    


    






    

        














    




