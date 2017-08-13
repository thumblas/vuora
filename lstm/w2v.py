from __future__ import print_function

import sys
import csv
import string
import numpy as np
import gensim
from gensim.corpora.dictionary import Dictionary
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer

from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


from keras.utils import np_utils                    # For encoding
from keras.preprocessing import sequence            # Padding
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Dropout
from keras.models import model_from_json
from gensim.models.keyedvectors import KeyedVectors
word2vec = gensim.models.Word2Vec

vocab_dim = 336026                         #2196018 for previous
dim = 300
batch_size = 16
n_epoch = 10
input_length = 50





def sentence_to_wordlist( sentence, remove_stopwords=False ):

    #text = sentence.translate(None, string.punctuation)
    #words = text.lower().split()
    words = nltk.word_tokenize(sentence.lower())
    wordnet_lemmatizer = WordNetLemmatizer()
    
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #print words
    
    b=[]
    # If stemming is to be used
    #stemmer = english_stemmer
    for word in words:
        b.append(wordnet_lemmatizer.lemmatize(word))
    #print b
    
    return(words)




def load_files():

    data =[]
    target = []
    
    with open('/home/flash/Documents/spinaxon/emotiondetection/affect_dataset.csv','rU') as csv_file:
        reader = csv.reader(csv_file,delimiter=",",quotechar='"',dialect=csv.excel_tab)
        for row in reader:
            row[0] = unicode(row[0],errors='ignore')
            data.append(row[0])
            target.append(row[1])
    
    with open('/home/flash/Documents/spinaxon/emotiondetection/combined_dataset.csv','rU') as csv_file:
        reader = csv.reader(csv_file,delimiter=",",quotechar='"',dialect=csv.excel_tab)
        for row in reader:
            row[0] = unicode(row[0],errors='ignore')
            data.append(" ".join(sentence_to_wordlist(row[0])))
            target.append(row[1])
    
    with open('/home/flash/Documents/spinaxon/emotiondetection/semval_dataset.csv','rU') as csv_file:
            reader = csv.reader(csv_file,delimiter=",",quotechar='"',dialect=csv.excel_tab)
            for row in reader:
                row[0] = unicode(row[0],errors='ignore')
                data.append(row[0]) 
                target.append(row[1])
    '''
    with open('/home/prathamesh/BE_project/emo.csv','rU') as csv_file:
            reader = csv.reader(csv_file,delimiter="@",quotechar='"',dialect=csv.excel_tab)
            for row in reader:
                row[0] = unicode(row[0],errors='ignore')
                data.append(row[0])
                target.append(row[1])

    '''
    print (len(data))
    return data,target






def evaluate_model(target_true,target_predicted):

    #print (f1_score(target_true, target_predicted, average='weighted'))
    print ("The accuracy score is {:.2%}".format(accuracy_score(target_true,target_predicted)))
    y_actu = pd.Series(target_true, name='Actual')
    y_pred = pd.Series(target_predicted, name='Predicted')
    df_confusion = pd.crosstab(y_actu,y_pred)
    print (df_confusion)
    print (classification_report(target_true, target_predicted))




def any2unicode(text, encoding='utf8', errors='strict'):
    
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    if isinstance(text, unicode):
        return text

    return unicode(text.replace('\xc2\x85', '<newline>'), encoding, errors=errors)


gensim.utils.to_unicode = any2unicode





model = gensim.models.KeyedVectors.load_word2vec_format('/home/flash/Documents/yash_papers/glove.6B_2/glove.6B.300d.txt', binary=False)

print("Model loaded")

#print (model.most_similar('good'))


gensim_dict = Dictionary()
gensim_dict.doc2bow(model.vocab.keys(), allow_update=True)
index_dict = {v: k+1 for k, v in gensim_dict.items()}
word_vectors = {word: model[word] for word in index_dict.keys()}


print('Setting up Arrays for Keras Embedding Layer...')
n_symbols = len(index_dict) + 1  # adding 1 to account for 0th index
embedding_weights = np.zeros((n_symbols, dim))
for word, index in index_dict.items():
    embedding_weights[index, :] = word_vectors[word]


print('All okay')


def sentence_to_vectors(data,test):

    max_len = 0
    transformed_train = []
    transformed_test = []
    
    for sent in data:
        #print (sent)
        #txt = sent.translate(None, string.punctuation)
        txt = nltk.word_tokenize(str(sent).lower().replace("'s",'is'))   #More text processing later
        if len(txt) > max_len:
            max_len = len(txt)
        new_txt = []
        for word in txt:
            try:
                new_txt.append(index_dict[word])
            except:
                new_txt.append(0) # Vector of new word is set to 0
        transformed_train.append(new_txt)

    
    for sent in test:
        #txt = sent.translate(None, string.punctuation)
        txt = nltk.word_tokenize(str(sent).lower().replace("'s",'is'))   #More text processing later
        if len(txt) > max_len:
            max_len = len(txt)
        new_txt = []
        for word in txt:
            try:
                new_txt.append(index_dict[word])
            except:
                new_txt.append(0) # Vector of new word is set to 0
        transformed_test.append(new_txt)

    
    print(len(transformed_train))
    print(max_len)
    print(len(transformed_test))
    
    return transformed_train, transformed_test, max_len



#SAVE LE NEXT TIME

le = preprocessing.LabelEncoder()
data, labels = load_files()
le.fit(labels)
print(le.classes_)
#encoded_labels = list(le.transform(labels))
#encoded_labels = np_utils.to_categorical(labels)
data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,labels,test_size=0.20,random_state=2357)
target_train = list(le.transform(target_train))
#target_test = list(le.transform(target_test))
target_train = np_utils.to_categorical(target_train)
#target_test = np_utils.to_categorical(target_test)
features_train, features_test, max_len = sentence_to_vectors(data_train,data_test)
#print(features)
#print(labels)
#print(encoded_labels)

print("Padding sequences")
train_features = sequence.pad_sequences(features_train, maxlen=max_len)
test_features = sequence.pad_sequences(features_test, maxlen=max_len)
#print('Data shape:', new_features.shape)


'''
print(data_train.shape)
print(data_test.shape)
print(target_train.shape)
print(target_test.shape)
'''






# LSTM
print('Defining a Simple Keras Model...')
model = Sequential()  
model.add(Embedding(output_dim=dim,
                    input_dim=n_symbols,
                    mask_zero=True,
                    weights=[embedding_weights],
                    input_length=max_len))  
model.add(LSTM(max_len))
model.add(Dropout(0.20))
model.add(Dense(5, activation='sigmoid'))


print('Compiling the Model...')
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print("Training...")
model.fit(train_features, target_train, batch_size=batch_size, nb_epoch=5)



print("Evaluate...")
#score, acc = model.evaluate(test_features, target_test,batch_size=batch_size)

prediction_values = model.predict(test_features)
predicted_labels = []
#print (prediction_values)

for i in prediction_values:
    label = i.tolist().index(max(i))
    predicted_labels.append(label)

#print(predicted_labels)
predicted_labels_final = list(le.inverse_transform(predicted_labels))
#print (predicted_labels_final)
#print (target_test)
#print (predicted_labels)
evaluate_model(predicted_labels_final,target_test)


model_json = model.to_json()
with open("model_1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_1.h5")
print("Saved model to disk")


'''
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
'''








'''

prediction_values = model.predict(test_features)
predicted_labels = []

for i in prediction_values:
    label = i.tolist().index(max(i))
    predicted_labels.append(label)


print(len(predicted_labels))
final_list = pd.DataFrame(
    {'Id': test['Id'].tolist(),
     'Prediction': predicted_labels
    })



final_list.to_csv('submission_1.csv', index = False)
'''
