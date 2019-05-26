#-------------------------------------------------------------PACKAGES-------------------------------------------------------#
import numpy as np
import pandas as pd 
import re
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import qgrid
import os.path

#-----TensorFlow packages----# 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import set_random_seed
print(tf.VERSION)
print(tf.keras.__version__)

#-----GPU config with Tensorflow----# 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#-----scikit-learn packages---# 
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, StratifiedShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
    
#-----Keras packages----# 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Flatten
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l2
from keras.utils import plot_model
from keras.models import load_model
import h5py

#-----Google's packages for language detection----# 
from langdetect import detect
from langdetect import DetectorFactory
DetectorFactory.seed = 0

#----------------------------------------------------GLOBAL VARIABLES---------------------------------------------------------#
SEED = 1 # do not modify - used for reproducible results
SEED_TF= 2 # do not modify - used for reproducible results

#----------------------------------------------------FUNCTIONS DEFINITIONS----------------------------------------------------#

def data_preprocess(samples, dataset, file_path, cols, labels):
    """
    Performs data processing: Outputs clean data corresponding to text with "target" values
    for positive and negative sentiments in a Pandas DataFrame.
    
    Arguments:
    samples -- number of samples to select from dataset.
    dataset -- string value corresponding to the name of a dataset  ('airline', 'sentiment140', 'self-driving-cars')
               used for selection of pre-processing options depending on the dataset structure.
    file_path -- file path to dataset location.
    cols -- column numbers in dataset associated with target and text values.
    SEED -- seed used in data sampling and for reproducible results.
    
    Returns:
    data -- clean data after pre-processing
    pos_sent_size -- number of positive sentiments in dataset
    neg_sent_size -- number of negative sentiments in dataset
    neut_sent_size -- number of neutral sentiments in dataset
    """
    
    np.random.seed(SEED)
    dataset = dataset.lower()
    
    file_name = file_path.split(".")[-1]
    
    if file_name == 'tsv'and labels:
        data = pd.read_csv(file_path, usecols = cols, sep ='\t',  names = ["id", "target", "label_a", "text"]) 
    elif file_name == 'csv' and labels:
        data = pd.read_csv(file_path, usecols = cols)
    elif file_name =='tsv' and not labels:
        data = pd.read_csv(file_path, usecols = cols, sep ='\t',  names = ["id", "text","test_label"], header = None) 
    elif file_name =='csv' and not labels:
        data = pd.read_csv(file_path, usecols = cols, sep ='\t',  names = ["id", "text","test_label"],header =None) 
    else:
        return print('File format not supported. Only .csv or .tsv!')
    
    #sample data
    data = data.sample(frac=1).reset_index(drop = True)
    data= data.iloc[0:samples]

#     #ID for rows
#     data['id'] = range(1,1+len(data))

#     #Alpha character for rows
#     data['alpha'] ='a' 
    
    if labels:
       
        #process labels to 0 and 1
        if dataset == 'airline':
            data['target'] = data['target'].apply(lambda x: 1 if x == 'negative' else 0 if x == 'positive' else 2)
        elif dataset == 'sentiment140':
            data['target'] = data['target'].apply(lambda x: 1 if x == 0 else 0)
        elif dataset == 'self-driving-cars':
            data['target'] = data['target'].apply(lambda x: 0 if x == '5' or x=='4' else 1 if x == '2' or x=='1' else 3)
        elif dataset == 'products':
            data['target'] = data['target'].apply(lambda x: 1 if x == 'negative' else 0 if x == 'positive' else 3)
        elif dataset == 'apple':
            data['target'] = data['target'].apply(lambda x: 0 if x == '5' else 1 if x=='1' else 3)
        elif dataset =='ied' or 'all':
            data['target'] = data['target'] #labels already in the correct format
        else:
            data['target'] = None
        
        if dataset == 'airline':
            neut_sent=data[data['target'] == 2]

            neut_sent_size =neut_sent['target'].size # number of neutral sentiments
        elif dataset == 'self-driving-cars' or dataset =='products' or dataset =='apple':
            neut_sent=data[data['target'] == 3]
            neut_sent_size =neut_sent['target'].size # number of neutral sentiments
        else:
            neut_sent = pd.DataFrame(np.array([None]))           
         
        neg_sent = data[data['target'] == 1]
        pos_sent=data[data['target'] == 0]
        neg_sent_size = neg_sent['target'].size # number of negative sentiments
        pos_sent_size = pos_sent['target'].size # number of positive sentiments
        
        #remove neutral text or those without 'positive' or 'negative' labels
        if dataset == 'airline':
            data.loc[data.target == 2, "text"] = ' '
            data= data.drop(data[data.text == ' '].index) #remove all neutral sentiments (those with target value of 2 and text = ' ')
        elif dataset == 'self-driving-cars' or dataset == 'products' or dataset== 'apple':
            data.loc[data.target == 3, "text"] = ' '
            data= data.drop(data[data.text == ' '].index) #remove all neutral sentiments (those with target value of 3 and text = ' ')


    #filter unwanted characters in text
    data['text'] = data.text.apply(lambda x: str(x).lower())
    data['text'] = data.text.apply((lambda x: re.sub(r'http\S+','',str(x))))#remove all text that contains starts with http
    data['text'] = data.text.apply((lambda x: re.sub(r'pic.twitter.com\S+','',str(x)))) #remove all text referencing pictures in tweets
    data['text'] = data.text.apply((lambda x: re.sub(r'@\S+','',str(x)))) #remove the @name in tweets
    data['text'] = data.text.apply((lambda x: re.sub('[^a-zA-Z0-9\s]','',str(x)))) #replace with spaces, with the exception of [^a-zA-Z0-9]
    data['text'] = data.text.apply((lambda x: re.sub('\s',' ',str(x)))) #remove new line characters (\n)
    
    #remove all retweets
    for idx,row in data.iterrows():
        row['text'] = row['text'].replace('rt',' ')

    if not labels:
        return (data,None,None,None)
    elif labels and not (None in neut_sent.values):
        return (data,pos_sent_size, neg_sent_size, neut_sent_size) 
    else:
        return (data,pos_sent_size, neg_sent_size, None)
    

def keras_tokenizer(text, max_fatures, col_name, max_len):
    """
    Performs text tokenization: Fits a tokenization function to the "text" values in the cleaned data after pre-processing.
    Outputs a sequence of tokenized text and an instance of the Tokenizer class specific to the "text" values of the dataset.
    
    Arguments:
    text -- text values in the cleaned data after pre-processing.
    max_fatures -- maximum number of features in the dataset to consider. 
    col_name -- 'text' column in a Pandas DataFrame; used to locate text for tokenization.
    
    Returns:
    sequences -- transformed text into sequences after tokenization; required for Keras Sequential model.
    tokenizer -- an object/ instance of the Tokenizer class specific to the text values of the dataset. 
    """
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(text[col_name].values)
    sequences = tokenizer.texts_to_sequences(text[col_name].values)
    sequences = pad_sequences(sequences, max_len) 
    return sequences, tokenizer


#----------------------------------------------------MODEL DEFINITIONS----------------------------------------------------#
def lstm_closure(embedding_dim, units, batch_size, max_features,input_length):
    """
    A closure: Allows the "create_lstm_model()" function to access "embedding_dim","units" and "batch_size" variables 
    through the closure’s copies of their values or references, even when the function is invoked outside their scope.
    
    Arguments:
    embedding_dim -- embedding size; number of neurons in the hidden layer.
    units -- dimensionality of the output space or hidden state of the lstm.
    batch_size -- number of training samples propagated through the network in one forward/backward pass.
    max_features -- maximum number of features (e.g. words) to consider.
    input_length --length of the input vector
    
    Returns:
    create_lstm_model -- Keras Sequential model as a nested function in the scope of "lstm_closure()".
    """
    def create_lstm_model():
        tf.set_random_seed(1234)
        model = tf.keras.Sequential()
        model.add(layers.Embedding(max_features, embedding_dim, input_length=input_length)) #(size of vocabulary, size of embedding vector, length of input (i.e. no. steps/words in each sample))
        model.add(layers.SpatialDropout1D(0.4)) # fraction of the input units to drop
        model.add(layers.CuDNNLSTM(units)) 
        model.add(layers.Dense(100, input_dim =2, activation = 'relu', kernel_regularizer=l2(0.001)))
        model.add(layers.Dense(2,activation='softmax')) # sigmoid function when no. of classes is 2 in a softmax function   
#         model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop',metrics = ['accuracy']) 
        model.compile(loss = 'binary_crossentropy', optimizer='rmsprop',metrics = ['accuracy']) 
        print(model.summary())
        return model
    return create_lstm_model


def plot_learning_performance(history, dataset, samples, batch_size, max_len, epochs):
    """
    Classifier learning performance: Plots accuracy and loss for training and validation sets in the model.
    
    Arguments:
    history -- Keras object from fit() function used to train the model. 
    
    Returns:
    min_epoch -- epoch where the validation loss is at its minimum.
    """
    # Plot training & validation accuracy values
    plt.figure(figsize=(10,5))
    plt.plot(history.history['acc'], 'b')
    plt.plot(history.history['val_acc'], 'r') 
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train acc', 'Validation acc'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'],'b')
    plt.plot(history.history['val_loss'],'r')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper left')
    plt.savefig(f'/home/hummus/algorithms-other/LSTM/figures/learning/learning_performance_{dataset}_{samples}_batchsize{batch_size}_maxlen{max_len}_epochs{epochs}.png')
    plt.show()

    #find epoch where the validation loss is at its minimum
    min_epoch = np.argmin(history.history['val_loss']) + 1
    return min_epoch


def keras_predict(text_array,flag, model, max_len):
    """
    Performs predicitons: returns logits (prediction probabilities for positive and negative sentiments) corresponding
    to text (e.g. tweets) for sentiment classification.
    
    Arguments:
    text_array -- numpy array containing text for sentiment classification. 
    flag -- string value to use keras_predict() in either of two functions:
            flag = 'one' returns a list with logits corresponding to probabilities for each text (i.e. a tweet) in the dataset.
            flag = 'all' returns a list of lists with logits corresponding to probabilities of all text entries (i.e. tweets)
            in the dataset.
    
    Returns:
    logits -- classification probabilities ([pos, neg]) corresponding to sentiments associated with text in text_array. 
    """ 
    if flag == 'one':
        for obj in text_array:
            text_array_reshaped = np.reshape(obj, (1,max_len))
        logits = model.predict(text_array_reshaped,batch_size=1,verbose = 0)[0] 
    elif flag == 'all':
#         for obj in text_array:
#             text_array_reshaped = np.reshape(obj, (text_array.shape[0],max_len-1))
        logits = model.predict(text_array)    
    return logits    

def display_predictions(sequences,tokenizer, model, max_len):     
    """
    Display predictions: calls keras_predict() function which returns prediction probabilities that are then formatted
    and stored into a Pandas DataFrame for display.
    
    Arguments:
    sequences -- text in the form of sequences after tokenization using the keras_tokenizer() function.
    tokenizer -- tokenizer fitted to sequences and returned by the keras_tokenizer() function.
       
    Returns:
    df -- Pandas DataFrame containing sentiment probabilities, text and sentiment predictions (i.e. 'positive', 'negative').
    """
    #splitting into test and train test is re-done here since in the previous steps
    #x_test and y_test are in the form of transformed sequences
    twt_sent_prediction =[]
    x_test_twt = tokenizer.sequences_to_texts(sequences)
    twt_sentiment = keras_predict(sequences,'all', model, max_len)
    for logits in twt_sentiment:
        if(np.argmax(logits) == 1):
            twt_sent_prediction.append('negative') 
        elif (np.argmax(logits) == 0):
            twt_sent_prediction.append('positive')
    df = pd.DataFrame({'Sentiment Probabilities' : [*twt_sentiment]})
    df['Text'] = x_test_twt
    df['Sentiment Prediction'] = twt_sent_prediction 
    return df


def detect_language(data):
    '''Detect language and remove non-english text'''
    languages = []
    lang = ''
    for obj in data.text:
        try:
            lang = detect(obj)
            counter += 1
            print(counter)
            print(lang)
            languages.append(lang)            
        except Exception as e:
    #         print(e)
            languages.append(lang)

    data['language'] = languages
    data.loc[data.language != 'en', "text"] = ' '   ## remove all rows that are not in English
    data = data.drop(data[data.text == ' '].index)
    return data
