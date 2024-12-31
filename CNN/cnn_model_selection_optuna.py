##########################
##### imports ############
##########################
import pandas as pd
import numpy as np
import json
import statistics

import re

import joblib

import os
import math

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, XLNetTokenizer, XLNetModel, XLNetLMHeadModel, XLNetConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn

from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from utils_crisis_classification import clean_text

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold


from tensorflow import keras
from tensorflow.keras import layers

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate

from tensorflow.keras import layers
from tensorflow.keras import regularizers
from attention import Attention


# specify GPU
device = torch.device("cuda")

from Baseline_Models import Display_metrics,Display_classification_report,Confusion_matrix

import optuna
from keras.backend import clear_session

import transformers
from transformers import AutoModel, BertTokenizerFast, BertModel, BertTokenizer

import numpy as np
from sklearn.metrics import average_precision_score

from Baseline_Models import Create_TFIDF,Vector_Encoding_TFIDF,Create_BOW,Vector_Encoding_BOW

### imports (1) ##
import pandas as pd
import numpy as pn
from numpy import mean

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

import string

import collections
from collections import Counter

### imports (2) ##
from string import punctuation
from os import listdir
from numpy import array

from pickle import load
from numpy import array

### imports (3) ##
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
STOPWORDS = set(stopwords.words('english'))

### imports (4) ##
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate

import pickle


##################################################
#### functions to convert labels to numerical ####
##################################################
def class2Index(classList,class2index):
    return [class2index[c] for c in classList]

def train_classes(classes):
    class2index = {}
    index2class = {}
    classCount = 0
    for cl in np.unique(classes):
        if cl not in class2index:
            class2index[cl] = classCount
            index2class[classCount] = cl
            classCount += 1
            
    return class2index,index2class

##################################################
#### functions to clean the corpus ###############
##################################################
def remove_punc(text): 
    text = "".join([char for char in text if char not in string.punctuation ])
    #text = re.sub('[0-9]+', '', text)
    return text

def tokenization(text):
    text = re.split('\W+',text)
    return text

def remove_url(text):
    text = re.sub(r'http\S+', '', text)
    return(text)

def lower_case(text):
    text = text.lower()
    return(text)

def remove_stopwords(text):
    text = [word for word in text if word not in STOPWORDS]
    return text


def clean_text(text): 
    # lower case
    text_lower = lower_case(text)
    
    # remove puntuation
    text_punc = remove_punc(text_lower) 
    
    #remove URLS
    text_url = remove_url(text_punc)
    
    # tokenization
    text_tokens = tokenization(text_url)
    
    # remove stop words
    no_stop_tokens = remove_stopwords(text_tokens)
    
    return no_stop_tokens

########################################################################################################
###        function to clean and tokenize a corpus (to create a vocab given a corpus)  #################
###        input: corpus         output: clean tokens ##################################################
########################################################################################################
def clean_corpus(corpus):
    # convert all to lower case 
    corpus_lower = lower_case(corpus)
    
    # remove punctuation
    corpus_punc = remove_punc(corpus_lower) 
    # remove punctuation from each token
    #table = str.maketrans('', '', string.punctuation)
    #tokens = [w.translate(table) for w in tokens]
    
    #remove URLS
    corpus_url = remove_url(corpus_punc)
    
    # tokenization
    corpus_tokens = tokenization(corpus_url)
    # split into tokens by white space
    #tokens = corpus.split()
    
    # remove stop words
    no_stop_tokens = remove_stopwords(corpus_tokens)
    # filter out stop words
    # stop_words = set(stopwords.words('english'))
    # tokens = [w for w in tokens if not w in stop_words]
    
    # remove remaining tokens that are not alphabetic
    #tokens = [word for word in tokens if word.isalpha()]
    
    # filter out short tokens
    final_tokens = [word for word in no_stop_tokens if len(word) > 1]

    return final_tokens

########################################################################################################
#########  function to clean and tokenize a document based on a given vocabulary   #####################
###        input: document         output: clean document's tokens #####################################
########################################################################################################
def clean_document_vocab(doc, vocab):
    # convert to lower case 
    doc_lower = lower_case(doc)
    
    # remove punctuation
    doc_punc = remove_punc(doc_lower) 

    #remove URLS
    doc_url = remove_url(doc_punc)
    
    # tokenization
    doc_tokens = tokenization(doc_url)
    
    # remove stop words
    doc_no_stop_tokens = remove_stopwords(doc_tokens)
    
    # filter out tokens not in vocab
    tokens = [w for w in doc_no_stop_tokens if w in vocab]
    
    #tokens = ' '.join(tokens)
    return tokens

##########################################################################
# Define encoder architecture MULTI-CHANNEL  #############################
##########################################################################
def create_kim_encoder(trial, length=36, vocab_size=100, embedding=False, embed_params={},optuna_params_encoder={}):
    
    ###### inputs #########
    inputs1 = Input(shape=(length,))
    
    ## Create the embedding layer    ######################################
    if embedding == True:   
        embedding_layer = Embedding(embed_params['Tokenizer_size'] + 1,
                            embed_params['EMBEDDING_DIM'],
                            weights=[embed_params['weights']],
                            input_length=embed_params['input_length'],
                            trainable=False)
        embedding1 = embedding_layer(inputs1)
    else: 
        embedding1 = Embedding(vocab_size, 100)(inputs1)
        
    
    ### LSTM layer #####
    #lstm_layer = LSTM(embed_params['EMBEDDING_DIM'],return_sequences=True)(embedding1)
    
    ### channel 1 #####
    conv1 = Conv1D(filters=trial.suggest_categorical("filters_ch1",optuna_params_encoder['filters']), kernel_size=trial.suggest_categorical("kernel_size_ch1",optuna_params_encoder['kernel_size']), activation='relu')(embedding1)
    drop1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling1D(pool_size=trial.suggest_categorical("pool_size_ch1",optuna_params_encoder['pool_size']),padding='same')(drop1)
    flat1 = Flatten()(pool1)

    ### channel 2 #####
    conv2 = Conv1D(filters=trial.suggest_categorical("filters_ch2",optuna_params_encoder['filters']), kernel_size=trial.suggest_categorical("kernel_size_ch2",optuna_params_encoder['kernel_size']), activation='relu')(embedding1)
    drop2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling1D(pool_size=trial.suggest_categorical("pool_size_ch2",optuna_params_encoder['pool_size']),padding='same')(drop2)
    flat2 = Flatten()(pool2)
    
    
    ### channel 3 #####
    conv3 = Conv1D(filters=trial.suggest_categorical("filters_ch3",optuna_params_encoder['filters']), kernel_size=trial.suggest_categorical("kernel_size_ch3",optuna_params_encoder['kernel_size']), activation='relu')(embedding1)
    drop3 = Dropout(0.2)(conv3)
    pool3 = MaxPooling1D(pool_size=trial.suggest_categorical("pool_size_ch3",optuna_params_encoder['pool_size']),padding='same')(drop3)
    flat3 = Flatten()(pool3)
    

    # merge
    union = concatenate([flat1, flat2, flat3])
    #union = union.reshape(union.size(0), -1)
    # interpretation
    #dense1 = Dense(300, activation='relu')(merged)
    #outputs = Dense(1, activation='sigmoid')(dense1)
    model = keras.Model(inputs=inputs1, outputs=union) #[inputs1, inputs2, inputs3]
  
    return model

###############################################################################
###### Optuna objective function ##############################################
###############################################################################
def objective(trial,df_training,vocab,embeddings_index,ls_save_results=[]):
    clear_session()
    print('************ trial *************')
       
    ############################################################
    ################## Split data for 10-fold CV ###############
    ############################################################
    skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=2018)

    X = df_training['non_cleaned_text']
    TX = np.array(X.tolist())

    Y = df_training['label']
    TY= np.array(Y.tolist())
    ############################################################
    
    folds_results = []
    rep_fold = 1
    for train_index, test_index in skf.split(TX,TY):
        print('--------- Fold ',str(rep_fold),'-------------------------')
        print('Length train index....',len(train_index))
        print('Length test index....',len(test_index))
        
        X_train, X_test = TX[train_index], TX[test_index]
        y_train, y_test = TY[train_index], TY[test_index]
    
        ####### check distribution of positive samples on each fold ######
        count_test = (y_test == 1).sum()
        count_train = (y_train == 1).sum()
        print('positive samples in train......',count_train)
        print('positive samples in test......',count_test)
        print('------------------------------------------')
        
        
        ######################################################################
        ############### Create CNN architecture  #############################
        ######################################################################
        
        

        
                
        ######################################################################
        ############### Create BERT features  ################################
        ######################################################################
        with torch.no_grad():
            outputs = fine_tuned_bert_model.bert(train_seq,train_mask)

        with torch.no_grad():
            outputs_test = fine_tuned_bert_model.bert(test_seq,test_mask)

        ls_input_train = [e.numpy() for e in outputs[1]]
        ls_input_test = [e.numpy() for e in outputs_test[1]] 

        print('Length training data (bert)',len(ls_input_train))
        print('Length testing data (bert)',len(ls_input_test))
        print('Length training vectors (bert)',len(ls_input_train[0]))
        print('Length testing vectors (bert)',len(ls_input_test[0]))

        
        ######################################################################
        ############### convert to logical features ##########################
        ######################################################################
        train_logical_features = Calculate_logical_filters(X_train)
        test_logical_features = Calculate_logical_filters(X_test)
        
        print('Length logical feature vector',len(train_logical_features[0]))
        print('Length logical feature vector',len(test_logical_features[0]))
        print('Length train logical',len(train_logical_features))
        print('Length test logical',len(test_logical_features))

        expanded_train_logical = []
        for e in train_logical_features: #logical_train_loc #train_logical_features #train_logical_loc
            expanded_e = expand_list(e)
            expanded_train_logical.append(expanded_e)   

        expanded_test_logical = []
        for e in test_logical_features: #logical_test_loc   #test_logical_features
            expanded_e = expand_list(e)
            expanded_test_logical.append(expanded_e)  

        print('Length logical feature vector',len(expanded_train_logical[0]))
        print('Length logical feature vector',len(expanded_test_logical[0]))
        print('Length train logical',len(expanded_train_logical))
        print('Length test logical',len(expanded_test_logical))


        ############################################################
        ####  search space #########################################
        ############################################################
        #learning_rates = [0.01,0.001,0.0001,0.00002]
        learning_rates = [0.01,0.001,0.0001]
        #nb_epochs = [50,100,200]
        nb_epochs = 50
        max_seq = 36
        #batch_size_ls = [8,16,32,64,128]
        batch_size_ls = [32,64,128]

        nb_layers_top = [1,2,3]

        ls_hidden_units_bert = [512,256,128,64]
        ls_hidden_units_hybrid = [128,100,96,64,32,16]

        select_nb_layers_bert = trial.suggest_categorical("nb_layers_bert_top",nb_layers_top)
        select_nb_layers_hybrid = trial.suggest_categorical("nb_layers_hybrid",nb_layers_top)
        ############################################################
        ############################################################
        ############################################################
        
    
        ############################################################
        #### Build integrated model ################################
        ############################################################
        #### create Keras neural network with both inputs ###
        input_bert = Input(shape=(768,))
        input_logical = Input(shape=(55,))

        if select_nb_layers_bert == 1:
            hid1 = trial.suggest_categorical("bert_top_hidden_1",ls_hidden_units_bert)

            x = Dense(hid1, activation="relu")(input_bert)
            b1 = Model(inputs=input_bert, outputs=x)

        elif select_nb_layers_bert == 2:
            hid1 = trial.suggest_categorical("bert_top_hidden_1",ls_hidden_units_bert)
            hid2 = trial.suggest_categorical("bert_top_hidden_2",ls_hidden_units_bert)

            x = Dense(hid1, activation="relu")(input_bert)
            x = Dense(hid2, activation="relu")(x)
            b1 = Model(inputs=input_bert, outputs=x)

        elif select_nb_layers_bert == 3:
            hid1 = trial.suggest_categorical("bert_top_hidden_1",ls_hidden_units_bert)
            hid2 = trial.suggest_categorical("bert_top_hidden_2",ls_hidden_units_bert)
            hid3 = trial.suggest_categorical("bert_top_hidden_3",ls_hidden_units_bert)

            x = Dense(hid1, activation="relu")(input_bert)
            x = Dense(hid2, activation="relu")(x)
            x = Dense(hid3, activation="relu")(x)
            b1 = Model(inputs=input_bert, outputs=x)

        # combine the output of the two branches
        combined = concatenate([b1.output, input_logical]) ##y.output #input_logical


        ## add layers
        if select_nb_layers_hybrid == 1:
            hybrid_top_hidden1 = trial.suggest_categorical("hybrid_top_hidden_1",ls_hidden_units_hybrid)

            z1 = Dense(hybrid_top_hidden1, activation="relu")(combined)
            z = Dense(2, activation="softmax")(z1)

        elif select_nb_layers_hybrid == 2:
            hybrid_top_hidden1 = trial.suggest_categorical("hybrid_top_hidden_1",ls_hidden_units_hybrid)
            hybrid_top_hidden2 = trial.suggest_categorical("hybrid_top_hidden_2",ls_hidden_units_hybrid)

            z1 = Dense(hybrid_top_hidden1, activation="relu")(combined)
            z1 = Dense(hybrid_top_hidden2, activation="relu")(z1)
            z = Dense(2, activation="softmax")(z1)


        elif select_nb_layers_hybrid == 3:
            hybrid_top_hidden1 = trial.suggest_categorical("hybrid_top_hidden_1",ls_hidden_units_hybrid)
            hybrid_top_hidden2 = trial.suggest_categorical("hybrid_top_hidden_2",ls_hidden_units_hybrid)
            hybrid_top_hidden3 = trial.suggest_categorical("hybrid_top_hidden_3",ls_hidden_units_hybrid)

            z1 = Dense(hybrid_top_hidden1, activation="relu")(combined)
            z1 = Dense(hybrid_top_hidden2, activation="relu")(z1)
            z1 = Dense(hybrid_top_hidden3, activation="relu")(z1)
            z = Dense(2, activation="softmax")(z1)


        # our model will accept the inputs of the two branches and then output a single value
        model = Model(inputs=[b1.input,input_logical], outputs=z) 
        print(model)

        print('---------- Start training -------------')
        lr=trial.suggest_categorical("learning_rate",learning_rates)
        batch=trial.suggest_categorical("batch_size",batch_size_ls)
        #n_epochs = trial.suggest_categorical("epochs",nb_epochs)

        Train_hybrid_bert = np.array(ls_input_train)
        Train_hybrid_logical = np.array(expanded_train_logical)
        Train_hybrid_Y = np.array(y_train.tolist())

        Test_hybrid_bert = np.array(ls_input_test)
        Test_hybrid_logical = np.array(expanded_test_logical)
        Test_hybrid_Y = np.array(y_test.tolist())

        model.compile(optimizer=keras.optimizers.Adam(lr),
                loss=keras.losses.SparseCategoricalCrossentropy(),  
                metrics=[keras.metrics.SparseCategoricalCrossentropy()], 
        )

        history = model.fit(x=[Train_hybrid_bert,Train_hybrid_logical], y=Train_hybrid_Y, batch_size=batch, epochs=nb_epochs,verbose=2) 

        predictions = model.predict([Test_hybrid_bert,Test_hybrid_logical])
        preds = np.argmax(predictions, axis=1)
        dict_r = classification_report(y_test.tolist(), preds, output_dict = True)
        
        ### calculate probs for precision-recall curve calculation #####
        pos_probs = predictions[:, 1]   
        _fold_AP = average_precision_score(y_test.tolist(),pos_probs)
        print("AP score for class 1 --->",_fold_AP)

        Fold_f1 = dict_r['1']['f1-score']
        print('fold F1 score:',Fold_f1)
        Fold_recall = dict_r['1']['recall']
        print('fold recall score:',Fold_recall)
        Fold_precision = dict_r['1']['precision']
        print('fold precision score:',Fold_precision)
        Fold = {'f1':Fold_f1,'recall':Fold_recall,'precision':Fold_precision,'AP':_fold_AP}
        folds_results.append(Fold)
        
        rep_fold =rep_fold + 1
        
    #### calculate average F1 score ####
    AP_scores_folds = []
    f1_scores_folds = []
    recall_scores_folds = []
    precision_scores_folds = []
    
    for i in folds_results: 
        f1_scores_folds.append(i['f1'])
        recall_scores_folds.append(i['recall'])
        precision_scores_folds.append(i['precision'])
        AP_scores_folds.append(i['AP'])
    
    average_AP_folds = statistics.mean(AP_scores_folds)
    print('trial average AP....',average_AP_folds)
    average_f1_folds = statistics.mean(f1_scores_folds)
    print('trial average F1....',average_f1_folds)
    average_recall_folds = statistics.mean(recall_scores_folds)
    print('trial average recall....',average_recall_folds)
    average_precision_folds = statistics.mean(precision_scores_folds)
    print('trial average precision....',average_precision_folds)
    
    trial_results = {'f1':average_f1_folds,'recall':average_recall_folds,'precision':average_precision_folds,'AP':average_AP_folds}
    ls_save_results.append(trial_results)
    

    return(average_AP_folds) 
    
###############################################################################
###### main function             ##############################################
###############################################################################   
def main():
    
    ##################################################################################
    ########      Data Reading and preprocessing    ##################################
    ##################################################################################
    
    ## Read Harvey data set ###########
    labeledDF=pd.read_csv("/home/wkhal001/Desktop/data_rescue_mining/labeled_ds_Corrected_csv_updated_22_09_13.csv") 
    del labeledDF['Unnamed: 0']
    
    ## Extract useful columns #########
    df_training = labeledDF[['id','text','cleaned_tweet','sos.correct']]
    df_training.columns= ['id','non_cleaned_text','text','label']
    
    print(df_training['label'].value_counts()
    
    ## Create vocabulary ############
    corpus = ''
    for i in range(len(df_training)) :
        st = df_training.iloc[i]["non_cleaned_text"]
        corpus = corpus + " " + st
    
    T = clean_corpus(corpus)

    ## count vocabulary with collection counter ####
    vocab = Counter()
    vocab.update(T)
    print('Vocabulary created...',len(vocab),' tokens')
    
    
    ##################################################################################
    ########      Load pretrained embedding    #######################################
    ##################################################################################
    path_to_glove = '/home/wkhal001/Desktop/gensim-data/glove-twitter-200/glove-twitter-200.txt' 

    ##### Load the file content in a dictionary ###
    embeddings_index = {}
    with open(path_to_glove) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))

    
    ### TOKENIZE DATA #######     
    tokenizer = Tokenizer(num_words=None,oov_token='OOV')
    tokenizer.fit_on_texts(df_training['text'])
    vocab_size = len(tokenizer.word_index) + 1
    print('Tokenizer vocabulary size ...',vocab_size)            
    
    ### Initialize the embedding layer ####  
    EMBEDDING_DIM=200
    vocabulary_size=len(tokenizer.word_index)+1
    print('Vocaublary size:',vocabulary_size)
    print('Embedding dimension:',EMBEDDING_DIM)
    print('Create embedding matrix in progress....')
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))

    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print('Embedding matrix created')
    print('Embedding matrix size',len(embedding_matrix))
    print('\n')
          
    ### Optimize ####
    study = optuna.create_study(direction="maximize")
    ls_save_results=[] 
    study.optimize(lambda trial: objective(trial,df_training,vocab,embeddings_index,ls_save_results), n_trials=400)  
    
    result_path = "/home/wkhal001/Crisis-classification/scripts-rescue-detection/___rescue_identification_experiments/CNN_experiments/cnn_results/optuna_cnn_results_harvey.pkl" 
    joblib.dump(ls_save_results,result_path)
    study_path = "/home/wkhal001/Crisis-classification/scripts-rescue-detection/___rescue_identification_experiments/CNN_experiments/cnn_results/study_optuna_cnn_results_harvey.pkl"
    
    ## Save data for this experiment ####
    joblib.dump(study, study_path)
    #study = joblib.load(study_path)
    #study.best_trial.number

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print('\n')


if __name__ == "__main__":
    main()