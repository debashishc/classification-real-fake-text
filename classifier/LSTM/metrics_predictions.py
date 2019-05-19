__author__ = "Sandra Arcos Holzinger"
__copyright__ = "Copyright 2019, Sentiment Analysis-BERT"
__version__ = "1.0"
__email__ = "Sandra.ArcosHolzinger@thalesgroup.com.au"
__status__ = "Pre-Production"

#-------------------------------------------------------------PACKAGES-------------------------------------------------------#
from utils_LSTM import keras_predict
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


#-----scikit-learn packages---# 
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, StratifiedShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
    


#----------------------------------------------------GLOBAL VARIABLES---------------------------------------------------------#
SEED = 1 # do not modify - used for reproducible results
#----------------------------------------------------FUNCTIONS DEFINITIONS----------------------------------------------------#


def accuracy_neg_pos_test(predictions):
    """
    Computes classification accuracy: percentage of correctly classified positive and negative text (e.g tweets).
    
    Arguments:
    predictions -- Pandas DataFrame containing text (e.g. tweets), target values and predictions.
    
    Returns:
    pos_correct -- number of correctly labeled positive sentiments using the trained model.
    pos_cnt -- total number of positive sentiments in x_test.
    neg_correct -- number of correctly labeled negative sentiments using the trained model.
    neg_cnt -- total number of negative sentiments in x_test.
    """
    #Note: x_test and y_test here are in the form of transformed sequences
    pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0

    for idx,row in predictions.iterrows():
        if row['prediction'] == row['target'] :
            if row['prediction'] == 'neg':
                neg_correct += 1
            else:
                pos_correct += 1
        #count overall positive and negative sentiments in the validation set     
        if row['prediction'] == 'neg':
            neg_cnt += 1
        else:
            pos_cnt += 1
    return pos_cnt, neg_cnt, pos_correct, neg_correct


def auc_roc_test(predictions):
    """
    Computes AUC and ROC: evaluates how well the classifier can discriminate between classes 
    (in this case, two classes corresponding to positive and negative sentiments).
    
    Arguments:
    predictions -- Pandas DataFrame containing true labels and predictions.
    
    Returns:
    auc_keras -- a scalar with double precision float (float64) corresponding to AUC (area under the curve).
    fpr_keras -- numpy array containing false positive rates of predictions.
    tpr_keras -- numpy array containing true positive rates of predictions.
    """
    label_list = []
    pred_list = []
    for idx,row in predictions.iterrows():
        if row['target'] == 'neg':
            label_list.append(1)
        else:
            label_list.append(0)
        if row['prediction'] == 'neg':
            pred_list.append(1)
        else:
            pred_list.append(0)
            
    true_sentiments = np.asarray(label_list)    
    predicted_sentiments = np.asarray(pred_list)

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(true_sentiments, predicted_sentiments)
    auc_keras = auc(fpr_keras, tpr_keras) # calculate AUC option 1: using trapezoidal rule, given points on a curve.
#     auc_keras2 = roc_auc_score(y_t.values, y_pred_keras_max_array)# calculate AUC option 2: using predictions.
    return auc_keras,fpr_keras,tpr_keras,true_sentiments,predicted_sentiments


def plot_auc_roc_test(auc_keras, fpr_keras,tpr_keras):
    """
    Plots ROC: Receiver Operating Characteristics to visualise how well the classifier can discriminate between classes.
    
    Arguments:
    auc_keras -- scalar with double precision float (float64) corresponding to AUC (area under the curve).
    fpr_keras -- numpy array containing false positive rates of predictions. 
    tpr_keras -- numpy array containing true positive rates of predictions.
    
    Returns:
    plt.show() -- matplotlib function with plot.
    """
    plt.figure(figsize=(10,5))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(f'BERT_auc_roc_.png')
    return plt.show()

def plot_confusion_matrix_test(y_true, y_pred, classes,normalize,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    Prints and plots the confusion matrix.

    
    Arguments:
    y_true -- numpy array containing true target values from labeled dataset.
    y_pred -- numpy array containing predicted target labels.
    classes -- numpy array containing all target values in the dataset. 
    normalize -- normalization can be applied by setting 'normalize=True'.
    
    Returns:
    ax -- returns confusion matrix plot.
    
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    print(classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

