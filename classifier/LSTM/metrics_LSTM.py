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
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report, log_loss, accuracy_score
from sklearn.utils.multiclass import unique_labels
    


#----------------------------------------------------GLOBAL VARIABLES---------------------------------------------------------#
SEED = 1 # do not modify - used for reproducible results
#----------------------------------------------------FUNCTIONS DEFINITIONS----------------------------------------------------#

def accuracy_neg_pos(text_sequences, target_indicator, model, max_len):
    """
    Computes classification accuracy: percentage of correctly classified positive and negative text (e.g tweets).
    
    Arguments:
    text_sequences -- numpy array containing text (e.g. tweets) in the form of sequences.
    target_indicator -- numpy array containing target labels (e.g. 'positive'or 'negative') in the form of a dummy/indicator variable.
    
    Returns:
    pos_correct -- number of correctly labeled positive sentiments using the trained model.
    pos_cnt -- total number of positive sentiments in x_test.
    neg_correct -- number of correctly labeled negative sentiments using the trained model.
    neg_cnt -- total number of negative sentiments in x_test.
    """

    pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
    predictions = keras_predict(text_sequences,'all', model, max_len) #returns logits
    for x in range(len(text_sequences)):       
    #compare indices of max values along predictions and y_validate[x]
        predicted_sent_index = np.argmax(predictions[x])
        expected_sent_index = np.argmax(target_indicator[x])
        if predicted_sent_index == expected_sent_index:
            if expected_sent_index == 1:
                neg_correct += 1
            else:
                pos_correct += 1
    #count overall positive and negative sentiments in the validation set     
        if expected_sent_index == 1:
            neg_cnt += 1
        else:
            pos_cnt += 1
    return pos_correct, pos_cnt, neg_correct, neg_cnt


def accuracy_neg_pos_scikit(text_sequences, target_indicator, predictions):
    """
    Computes classification accuracy: percentage of correctly classified positive and negative text (e.g tweets).
    
    Arguments:
    text_sequences -- numpy array containing text (e.g. tweets) in the form of sequences.
    target_indicator -- numpy array containing target labels (e.g. 'positive'or 'negative') in the form of a dummy/indicator variable.
    
    Returns:
    pos_correct -- number of correctly labeled positive sentiments using the trained model.
    pos_cnt -- total number of positive sentiments in x_test.
    neg_correct -- number of correctly labeled negative sentiments using the trained model.
    neg_cnt -- total number of negative sentiments in x_test.
    """

    pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
    for x in range(len(predictions)):       
    #compare indices of max values along predictions and y_validate[x]
        predicted_sent_index = np.argmax(predictions[x])
        expected_sent_index = np.argmax(target_indicator[x])
        if predicted_sent_index == expected_sent_index:
            if expected_sent_index == 1:
                neg_correct += 1
            else:
                pos_correct += 1
    #count overall positive and negative sentiments in the validation set     
        if expected_sent_index == 1:
            neg_cnt += 1
        else:
            pos_cnt += 1
    return pos_correct, pos_cnt, neg_correct, neg_cnt

def auc_roc(x_test,y_test, model, max_len):
    
    """
    Computes AUC and ROC: evaluates how well the classifier can discriminate between classes 
    (in this case, two classes corresponding to positive and negative sentiments).
    
    Arguments:
    x_test -- numpy array containing text (e.g. tweets) from the test set.
    y_test -- numpy array containing target labels (e.g. 'positive'or 'negative') from the test set.
    
    Returns:
    auc_keras -- A scalar with double precision float (float64) corresponding to AUC (area under the curve).
    fpr_keras -- numpy array containing false positive rates of predictions.
    tpr_keras -- numpy array containing true positive rates of predictions.
    """
#     all_predictions=keras_predict(x_test, 'all', model, max_len)
    all_predictions=keras_predict(x_test, 'all', model, max_len)[:,1]
    y_pred_keras_max = []

    for index in range(len(all_predictions)):
        #get the predictions for positive or negative from a list of probabilities
        y_pred_keras_max.append(all_predictions[index])#in parentheses = returns 1 or 0 for [0=pos,1=neg]

    #-----get the labelled data from y_test. The expression in parentheses (below) finds the values where y_test!=0 and
    #returns 1 for true, 0 for false. This can be translated to false = positive sentiment.-----#
    y_t = pd.Series(np.where(y_test!=0)[1])
    y_pred_keras_max_array = np.asarray(y_pred_keras_max)#convert to array as required by roc_curve function
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_t.values, y_pred_keras_max_array)
    auc_keras = auc(fpr_keras, tpr_keras) # calculate AUC option 1: using trapezoidal rule, given points on a curve.
    auc_keras2 = roc_auc_score(y_t.values, y_pred_keras_max_array)# calculate AUC option 2: using predictions.
    return auc_keras,fpr_keras,tpr_keras

def plot_auc_roc(auc_keras, fpr_keras,tpr_keras,dataset, samples, batch_size, max_len,epochs):
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
#     plt.subplot(1,2,1)
#     plt.subplots_adjust(wspace=.05, hspace=1)
    plt.plot([0, 1], [0, 1], 'k--')
#     plt.plot(fpr_keras, tpr_keras, label='LSTM (area = {:.3f})'.format(auc_keras))
    plt.plot(fpr_keras, tpr_keras, label='LSTM (AUC = %0.3f)' % (auc_keras))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    ##############################################################################
    plt.savefig(f'/home/hummus/algorithms-other/LSTM/figures/auc_roc/auc_roc_{dataset}_{samples}_batchsize{batch_size}_maxlen{max_len}_epochs{epochs}.png')
    return plt.show()

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
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
