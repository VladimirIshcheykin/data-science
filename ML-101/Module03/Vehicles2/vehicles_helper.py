import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')


def attributes_counts(dataset):
    """ Basic descriptions of the target attribute.
    
    This module displays 'Class' attribute value counts 
    and visualisation plot of the dataset.
     
    Parameters:
        dataset.
    """
    print("'Class' Value Counts: "+" \n", dataset['Class'].value_counts())
    print("\n Visualisation plot: "+" \n", dataset['Class'].value_counts().plot(x = dataset['Class'], kind='bar'))


def all_attrubutes_vizual(dataset, one, two, three):
    """ Visual presentation of all attributes in the dataset.
    
   This module shows all attributes divided into 3 parts
   for better visualization.
   
    Parameters:
        dataset, 
        one: selected attributes for part one
        two: selected attributes for part two 
        three: selected attributes for part three
    """
    print("Part one:"+"\n")
    df1 = dataset[one] 
    sns.pairplot(df1, kind="scatter",  hue="Class", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
    plt.show()
    print("\n")
    print("Part two:"+"\n")
    df2 = dataset[two] 
    sns.pairplot(df2, kind="scatter",  hue="Class", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
    plt.show()
    print("\n")
    print("Part three:"+"\n")
    df3 = dataset[three] 
    sns.pairplot(df3, kind="scatter",  hue="Class", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
    plt.show()


def corr_plot_list(dataset): 
    """ This module presents correlation plot and list of each attribute"""
    
    print("'Correlation list of each attribute: ")
    corr = dataset.corr()
    corr_abs = corr.abs()
    num_cols = len(dataset)
    num_corr = corr_abs.nlargest(num_cols, 'Class')['Class']
    print(num_corr)
    print("\n")
    print("'Correlation plot of each attribute: "+"\n", dataset.corr()['Class'].sort_values().plot(kind='bar', figsize=(18, 6)))


def standart_scaler(dataset): 
    """ createstandart scaler data and displaying """
    sc = StandardScaler()
    dataset_sc = sc.fit_transform(dataset)
    dataset_sc = pd.DataFrame(dataset_sc)
    print("StandardScaler:\n")
    dataset_sc.head()
    return dataset_sc

def use_svc(x_train, y_train, x_test, y_test):
    """ applying SVC model, getting predictions and displaying accuracy """
    classifier = SVC(kernel='linear', decision_function_shape='ovo', tol=0.03)
    classifier.fit(x_train, y_train)
    SVC_pred = classifier.predict(x_test)
    accuracy = accuracy_score(SVC_pred, y_test)
    print("Accuracy:", accuracy * 100, "\n")
    return SVC_pred

def show_confusion_matrix(y, pred, name_model):
    """ getting confusion matrix and displaying it with misclassification plot """
    labels = ["bus", "opel", "saab", "van"]
    cm = confusion_matrix(y, pred, labels=None)
    print(name_model, " Confusion Matrix\n")

    # visualisation of matrix
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    sns.heatmap(df_cm, annot=True, fmt="d", cmap = 'Reds', linewidths = 0.5, annot_kws = {'size': 15})
    plt.ylabel("Actual Vehicle Class")
    plt.xlabel("Predicted Vehicle Class")    
    plt.show()
    
    print("Misclassification plot\n")
    
    # misclassification vehicle plot 
    for label in df_cm.columns:
        df_cm.at[label, label] = 0

    ax = df_cm.plot(kind="bar", title="Misclassified Vehicle Classes")
    ax.set_xlabel("Vehicle Classes")
    ax.set_ylabel("Number of Incorrectly Predicted Class")    
    plt.show()    

def save_data(index, pred):
    """" saving data predicted data to .csv file """
    sub = pd.DataFrame()
    sub['ID'] = index
    sub['Class'] = pred
    sub.to_csv('ShortVehiclesPredictionsTest.csv', index=False)
    