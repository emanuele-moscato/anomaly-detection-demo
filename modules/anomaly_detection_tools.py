import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go

init_notebook_mode(connected=True)

DATA_DIR = '../data/'

def read_training_data():
    features_names = [
        'existing_account_status',
        'duration_months',
        'credit_history',
        'purpose',
        'credit_amount',
        'savings_account_bonds',
        'present_employment_since',
        'installment_rate_percent',
        'personal_status_gender',
        'other_debtors_guarantors',
        'present_residence_since',
        'property',
        'age_years',
        'other_installment_plans',
        'housing',
        'n_credits',
        'job',
        'n_people_maintenance',
        'telephone',
        'foreign_worker'
    ]
    data_df = pd.read_csv(DATA_DIR+'german.adcg.tr', sep=' ', header=None)
    data_df.columns = features_names
    labels_df = pd.read_csv(DATA_DIR+'german.adcg.tr.label', sep=',')
    data_df['label'] = labels_df['Prediction']
    
    return data_df
    
def filter_numerical_only(data_df):
    numerical_columns = [
        'duration_months',
        'credit_amount',
        'installment_rate_percent',
        'present_residence_since',
        'age_years',
        'n_credits',
        'n_people_maintenance'
    ]
    numerical_data_df = data_df[numerical_columns+['label']]
    
    return numerical_data_df
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot2d(x, labels):
    """
    Plots 2-dimensional data x coloring it with the labels y.
    """
    trace0 = go.Scatter(
        x = x[labels[:,0]==0][:,0],
        y = x[labels[:,0]==0][:,1],
        mode = 'markers',
        name = 'class 0',
        hoverinfo = 'all'
    )
    
    trace1 = go.Scatter(
        x = x[labels[:,0]==1][:,0],
        y = x[labels[:,0]==1][:,1],
        mode = 'markers',
        name = 'class 1',
        hoverinfo = 'all'
    )
    
    layout=go.Layout(
        hovermode = 'closest'
    )
    
    data = [trace0, trace1]
    
    fig = go.Figure(data=data, layout=layout)
    
    iplot(fig)

def plot3d(x, labels):
    """
    Plots 3-dimensional data x coloring it with the labels y.
    """
    trace0 = go.Scatter3d(
        x = x[labels[:,0]==0][:,0],
        y = x[labels[:,0]==0][:,1],
        z = x[labels[:,0]==0][:,2],
        mode = 'markers',
        marker = dict(
            size=5
        )
    )
    
    trace1 = go.Scatter3d(
        x = x[labels[:,0]==1][:,0],
        y = x[labels[:,0]==1][:,1],
        z = x[labels[:,0]==1][:,2],
        mode = 'markers',
        marker = dict(
            size=5
        )
    )
    
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    
    data = [trace0, trace1]
    
    fig = go.Figure(data=data, layout=layout)
    
    iplot(fig)