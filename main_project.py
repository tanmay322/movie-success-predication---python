# NAME: TANMAY S. SHUKLA
# STUDENT ID: 001340336
import warnings

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

imdb_data = pd.read_csv("movie_metadata.csv") # loading and reading the data file
print "------Overlook of the dataset------------"
print imdb_data.head()  # imdb_data represents a data frame here
print "------no. of values and cols---------------"
print imdb_data.shape

# we are creating a column "movie_Success" which rate movies from 1 to 5
# based on the number of facebook likes for that movie
# this data shall be used as an input for the data mining techniques
def find_success(col):
    if col['movie_facebook_likes'] >= 0 and col['movie_facebook_likes'] < 20000:
        return 1
    elif col['movie_facebook_likes'] >= 20000 and col['movie_facebook_likes'] < 40000:
        return 2
    elif col['movie_facebook_likes'] >= 40000 and col['movie_facebook_likes'] < 60000:
        return 3
    elif col['movie_facebook_likes'] >= 60000 and col['movie_facebook_likes'] < 80000:
        return 4
    elif col['movie_facebook_likes'] >= 80000:
        return 5

imdb_data['movie_Success'] = imdb_data.apply(find_success, axis = 1)
modify_success = pd.DataFrame(imdb_data['movie_Success']) #adds a Success column based on the range of number of facebook likes
modify_success.to_csv("success.csv")

print "---------ADDING THE movie_SUCCESS COLUMN--------------"
print imdb_data.head()

# now filling the NaN with 0 value

def fill_NaN(col):
    imdb_data[col] = imdb_data[col].fillna(0)
colums = list(imdb_data.columns)
fill_NaN(colums)

print "--------MOVIE TITLES---------"

print imdb_data['movie_title']

print "---------DATA MINING TECHNIQUES-------------"

warnings.filterwarnings("ignore", category = FutureWarning)
print "1. Logistic Regression*******"
# collecting data for analyzing
X = imdb_data[imdb_data.columns[23:54]]
Y = imdb_data.iloc[:,-1]
X_new, Y_new = shuffle(X,Y)
#data for training and testing
X_train = X_new[:1000]
Y_train = Y_new[:1000]
X_test = X_new[1000:]
Y_test = Y_new[1000:]

# Training the model
#print X_train.head()
log = LogisticRegression()
log.fit(X_train, Y_train)
Y_predict = log.predict(X_test)

# Evaluating the model
#accuracy = accuracy_score(Y_test,Y_predict)
#precision = precision_score(Y_test,Y_predict, pos_label='positive', average='micro')
#recall = recall_score(Y_test,Y_predict, pos_label='positive', average='micro')


print("Training Accuracy is: ")
print(log.score(X_train, Y_train) * 100)
print(" Testing accuracy is: ")
print(log.score(X_test, Y_test) * 100)
#print("Precision is: ")
#print(precision)
#print("Recall is: ")
#print(recall)

print "-----------Random Forest Classifier----------------"

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
print X_train.shape
print Y_test.shape
rand_forest = RandomForestClassifier(random_state=0, n_estimators=250, min_samples_split=8, min_samples_leaf=4)
rand_forest.fit(X_train, Y_train)
Y_predict = rand_forest.predict(X_test)
confusion_mat_random = confusion_matrix(Y_test, Y_predict)
print ("Random forest accuracy: ")
print  accuracy_score(Y_test, Y_predict) * 100
print ('Confusion Matrix for Random forest: ')
print (pd.crosstab(Y_test, Y_predict, rownames=['Predicted Values'], colnames=['True Values']))


print "-------------SVM---------------------------------"


clf = svm.SVC()
y_pred = clf.fit(X_train, Y_train).predict(X_test)
accuracy = accuracy_score(Y_test,y_pred)

print('\nAccuracy of SVM: ')
print(accuracy)


print ('------------------Visualization------------------')
top_5_actors = imdb_data.actor_1_name[:5]
imdb_data.actor_1_name.value_counts()[:5].plot.pie(figsize = (6,6),autopct = '%1d%%')
plt.title('TOP 5 MOVIE ACTORS')

plt.show()
print ('Pie chart printed for viewing Top 5 actors from dataset')

print ('-------Histogram---------------')
imdb_data['genres'].value_counts()[:5].plot.bar(figsize=(10,6))
plt.ylabel('Number of successful movies')
plt.title('Top types of genres')
plt.show()
print ('Histogram plotted for top counrt of genres')











