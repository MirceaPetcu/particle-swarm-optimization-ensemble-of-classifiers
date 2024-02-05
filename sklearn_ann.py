from sklearn.neural_network import MLPClassifier   
from sklearn.ensemble import RandomForestClassifier 
import pandas as pd 
from sklearn import metrics
import warnings 
warnings.filterwarnings('ignore')
df_train = pd.read_csv('diabetes_train.csv')
df_test = pd.read_csv('diabetes_test.csv')
df_val = pd.read_csv('diabetes_val.csv')

df_test = pd.concat([df_test,df_val])

x_train,y_train = df_train.drop(columns=['Outcome']),df_train['Outcome']
x_test,y_test = df_test.drop(columns=['Outcome']),df_test['Outcome']

epochs = 100
clf = MLPClassifier(hidden_layer_sizes=(30,30),activation='relu',solver='sgd',max_iter=epochs)
clf.fit(x_train,y_train)
with open('ann_performance.txt','a') as f:
    f.write('Sklearn ANN\n')
    f.write(f'2 hidden layers with 30 neurons each, relu activation function and trained for {epochs} epcochs\n')
    f.write(f'Train score: {clf.score(x_train,y_train)}\n')
    f.write(f'Test score: {clf.score(x_test,y_test)}\n')
    f.write(f'Confusion matrix: {metrics.confusion_matrix(y_test,clf.predict(x_test))}\n')
    f.write('--------------------------------\n')


