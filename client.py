import pickle as pkl
import numpy as np
import pandas as p
import json
#import these because sometimes not importing them does shit
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
#validate loaded shit
namecol=None
with open('namescol.txt') as fp:
    namecol=[a.replace('\n','') for a in fp.readlines()]
#print(namecol)
df=p.read_csv('spambase.data',names=namecol)
#print(df)

#print(df[col])
#print(df['class'])


df_train,df_test= train_test_split(df,train_size=.70, random_state=999)
#load model
testm=None
with open('eldummy.pkl','rb') as fp:
    testm=pkl.load(fp)
    col=df.columns.difference(['class'])
    y_pred=testm.predict(df_test[col])
    print('correct classification')
    print(np.mean(y_pred==df_test['class']))
#handler 
def predict(event,context):
    data=json.loads(event['body'])
    r=testm.predict([data['row']])
    return {'spam':r[0]}
# fake test
if __name__ == '__main__':
    print('original value',df_test.iloc[0,-1].tolist())
    res=predict({'row':df_test[col].iloc[0,:].tolist()},None)
    with open('example.json','w') as fp:
        json.dump({'row':df_test[col].iloc[0,:].tolist()},fp)
    print(res)
