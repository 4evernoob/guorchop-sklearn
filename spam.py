import numpy as np
import pandas as p
import pickle as pkl
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
if __name__ == '__main__':

    omg = GaussianNB()
    namecol=None
    with open('namescol.txt') as fp:
        namecol=[a.replace('\n','') for a in fp.readlines()]
    #print(namecol)
    df=p.read_csv('spambase.data',names=namecol)
    #print(df)

    #print(df[col])
    #print(df['class'])


    df_train,df_test= train_test_split(df,train_size=.70, random_state=999)
    print(df_train[df_train['class']==0].mean()-df_train[df_train['class']==1].mean())
    #print(df_train[df_train['class']==1].mean())
    fig,ax = plt.subplots(3,figsize=(15,12))
    df_train[df_train['class']==0]['capital_run_length_total'].plot.hist(bins=300,ax=ax[0],alpha=.5)
    df_train[df_train['class']==1]['capital_run_length_total'].plot.hist(bins=300,ax=ax[0],alpha=.5)

    df_train[df_train['class']==0]['capital_run_length_longest'].plot.hist(bins=300,ax=ax[1],alpha=.5)
    df_train[df_train['class']==1]['capital_run_length_longest'].plot.hist(bins=300,ax=ax[1],alpha=.5)

    df_train[df_train['class']==0]['capital_run_length_average'].plot.hist(bins=300,ax=ax[2],alpha=.5)
    df_train[df_train['class']==1]['capital_run_length_average'].plot.hist(bins=300,ax=ax[2],alpha=.5)
    plt.show()
    #input('continue to dummy model')
    col=df.columns.difference(['class'])
    print('col len ',len(col))
    omg.fit(df_train[col],df_train['class'])
    y_pred=omg.predict(df_test[col])

    print('correct classification')
    print(np.mean(y_pred==df_test['class']))
    print('delete some cols')
    mm=df.corr()
    im=plt.matshow(mm)
    plt.colorbar(im)
    plt.show()
    #df['kaput']=df['capital_run_length_total']/df['capital_run_length_longest']
    df_train,df_test= train_test_split(df,train_size=.70, random_state=999)
    #print(df)
    print(df_train[df_train['class']==0].mean()-df_train[df_train['class']==1].mean())
    #kk=['capital_run_length_total','capital_run_length_longest','capital_run_length_average']
    col=df.columns.difference(['class'])
    #col=kk#df.columns.difference(['class']+col)
    print('col len ',len(col))
    print(col)
    omg.fit(df_train[col],df_train['class'])

    y_pred=omg.predict(df_test[col])
    print('correct classification')
    print(np.mean(y_pred==df_test['class']))
    print('no hay mucho que hacer por que no hay tiempo pero')
    print('chequemos un par de parametros y un esquema de voto sencillo')
    omg2=KNeighborsClassifier(n_neighbors=3)#(n_neighbors=3)
    omg3=tree.DecisionTreeClassifier(random_state=333)#(max_depth=2,min_samples_split=6,random_state=333)
    jepl=VotingClassifier(estimators=[('bai', omg), ('bezinos', omg2), ('arbol', omg3)],voting='hard')
    jepl.fit(df_train[col],df_train['class'])
    y_pred=jepl.predict(df_test[col])
    print("muchas veces no sabes si el problema en cuestion puede mejorar jhaciendole muchas cosas")
    print("alguna veces ni con ensambles mejora")
    print("aqui se insertan innumerables try y catch")

    print('correct classification')
    print(np.mean(y_pred==df_test['class']))
    with open('eldummy.pkl','wb') as fp:
        pkl.dump(jepl,fp)
    print("test model load")
    with open('eldummy.pkl','rb') as fp:
        testm=pkl.load(fp)
        y_pred=testm.predict(df_test[col])
        print('correct classification')
        print(np.mean(y_pred==df_test['class']))

