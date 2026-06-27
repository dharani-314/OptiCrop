"""crop_recommendation_pipeline.py
ML pipeline for OptiCrop (assigned tasks only).
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def load_dataset(path="Crop_recommendation.csv"):
    df=pd.read_csv(path)
    print("Dataset Loaded Successfully")
    print(df.head())
    return df

def seasonal_analysis(df):
    print("\n========== SEASONAL CROPS ==========")
    print("\nSummer crops")
    print(df[(df["temperature"]>30)&(df["humidity"]>50)]["label"].unique())
    print("-"*60)
    print("Winter crops")
    print(df[(df["temperature"]<20)&(df["humidity"]>30)]["label"].unique())
    print("-"*60)
    print("Rainy crops")
    print(df[(df["rainfall"]>200)&(df["humidity"]>50)]["label"].unique())

def split_dataset(df):
    X=df.drop("label",axis=1)
    y=df["label"]
    print("Shape of X :",X.shape)
    print("Shape of y :",y.shape)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
    return X,X_train,X_test,y_train,y_test

def perform_kmeans(df):
    X=df.drop("label",axis=1)
    wcss=[]
    for i in range(1,11):
        km=KMeans(n_clusters=i,init="k-means++",max_iter=300,n_init=10,random_state=0)
        km.fit(X)
        wcss.append(km.inertia_)
    plt.plot(range(1,11),wcss,marker="o")
    plt.title("The Elbow Method")
    plt.show()
    km=KMeans(n_clusters=4,init="k-means++",max_iter=300,n_init=10,random_state=0)
    result=df.copy()
    result["cluster"]=km.fit_predict(X)
    return km

def train_logistic_regression(X_train,y_train):
    model=LogisticRegression(max_iter=1000)
    model.fit(X_train,y_train)
    return model

def evaluate_model(model,X_test,y_test):
    y_pred=model.predict(X_test)
    print("Accuracy :",accuracy_score(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))

def save_model(model,filename="model.pkl"):
    with open(filename,"wb") as f:
        pickle.dump(model,f)

def load_model(filename="model.pkl"):
    with open(filename,"rb") as f:
        return pickle.load(f)

def predict_crop(model,N,P,K,temperature,humidity,ph,rainfall):
    pred=model.predict([[N,P,K,temperature,humidity,ph,rainfall]])
    print("Recommended Crop :",pred[0])
    return pred[0]

def main():
    df=load_dataset()
    seasonal_analysis(df)
    _,X_train,X_test,y_train,y_test=split_dataset(df)
    perform_kmeans(df)
    model=train_logistic_regression(X_train,y_train)
    evaluate_model(model,X_test,y_test)
    save_model(model)
    predict_crop(model,90,42,43,20.8,82,6.5,202.9)

if __name__=="__main__":
    main()
