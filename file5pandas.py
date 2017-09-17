import pandas as pd

titanic = pd.read_csv("titanic.csv" , header=0)
# print(titanic.shape)
#print(titanic);
#print(titanic["Survived"]);

titanic = titanic[pd.notnull(titanic["Age"])]
print(titanic.shape)

features = titanic[["Age" , "SexCode"]].values
labels = titanic["Survived"].values

print(features)
print(labels)