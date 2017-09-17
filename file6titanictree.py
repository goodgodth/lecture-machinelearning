import pandas as pd
from sklearn import tree

titanic = pd.read_csv("titanic.csv" , header=0)

titanic = titanic[pd.notnull(titanic["Age"])]
print(titanic.shape)

features = titanic[["Age" , "SexCode"]].values
labels = titanic["Survived"].values


#Train Data 70%
train_f = features[:round(len(features)*0.7)]
train_l = labels[:round(len(labels)*0.7)]

#Test Data 30%
test_f = features[round(len(features)*0.7)+1:]
test_l = labels[round(len(labels)*0.7)+1:]

dt = tree.DecisionTreeClassifier()

#Learn
dt.fit(train_f, train_l)



ans = []
_true = 0

#Predict
for i in range(len(test_l)):
    p = dt.predict([test_f[i]])
    l = test_l[i]

    ans.append("Predict: {}\tLabel: {}".format(p,l))

    if p == l:
        _true += 1

# print(len(ans));
# print(_true);

for i in range(len(ans)):
    print (ans[i])

print("Accurately: {} %".format(round(_true*100/len(ans) , 2)));

