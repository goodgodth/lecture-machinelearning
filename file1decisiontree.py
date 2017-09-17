# -*- coding: utf-8 -*-
from sklearn import tree
from sklearn.externals import joblib

tank_f = ["Energy(%)" , "Bullet" , "Soldier"]
ready_f = ["Front" , "Middle" , "Back"]


tank = [[100,10,5] , [65,7,3] , [80,2,5] , [50,10,3] , [45,9,2] , [75,8,5]]
ready = [0,1,2,1,2,0]




# dt = tree.DecisionTreeClassifier();
# dt = dt.fit(tank , ready)   #สั่งให้ computer  ได้ predict model
#
# # Save Model
# joblib.dump(dt , 'dt1Model.pkl')
#
# # Load Model
dt = joblib.load('dt1Model.pkl')
#
e , b , s = 75 , 4 , 5;
print("{}: {}, {}: {} , {}: {} = {}".format(tank_f[0] , e , tank_f[1] , b , tank_f[2] , s , ready_f[dt.predict([[e,b,s]])[0]]));
#
# print(dt.predict([[e,b,s]]));