# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:34:20 2021

@author: IntekPlus
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 11:42:06 2021

@author: IntekPlus
"""

import pandas

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn import model_selection


import matplotlib

import matplotlib.pyplot as plt

import numpy as np




names = ['farea','fSize','fSizeY','fSizeXYMean','fWidth','fLength','fWidthLengthMean','fAspectRatio','fAreaRatio','fLocalWidth','fKeyContrast','fAverageContrast','fDeviation','fEdgeEnergy','class']
url = "d:\\data.csv"
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:14]
Y = array[:,14]
test_size = 0.33
seed = 7
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)


gbc = GradientBoostingClassifier(random_state=0, max_depth=1, learning_rate=0.3) # 기본값: max_depth=3, learning_rate=0.1

gbc.fit(x_train, y_train)



score_train = gbc.score(x_train, y_train) # train set 정확도

print('{:.3f}'.format(score_train))

# 1.000

x_test2 =  []
for _ in range(8):
    line = []
    for _ in range(14):
        line.append(0)
    x_test2.append(line)
    
x_test2[0] = np.array([[42.280415	,	15.216	,	4.944	,	0.324921	,	4.95047	,	15.209797	,	10.080133	,	3.072394	,	56.152534	,	4.944008	,	47	,	21.153585	,	1.592056	,	0.979151 ]])
x_test2[1] = np.array([[49.275932	,	3.024	,	27.227999	,	9.003968	,	3.065476	,	27.226028	,	15.145752	,	8.8815	,	59.040852	,	3.024095	,	47	,	35.244656	,	2.240767	,	1.630034 ]])
x_test2[2] = np.array([[39.189167	,	2.16	,	29.4	,	13.611112	,	2.228437	,	29.398932	,	15.813684	,	13.192621	,	59.818298	,	2.160035	,	45	,	36.288597	,	1.794272	,	1.512073 ]])
x_test2[3] = np.array([[18.197998	,	6.684	,	12.216	,	1.827648	,	5.786795	,	12.779437	,	9.283115	,	2.208379	,	24.607857	,	3.3779	,	52	,	35.518242	,	2.155454	,	1.485998 ]])
x_test2[4] = np.array([[8.842896	,	3.012	,	7.224	,	2.398407	,	3.040362	,	7.161468	,	5.100915	,	2.355465	,	40.61319	,	2.987153	,	25	,	19.611408	,	1.188498	,	0.895325 ]])


#x_test2 = np.array([[    ]])
print("reject test")

for n in range(0, 5):
    labels = ['pass', 'reject']
    y_predict = gbc.predict(x_test2[n])
    
    
    y_predict = gbc.predict_proba(x_test2[n])
    confidence = y_predict[0][y_predict[0].argmax()]
    f1 = float(y_predict[0][0])
    f2 = float(y_predict[0][1])
    if(f1>f2):
       label = labels[0]
    else:
       label = labels[1]

    print( label,confidence) #





x_test2 =  []
for _ in range(8):
    line = []
    for _ in range(14):
        line.append(0)
    x_test2.append(line)
    

x_test2[0] = np.array([[0.246384	,	0.552	,	1.176	,	2.130435	,	0.544915	,	1.177494	,	0.861205	,	2.160877	,	38.399456	,	0.492586	,	60	,	49.587959	,	2.762414	,	1.789889	]])
x_test2[1] = np.array([[0.22104	,	0.708	,	0.696	,	0.983051	,	0.654469	,	0.755613	,	0.705041	,	1.154544	,	44.697414	,	0.607746	,	59	,	49.416939	,	2.726291	,	2.062541	]])
x_test2[2] = np.array([[0.286416	,	0.528	,	1.272	,	2.409091	,	0.540762	,	1.266351	,	0.903557	,	2.34179	,	41.825081	,	0.528479	,	62	,	49.614883	,	3.445839	,	2.852438	]])
x_test2[3] = np.array([[0.744768	,	1.176	,	1.116	,	0.94898	,	1.115576	,	1.128562	,	1.122069	,	1.011641	,	59.155655	,	1.030233	,	62	,	48.723125	,	3.331618	,	2.480085	]])
x_test2[4] = np.array([[0.22536	,	0.516	,	1.116	,	2.162791	,	0.526363	,	1.112285	,	0.819324	,	2.113152	,	38.492432	,	0.51651	,	67	,	49.624283	,	3.104883	,	2.707668	]])
x_test2[5] = np.array([[1.018224	,	1.26	,	1.152	,	0.914286	,	1.121332	,	1.286519	,	1.203926	,	1.147314	,	70.581802	,	1.088836	,	60	,	48.054588	,	2.740386	,	1.762834	]])
x_test2[6] = np.array([[0.256176	,	0.66	,	0.696	,	1.054546	,	0.664418	,	0.79766	,	0.731039	,	1.20054	,	48.336948	,	0.618803	,	59	,	49.730747	,	2.840925	,	2.155143	]])
x_test2[7] = np.array([[0.267408	,	0.744	,	0.612	,	0.822581	,	0.634348	,	0.754363	,	0.694355	,	1.189195	,	55.881325	,	0.567199	,	58	,	49.411415	,	2.772629	,	1.720248 ]])

#x_test2 = np.array([[    ]])
print("pass test")

for n in range(0, 8):
    labels = ['pass', 'reject']
    y_predict = gbc.predict(x_test2[n])
    
    
    y_predict = gbc.predict_proba(x_test2[n])
    confidence = y_predict[0][y_predict[0].argmax()]
    f1 = float(y_predict[0][0])
    f2 = float(y_predict[0][1])
    if(f1>f2):
       label = labels[0]
    else:
       label = labels[1]

    print( label,confidence) #
    







