import numpy as np
import math
import random
import sklearn.preprocessing


nodes = 65608366
dim = 100
iii = np.random.choice(np.arange(dim),nodes%dim,replace=False)
idxx = list(np.arange(dim))*int(nodes/dim)+list(iii)
np.random.shuffle(idxx)
feat = np.eye(dim)[idxx]
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(feat)
feat = scaler.transform(feat)
print(feat.shape)
step = int(feat.shape[0]/10)+1
feat1 = feat[:step]
feat2 = feat[step:step*2]
feat3 = feat[step*2:step*3]
feat4 = feat[step*3:step*4]
feat5 = feat[step*4:step*5]
feat6 = feat[step*5:step*6]
feat7 = feat[step*6:step*7]
feat8 = feat[step*7:step*8]
feat9 = feat[step*8:step*9]
feat10 = feat[step*9:]
np.save('friendster_feat1.npy',feat1)
np.save('friendster_feat2.npy',feat2)
np.save('friendster_feat3.npy',feat3)
np.save('friendster_feat4.npy',feat4)
np.save('friendster_feat5.npy',feat5)
np.save('friendster_feat6.npy',feat6)
np.save('friendster_feat7.npy',feat7)
np.save('friendster_feat8.npy',feat8)
np.save('friendster_feat9.npy',feat9)
np.save('friendster_feat10.npy',feat10)