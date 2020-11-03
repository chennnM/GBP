import numpy as np

n = 65608366
m = 1806067135


u = list(range(m))
v = list(range(m))
idx = 0
file_ = 'friendster.txt'
print('start load')
with open(file_) as f:
    for l in f:
        x, y = l.strip().split()[:2]
        u[idx] = int(x)
        v[idx] = int(y)
        idx+=1

print(idx)
edgelist = np.array([u,v])

# np.save('friendster.npy',edgelist)


feat = edgelist.T
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
np.save('friendster1.npy',feat1)
np.save('friendster2.npy',feat2)
np.save('friendster3.npy',feat3)
np.save('friendster4.npy',feat4)
np.save('friendster5.npy',feat5)
np.save('friendster6.npy',feat6)
np.save('friendster7.npy',feat7)
np.save('friendster8.npy',feat8)
np.save('friendster9.npy',feat9)
np.save('friendster10.npy',feat10)