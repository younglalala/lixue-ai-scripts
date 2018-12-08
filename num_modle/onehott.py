import numpy as np

# data=[1,1,0,1,1,1]
def one_hot(maxdat,data):
    max_x=maxdat
    data_onehot=[]
    for i in data:
        data_onehot1=[]
        for j in range(max_x+1):
            data_onehot1.append(0)
        data_onehot1[int(i)]=1
        data_onehot.append(data_onehot1)

    return np.array(data_onehot)
# print(one_hot(1,data))