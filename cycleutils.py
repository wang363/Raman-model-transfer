import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from scipy import spatial
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

sigmoid_loss = nn.Sigmoid()


#归一化函数
def nor_max(arrlist):
    max_val = np.max(np.abs(arrlist) )
    #arrlist[arrlist<0] = 0
    arr_norm = arrlist / max_val
    return arr_norm

#余弦相似度比较函数
def similar_data(data_1,data_2):#做个相似度对比一下
    similar_score = 1 - spatial.distance.cosine(data_1, data_2)
    return similar_score

def train_test_dataloader(trainpath, testpath, BATCH_SIZE):
    # 放小拉曼的csv (5000,1301)
    xiaodata = pd.read_csv(trainpath, header= None)
    xiaodata = xiaodata.to_numpy()

    #行是种类，列是inputlength
    num_row, num_lie = xiaodata.shape

    # 放大拉曼的csv
    bigdata = pd.read_csv(testpath, header= None)
    bigdata = bigdata.to_numpy()

    up_limit = num_row//1000

    #定义两个列表，用来存放数据
    train_data = []
    test_data = []

    for i in range(up_limit):

        #train_name = 'train_' + str(i + 1)
        #test_name = 'test_' + str(i + 1)
        right_flag = 1000*(i+1)

        #这些数据同种数据有1000个，有5种因此形状是(5000,1301)
        train_name = xiaodata[1000*i:right_flag,:]#小拉曼的数据
        test_name = bigdata[1000*i:right_flag,:]

        train_data.append(train_name)
        test_data.append(test_name)

        #train_name = torch.tensor(train_name)
        #test_name = torch.tensor(test_name)

    #将列表组合起来
    train_data = torch.tensor(np.concatenate(train_data, axis=0))
    test_data = torch.tensor(np.concatenate(test_data, axis=0))

    dataset = TensorDataset(train_data, test_data)

    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_dataloader

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        # n_epochs必须大于decay_start_epoch， 否则会有如下异常提示
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        #计算出当前轮次下的衰减系数，用于更新模型的学习率
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

class ReplayBuffer():
    def __init__(self, max_size=130):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)
    

'''
target = torch.randint(low=-1, high=2, size=(4,2), dtype=torch.float32).float()
aaa = torch.ones_like(target)
bbb = torch.zeros_like(target)

def D_loss_real(real_data):
    randata = 0.1 * random.random()
    #loss = Cosine_loss(same_data, real_data)
    real_loss = cross_entropy_DA(real_data, torch.ones_like(real_data) * (0.949 + 0.25 * randata))
    return real_loss

def D_loss_fake(fake_data):
    randata = 0.1 * random.random()
    fake_loss = -1.0 * cross_entropy_DA(fake_data, torch.ones_like(fake_data) * (0.85 + 0.5 * randata))
    return fake_loss
'''

#定义源损失 交叉熵
cross_entropy = nn.BCELoss()
cross_entropy_DA = nn.BCELoss()
cross_entropy_DB = nn.BCELoss()
cross_sig_entropy = nn.BCEWithLogitsLoss()

#余弦相似度损失函数，用于判断输入的两个向量是否相似
GAN_Cosine_loss =  torch.nn.CosineEmbeddingLoss(margin=0.2)#设置阈值0.2，低于此数直接置0
Cosine_loss =  torch.nn.CosineEmbeddingLoss(margin=0.2)

#KL散度函数
kl_loss = torch.nn.KLDivLoss(reduction = 'batchmean')
#kl_loss = torch.nn.KLDivLoss()

def loss_identity(same_data, real_data):

    #loss = Cosine_loss(same_data, real_data)
    randata = 0.05 * random.random()
    kl_1 = torch.abs(kl_loss(same_data, (real_data+randata))) 

    sim_1 = torch.min(torch.cosine_similarity(same_data, real_data, dim=1)) 

    #iden_loss = torch.abs(0.5 - torch.sigmoid(kl_1.log())) + 1 - sim_1

    #iden_loss = 0.5 - torch.sigmoid(kl_1.log()) + 1 - sim_1
    iden_loss = torch.sigmoid(kl_1.log()) + sim_1

    return iden_loss

def Gan_loss(fake_data, real_data, fake_target):
    #target_0 = torch.ones_like(real_data)      # 1
    #target_1 = torch.zeros_like(real_data) - 1 #-1
    real_target = torch.ones_like(fake_target)
    sig_loss = cross_sig_entropy(fake_target, real_target)  

    real_target = torch.squeeze(real_target)#target需要一维数据,求余弦相似度
    loss = GAN_Cosine_loss(fake_data, real_data, real_target)
    #print(loss)
    #final_loss =torch.abs((sig_loss + loss)/2 -1) 
    #final_loss = 1 - (sig_loss + loss)/2 
    final_loss = sig_loss + loss

    return final_loss

def Cycle_loss(recover_data, real_data, fake_target):
    real_target = torch.ones_like(fake_target)
    #target_0 = torch.ones_like(real_data)      # 1
    #target_1 = torch.zeros_like(real_data) - 1 #-1
    sig_loss = cross_sig_entropy(fake_target, real_target)  

    real_target = torch.squeeze(real_target)#target需要一维数据
    loss = Cosine_loss(recover_data, real_data, real_target)
    #kl_1 = kl_loss(recover_data.log(), (real_data).log())

    #final_loss =torch.abs(0.3*sig_loss + 0.7*loss -1) + kl_1
    #final_loss = 2 - (sig_loss + loss) 
    final_loss = sig_loss + loss

    return final_loss

#target_real = Variable(Tensor(opt.batchSize,1).fill_(1.0), requires_grad=False)
#target_fake = Variable(Tensor(opt.batchSize,1).fill_(0.0), requires_grad=False)



def D_Aloss(fake_data, real_data):
    randata = 0.1 * random.random()
    #loss = Cosine_loss(same_data, real_data)
    real_loss = cross_entropy_DA(torch.ones_like(real_data) * (0.949 + 0.15 * randata), real_data)
    fake_loss = -cross_entropy_DA(torch.ones_like(fake_data) * (0.8 + 0.15 * randata), fake_data)   
    #fake_loss = -cross_entropy_DA(fake_data, torch.ones_like(fake_data))  
    #gen_fake_loss = cross_entropy(fake_output, torch.ones_like(fake_output) * -0.5)
    total_loss = torch.abs(real_loss + fake_loss ) 
    #total_loss = real_loss + fake_loss 
    #total_loss = torch.abs(0.5 - sigmoid_loss(real_loss + fake_loss))   
    #total_loss = torch.sigmoid(torch.sub(1, (torch.tanh(real_loss) - torch.tanh(fake_loss))/2) )
    return total_loss

def D_Bloss(fake_data, real_data):
    randata = 0.1 * random.random()
    #loss = Cosine_loss(same_data, real_data)
    real_loss = cross_entropy_DB(torch.ones_like(real_data) * (0.949 + 0.15 * randata), real_data)
    fake_loss = -cross_entropy_DB(torch.ones_like(fake_data) * (0.78 + 0.15 * randata), fake_data)
    #fake_loss = -cross_entropy_DB(fake_data, torch.ones_like(fake_data))  
    #fake_loss = -cross_entropy_DA(fake_data, torch.ones_like(fake_data) * randata)  
    #gen_fake_loss = cross_entropy(fake_output, torch.ones_like(fake_output) * -0.5)
    total_loss =  torch.abs(real_loss + fake_loss ) 
    #total_loss = real_loss + fake_loss 
    #total_loss = torch.abs(0.5 - sigmoid_loss(real_loss + fake_loss))  
    #total_loss = torch.sigmoid(torch.sub(1, (torch.tanh(real_loss) - torch.tanh(fake_loss))/2) ) 

    return total_loss

