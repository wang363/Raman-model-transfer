# encoding: utf-8
from cyclemodels import *
from cycleutils import *
# from function import pull_baseline
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

#归一化函数
def nor_max(arrlist):
    max_val = np.max(np.abs(arrlist) )
    min_val = np.min(np.abs(arrlist) )
    #arrlist[arrlist<0] = 0
    arr_norm = (arrlist-min_val) / (max_val-min_val)
    return arr_norm

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cpu'
#device = 'cuda'
pic_path = '/test_A2B_pic/'
GAN_path = 'generated_data/' # save_generated_data

#测试模型生成数据
model = torch.load('A2B_model.pt', map_location=device)
model.eval()
#model.train()
'''
for i, (aa, bb) in enumerate(train_dataloader):
    print(aa.shape)
    newdata = model(aa.to(torch.float32))
    asd = pd.DataFrame(newdata.detach().numpy())
    asd.to_csv('testcycle.csv',header=False,index=False)
    break
'''
###############################################################################
#模型测试
datacsv = pd.read_csv('val.csv')
xdata = np.arange(200,2001)#x轴

#小拉曼原始数据
val_csv = pd.read_csv('val.csv')
csv_column = val_csv.columns.tolist()

# 设立三个旗帜变量
set_flag = 0
num_epoch = 0
flag = 0
for col in csv_column:
    gan_csv = pd.DataFrame()
    if num_epoch%4 ==0:# 每种选了四条数据做验证
        set_flag = 100 * flag  # 每种数据100个，因此可以从整数部分开始，就是一个新的数据
        flag = flag + 1
    

    ran_data = random.randint(0, 20) # 弄个随机数
    set_flag = set_flag + ran_data

    #大拉曼原始数据
    test_csv = pd.read_csv('test.csv',header=None) #
    noise = np.random.rand(1801,)# 加噪声
    test_data_csv = np.array(test_csv.iloc[:,set_flag]) + noise
    #train_data_csv = pull_baseline(xdata, nor_max(train_csv.iloc[:,0])) 

    val_data_csv = val_csv[col]#用于测试的小拉曼数据
    #data = datacsv.iloc[6,:]#用于测试的小拉曼数据
    #print(xiao_4atp.shape)
    #GAN网络生成数据
    gendata = np.array(val_data_csv, dtype=np.float32) 
    #gendata = datacsv.iloc[0,:]
    #gendata = pull_baseline(xdata,gendata)#预处理GAN网络生成数据

    gendata =np.reshape(gendata,(1,1801)) 
    gendata = torch.as_tensor(gendata, dtype=torch.float32,device=device)
    new_gendata = model(gendata)

    new_gendata =np.squeeze(new_gendata.detach().numpy()) #先将tensor转化为np，然后压缩多余的1维度
    #new_gendata = new_gendata# 
    gan_csv.loc[:,0] =  xdata 
    gan_csv.loc[:,1] = new_gendata # 将生成数据保存
    floder_name = max(col.split('_'), key=len) 
    if not os.path.exists(GAN_path + floder_name):
        os.mkdir(GAN_path + floder_name)
    save_gan_path = GAN_path + floder_name
    gan_csv.to_csv(save_gan_path + os.sep + col + '.csv', header=False, index=False)
    #data = datacsv.iloc[0,:]

    sim_gen_xiao = similar_data(new_gendata,val_data_csv)
    print(col + '|||' + 'GAN生成数据与小拉曼-余弦相似度为:',sim_gen_xiao)

    sim_gen_big = similar_data(new_gendata,test_data_csv)
    print(col + '*-->*' + 'targrt:GAN生成数据与大拉曼-余弦相似度为:',sim_gen_big.round(5))

    sim_xiao_big = similar_data(val_data_csv,test_data_csv)
    print(col + '|||' + '小拉曼数据与大拉曼-余弦相似度为:',sim_xiao_big)


    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.suptitle('GAN--Target-cosine similarity:' + str(sim_gen_big.round(4)),x=0.625,y=0.98,fontweight='bold',fontsize=14)
    plt.subplot(411)
    #plt.plot(new_4atp_gendata, 'y', label = 'GAN')
    plt.plot(xdata, val_data_csv, 'r', label = 'Master')
    plt.title('Master--Target-cosine similarity:' + str(sim_xiao_big.round(4)),loc='right',fontweight='bold',fontsize=14)
    plt.legend(prop={'size':12},loc ='upper right')
    

    plt.subplot(412)
    #plt.plot(new_4atp_gendata, 'y', label = 'GAN')
    plt.plot(xdata, test_data_csv, 'black', label = 'Target')
    plt.legend(prop={'size':13},loc ='upper right')

    plt.subplot(413)
    plt.plot(xdata, nor_max(new_gendata), 'y', label = 'GAN')
    #plt.plot(data, 'r', label = 'testdata')
    plt.legend(prop={'size':13},loc ='upper right')

    plt.subplot(414)
    labels = ['Master', 'Target', 'GAN']
    colors = ['r','black','yellow']
    plt.plot(xdata,nor_max(val_data_csv) , 'r', label = 'Master')
    plt.plot(xdata, nor_max(test_data_csv), 'black', label = 'Target')
    plt.plot(xdata, nor_max(new_gendata), 'y', label = 'GAN')

    #plt.plot(data, 'r', label = 'testdata')
    plt.xlabel('Raman_shift')
    #plt.ylabel('Intensity')
    plt.legend(prop={'size':13},loc ='lower center',bbox_to_anchor=(0.5,0.5),ncol=3,frameon=False) # ncol=3，3个并排,frameon=False不显示框线
    plt.show()
    # plt.savefig(pic_path + str(col) + '_test.png')
    plt.clf()

    num_epoch += 1
