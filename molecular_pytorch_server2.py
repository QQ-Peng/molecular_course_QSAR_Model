# TODO：保存预测值和真实值，画图，计算MRSE

import pandas
import pandas as pd
import math
import numpy as np
from rdkit import Chem
from rdkit import DataStructs

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
device = 'cuda'

extract = False
add_C50 = False
train = False
_torch = False
use_smile = True
feature = 'descriptors'
# feature = 'morgan'
split_n = 10
load_train = True
use_torch = False
feature_selection = True
feature_num = 169

import joblib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

class My_dataset(Dataset):
    def __init__(self, data_x, data_y):
        super().__init__()
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.data_x.shape[0]


class Lr(nn.Module):
    def __init__(self, nt_index=1):
        super(Lr, self).__init__()  # 继承父类init的参数
        self.nt_dict = {
            0:
                nn.Sequential(
                    nn.Linear(2048, 1024),
                    nn.Sigmoid(),
                    nn.Linear(1024, 512),
                    nn.Sigmoid(),
                    nn.Linear(512, 216),
                    nn.Sigmoid(),
                    nn.Linear(216, 1)
                ),
            1:
                nn.Sequential(
                    nn.Linear(206, 103),
                    nn.Sigmoid(),
                    nn.Linear(103, 52),
                    nn.Sigmoid(),
                    nn.Linear(52, 26),
                    nn.Sigmoid(),
                    nn.Linear(26, 1)
                ),
            2:
                nn.Sequential(
                    nn.Linear(206, 103),
                    nn.Sigmoid(),
                    nn.Linear(103, 1)
                )
        }
        self.nt = self.nt_dict[nt_index]

    def forward(self, x):
        out = self.nt(x)
        return out

    def predict(self, x):
        return self.nt(x)

def mol2image(smiles):
    train_x = []
    for i in smiles:
        mol = Chem.MolFromSmiles(i)
        fp = Chem.RDKFingerprint(mol, maxPath=4, fpSize=2048)
        res = np.zeros(len(fp))
        DataStructs.ConvertToNumpyArray(fp, res)
        train_x.append(res)
    return np.array(train_x)


if train:
    if use_smile:
        if not load_train:
            # read and shuffle
            pre_data = pandas.read_excel("./data/dataset_filter.xlsx")
            pre_data = shuffle(pre_data)
            # split data for training and shuffle
            pre_data_train, pre_data_test, _, _ = train_test_split(pre_data, [0]*pre_data.shape[0], random_state=33, train_size=0.9)
            pre_data_train.to_excel('./data/dataset_filter_train.xlsx', index=False)
            pre_data_test.to_excel('./data/dataset_filter_test.xlsx', index=False)
        else:
            pre_data_train = pandas.read_excel("./data/dataset_filter_train.xlsx")
            pre_data_test = pandas.read_excel("./data/dataset_filter_test.xlsx")

        # generate topological features for smiles
        X_train_smile = mol2image(pre_data_train.Smiles)
        X_test_smile = mol2image(pre_data_test.Smiles)
        Y_train = np.array(pre_data_train.pC50)
        Y_test = np.array(pre_data_test.pC50)
        # read 2-dimension descriptors
        columns = list(pre_data_train.columns)
        X_train_2d = np.array(pre_data_train[columns[1:-4]])
        X_test_2d = np.array(pre_data_test[columns[1:-4]])
        if feature == 'descriptors':
            print("use 2d-descriptors")
            X_train = X_train_2d
            X_test = X_test_2d
        else:
            print("use morgan fingerprint")
            X_train = X_train_smile
            X_test = X_test_smile
        if not use_torch:
            # train
            print("use sklearn LR")
            # 交叉验证集
            split = KFold(n_splits=split_n, shuffle=True)
            folds = list(split.split(X_train, Y_train))
            cross_val_x = []
            cross_val_y = []
            for fold in folds:
                cross_val_x.append(X_train[fold[1]])
                cross_val_y.append(Y_train[fold[1]])
            model_base = LR()
            max_score = -1000000
            if feature == 'descriptors':
                save_path = './checkpoint/model_lr_2d_best.pkl'
            else:
                save_path = './checkpoint/model_rfr_mfpt_best.pkl'
            if feature_selection:
                print("use feature selection.")
                save_path = './checkpoint/model_lr_2d_feature_selection_best.pkl'
                model = RFE(model_base, n_features_to_select=feature_num)
            for i in range(split_n):
                train_x = np.concatenate(cross_val_x[:i] + cross_val_x[(i + 1):])
                val_x = cross_val_x[i]
                train_y = np.concatenate(cross_val_y[:i] + cross_val_y[(i + 1):])
                val_y = cross_val_y[i]
                model.fit(train_x, train_y)
                score = model.score(val_x, val_y)
                print("*"*20)
                print("validation score: ", score)
                if max_score < score:
                    max_score = score
                    joblib.dump(model, save_path)
                print("test score: ", model.score(X_test, Y_test))

            print("max R2 score: ",max_score)
            feature_rank = sorted(zip(map(lambda x: round(x, 4), model.ranking_), columns))
            with open("./checkpoint/selected_features_lr.txt",'w',newline='\n',encoding='utf-8') as f:
                for j in range(feature_num):
                    print(feature_rank[j][1])
                    f.write(feature_rank[j][1]+'\n')

        else:
            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, random_state=133, train_size=0.9)

            X_train = torch.tensor(X_train).float().to(device)
            X_test = torch.tensor(X_test).float().to(device)
            X_val = torch.tensor(X_val).float().to(device)
            Y_train = torch.tensor(Y_train).float().to(device)
            Y_test = torch.tensor(Y_test).float().to(device)
            Y_val = torch.tensor(Y_val).float().to(device)

            lr_model = Lr(0).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(lr_model.parameters(), lr=1e-3)
            data_train = My_dataset(X_train, Y_train)
            # 3. 训练模型
            all_r_squared = []
            max_r_squared = -1000000
            for i in range(3000):
                lr_model.train()
                data_loader_train = DataLoader(data_train, batch_size=128, shuffle=True)
                for i_batch, batch_data in enumerate(data_loader_train):
                    out = lr_model(batch_data[0])  # 3.1 获取预测值
                    # print(out.shape)
                    # print(batch_data[1].shape)
                    loss = criterion(batch_data[1].reshape(-1, 1), out.reshape(-1, 1))  # 3.2 计算损失
                    optimizer.zero_grad()  # 3.3 梯度归零
                    loss.backward()  # 3.4 计算梯度
                    optimizer.step()  # 3.5 更新梯度
                if (i + 1) % 10 == 0:
                    # print('Epoch[{}/{}], loss: {:.6f}'.format(i, 30000, loss.data))

                    with torch.no_grad():
                        lr_model.eval()
                        Y_pred = lr_model.predict(X_val).to('cpu').numpy()
                        R_squared = 1 - (mean_squared_error(Y_val.to('cpu').numpy(), Y_pred) / np.var(
                            Y_val.to('cpu').numpy()))
                        print('Epoch[{}/{}], R_Squared: {:.6f}'.format(i, 3000, R_squared))
                        all_r_squared.append(R_squared)
                        if R_squared > max_r_squared:
                            torch.save(lr_model.state_dict(), './checkpoint/model_torch_nolr_mfpt_best.pt')
                            max_r_squared = R_squared.data
                            print("R-Squared: ", R_squared)
                            print("Loss: ", loss.data)


            # print(min(all_r_squared))
            print("max R-squared: ", max(all_r_squared))
            # print(sum(all_r_squared) / len(all_r_squared))
else:
    _plot = True
    _compute = False
    if _plot:

        import matplotlib.pyplot as plt
        pre_data_test = pandas.read_excel("./data/dataset_filter_test.xlsx")
        X_test_smile = mol2image(pre_data_test.Smiles)
        Y_test = np.array(pre_data_test.pC50)
        columns = list(pre_data_test.columns)
        X_test_2d = np.array(pre_data_test[columns[1:-4]])
        model = joblib.load('./checkpoint/model_rfr_mfpt_best.pkl')
        # model2 = joblib.load('./checkpoint/model_lr_2d_best.pkl')
        with torch.no_grad():
            model2 = Lr(0)
            model2.load_state_dict(torch.load('./checkpoint/model_torch_nolr_mfpt_best.pt'))
            model2.eval()
            Y_pred2 = model2(torch.tensor(X_test_smile).float()).view(-1).numpy()

            # print(Y_pred)

        Y_pred = model.predict(X_test_smile)
        # Y_pred2 = model2.predict(X_test_smile)
        #
        # R_squared = 1 - (mean_squared_error(Y_test, Y_pred) / np.var(
        #     Y_test))
        # print(R_squared)

        # with open('./prediction_result/lr_2d_feature_selection.txt', 'w', newline='\n') as f:
        #     f.write("pred\ttrue\n")
        #     for i in range(len(Y_test)):
        #         f.write(str(Y_pred[i])+'\t'+str(Y_test[i])+'\n')
        plt.scatter(Y_pred, Y_test, c="#DC143C", alpha=0.4,label="TPFP RFR")
        plt.scatter(Y_pred2, Y_test, c="#7B68EE", alpha=0.4,label="TPFP NN")
        plt.xlabel("predicted value",fontsize=14)
        plt.ylabel("true value",fontsize=14)
        plt.title("TPFP-QSAR Model")


        plt.legend()


        plt.savefig(r'./png/mfpt_embed.png', dpi=300)
    elif _compute:
        file = open('./prediction_result/lr_2d_feature_selection.txt', 'r')
        file.readline()
        pred = []
        gold = []
        for line in file:
            line = line.strip()
            line = line.split('\t')
            pred.append(float(line[0]))
            gold.append(float(line[1]))
        RMSE = np.sqrt(mean_squared_error(np.array(pred), np.array(gold)))
        MAE = mean_absolute_error(np.array(pred), np.array(gold))
        print("RMSE: ",RMSE)
        print("MAE: ",MAE)
exit()




if extract:
    target_list = open('D:/坚果云/我的坚果云/课程文件/分子模拟/JAK2_CHEMBLID.txt', 'r').read().split('\n')
    # print(target_list)
    target_hash = {}
    n = 0
    for i in target_list:
        target_hash[i] = n
        n += 1

    record_one = ''
    target_result = []
    out = open('D:/坚果云/我的坚果云/课程文件/分子模拟/JAK2_target.mol2', 'w', encoding='utf-8', newline='\n')
    n = 0
    for line in open('D:/Data/分子模拟/chembl_29.mol2', 'r', encoding='utf-8'):
        if line.startswith('@<TRIPOS>MOLECULE'):

            # print(record_one)
            if record_one == '':
                continue
            n += 1
            if n % 10000 == 0:
                print(f"process {n}")
            record_one = record_one.split('\n')
            if target_hash.get(record_one[1]) is not None:
                out.write('\n'.join(record_one))
            record_one = line
        else:
            record_one += line
    out.close()

if add_C50:
    dataset = pd.read_excel('./data/dataset.xlsx')
    source_data = pd.read_excel('./data/JAK2.xls')

    single_c50_idlist = []
    id_list = list(source_data['Molecule ChEMBL ID'])
    for i in id_list:
        if id_list.count(i) == 1:
            single_c50_idlist.append(i)
    source_data_filter = source_data[
        (source_data['Molecule ChEMBL ID'].isin(single_c50_idlist)) & (source_data['Standard Relation'] == "'='") & (
                    source_data['Assay Organism'] == "Homo sapiens") & (~source_data['pChEMBL Value'].isnull())]
    target_ChEMBLID = list(source_data_filter['Molecule ChEMBL ID'])
    dataset_train = dataset[dataset['chembl_id'].isin(target_ChEMBLID)]
    ID_C50_PChEMBL = source_data_filter[['Molecule ChEMBL ID', 'Standard Value', 'pChEMBL Value', 'Smiles']]
    ID_C50_PChEMBL = ID_C50_PChEMBL.sort_values(by='Molecule ChEMBL ID')

    dataset_train = dataset_train.sort_values(by='chembl_id')
    ID_C50_PChEMBL = ID_C50_PChEMBL.reset_index()
    dataset_train = dataset_train.reset_index()
    # ID_C50_idlist = list(ID_C50['Molecule ChEMBL ID'])
    # dataset_train = list(dataset_train['chembl_id'])
    dataset_train['Smiles'] = ID_C50_PChEMBL['Smiles']
    dataset_train['C50'] = ID_C50_PChEMBL['Standard Value']
    dataset_train['pChEMBL Value'] = ID_C50_PChEMBL['pChEMBL Value']
    f = lambda x: -math.log(x)
    dataset_train['pC50'] = dataset_train['C50'].apply(f)

    dataset_train.to_excel('./data/dataset_filter.xlsx', index=False)
