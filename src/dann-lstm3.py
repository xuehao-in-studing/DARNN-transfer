import torch
import torch.nn as nn
import numpy as np
import time
import math
from functions import ReverseLayerF
import matplotlib.pyplot as plt

torch.manual_seed(100)
np.random.seed(0)

# This concept is also called teacher forceing. 
# The flag decides if the loss will be calculted over all 
# or just the predicted values.
input_window = 10
output_window = 1
batch_size = 24  # batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = ("cpu")


class LSTMnetwork1(nn.Module):
    def __init__(self, input_size=5, hidden_size=100, num_layers=1, output_size=7):
        super(LSTMnetwork1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        # 定义LSTM层
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=False)
        # 定义全连接层
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # 初始化h0，c0

    def forward(self, seq):
        batch_size, seq_len = seq.shape[1], seq.shape[0]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # input(batch_size, seq_len, input_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(seq, (h_0, c_0))
        pred = self.linear(output)

        return pred  # 输出只用取最后一个值

class LSTMnetwork2(nn.Module):
    def __init__(self, input_size=7, hidden_size=100, num_layers=1, output_size=7):
        super(LSTMnetwork2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        # 定义LSTM层
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=False)
        # 定义全连接层
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # 初始化h0，c0

    def forward(self, seq):
        batch_size, seq_len = seq.shape[1], seq.shape[0]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # input(batch_size, seq_len, input_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(seq, (h_0, c_0))
        pred = self.linear(output)

        return pred  # 输出只用取最后一个值

class dann_LSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=1, n_outputs=1):
        super(dann_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.n_outputs = n_outputs
        # self.fcs = [nn.Linear(self.hidden_size, self.output_size).to(device) for i in range(self.n_outputs)]
        self.relu = nn.ReLU(inplace=True)

        self.feature = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=False)

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(64, 1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_flatten', nn.Flatten())
        self.domain_classifier.add_module('d_fc1', nn.Linear(640, 100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_seq, alpha):
        # batch_size, seq_len = input_seq.shape[1], input_seq.shape[0]
        # h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # input(batch_size, seq_len, input_size)
        # output(batch_size, seq_len, num_directions *

        feature, _ = self.feature(input_seq)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        reverse_feature=reverse_feature.permute(1, 0, 2)
        pred = self.class_classifier(feature)

        domain_output = self.domain_classifier(reverse_feature)
        return pred[-1], domain_output



import pandas as pd
from datetime import date

df_source = pd.read_csv('西线.csv', header=0, index_col=0)
df_target = pd.read_csv('东线2.csv', header=0, index_col=0)
df_source = df_source.values.astype(float)
df_target = df_target.values.astype(float)


# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
def create_inout_sequences(input_data, inputwindow):
    inout_seq = []
    L = len(input_data)
    for i in range(L - inputwindow):
        train_seq = input_data[i:i + inputwindow]
        train_label = input_data[i + output_window:i + inputwindow + output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)


def get_source_data(data):
    # time        = np.arange(0, 400, 0.1)
    # amplitude   = np.sin(time) + np.sin(time*0.05) +np.sin(time*0.12) *np.random.normal(-0.2, 0.2, len(time))
    # series = read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # print('a',series.shape)
    amplitude = scaler.fit_transform(data)
    # print('b', amplitude.shape)
    # amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
    # print(amplitude.shape)
    # print(train_data.shape,test_data.shape)
    # convert our train data into a pytorch train tensor
    # train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment.. 
    # print('c',train_data.shape)
    train_sequence = create_inout_sequences(amplitude, input_window)
    # print('a',train_sequence.size())
    # test_data = torch.FloatTensor(test_data).view(-1)
    return train_sequence.to(device), scaler


def get_target_data(data, sampels):
    # time        = np.arange(0, 400, 0.1)
    # amplitude   = np.sin(time) + np.sin(time*0.05) +np.sin(time*0.12) *np.random.normal(-0.2, 0.2, len(time))
    # series = read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_prediction = MinMaxScaler(feature_range=(0, 1))
    # print('a',series.shape)
    amplitude = scaler.fit_transform(data)
    train_norm_prediction = scaler_prediction.fit_transform(data[:, 0].reshape(-1, 1))

    amplitude = create_inout_sequences(amplitude, input_window)
    # print('b', amplitude.shape)
    # amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
    # print(amplitude.shape)
    train_data = amplitude[:sampels]
    test_data = amplitude[sampels:]
    # print(train_data.shape,test_data.shape)
    # convert our train data into a pytorch train tensor
    # train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment..
    # print('c',train_data.shape)
    return train_data.to(device), test_data.to(device), scaler, scaler_prediction


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source)  - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1)).squeeze()  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1)).squeeze()
    return input, target


train_source_data, source_scaler = get_source_data(df_source)
train_target_data, test_data, target_scaler, scaler_prediction= get_target_data(df_target, 96)


# print(train_data.shape)
# print(val_data.shape)
# print(train_data.size(), val_data.size())
# tr,te = get_batch(train_data, 0,batch_size)
def train(source_data, target_data, epoch):
    model1.train()
    model2.train()
    model3.train()# Turn on the train mode
    total_loss = 0.

    start_time = time.time()

    for batch, i in enumerate(range(0, len(source_data), batch_size)):
        data_source, data_source_label = get_batch(source_data, i, batch_size)
        data_target, data_target_label = get_batch(target_data, i, batch_size)
        p = float(epoch / epochs)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        optimizer.zero_grad()
        domain_label = torch.zeros(batch_size).long()

        lstmout = model1(data_source)
        class_output, domain_output = model2(lstmout, alpha)

        err_s_label = criterion(class_output, data_source_label[:,:,[0]][-1])
        err_s_domain = loss_domain(domain_output, domain_label)


        domain_label = torch.ones(batch_size).long()
        lstmout2 = model3(data_target)
        class_output, domain_output = model2(lstmout2, alpha)
        err_t_label = criterion(class_output, data_target_label[:,:,[0]][-1])
        err_t_domain = loss_domain(domain_output, domain_label)
        loss = 0.5*err_t_domain + 0.5*err_s_domain + 0.1*err_s_label + 0.9*err_t_label
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model1.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(model3.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(source_data) / batch_size/1)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                epoch, batch, len(source_data) // batch_size, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# entweder ist hier ein fehler im loss oder in der train methode, aber die ergebnisse sind unterschiedlich
# auch zu denen der predict_future
def evaluate(eval_model1, eval_model2,eval_model3, data_source, alpha, scaler):
    eval_model1.eval()
    eval_model2.eval()
    eval_model3.eval()
    total_loss = 0.
    MSE = 0.
    MAE = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    pred, pred_true = get_batch(data_source, 0, 1)
    with torch.no_grad():
        for i in range(0, len(data_source) ):
            data, target = get_batch(data_source, i, 1)
            data = data.unsqueeze(1)
            target = target.unsqueeze(1)
            # look like the model returns static values for the output window
            data2=eval_model3(data)
            output, _ = eval_model2(data2,alpha)
            true_predictions = scaler.inverse_transform(np.array(output).reshape(-1, 1))
            # 逆归一化还原真实值
           #test = torch.tensor(target)
            test=target[:,:,[0]][-1]
            Mse = criterion(output, test)
            Mae = np.abs(output - test)
            test=scaler.inverse_transform(test)
            true_predictions = torch.tensor(true_predictions)

            # loss_Mse= criterion(true_predictions, test)
            average_accuracy = 1 - np.abs(np.array(test) - np.array(true_predictions)) / np.array(test)
            total_loss += average_accuracy.item()
            MSE += Mse.item()
            MAE += Mae.item()
    return total_loss / 5,MSE/ 5,MAE/5

def plot(eval_model1, eval_model2,eval_model3, data_source, alpha, scaler):
    eval_model1.eval()
    eval_model2.eval()
    eval_model3.eval()
    total_loss = 0.
    MSE = 0.
    MAE = 0.
    test_result = []
    truth = []
    pred, pred_true = get_batch(data_source, 0, 1)
    with torch.no_grad():
        for i in range(0, len(data_source) ):
            data, target = get_batch(data_source, i, 1)
            data = data.unsqueeze(1)
            target = target.unsqueeze(1)
            # look like the model returns static values for the output window
            data2=eval_model3(data)
            output, _ = eval_model2(data2,alpha)
            true_predictions = scaler.inverse_transform(np.array(output).reshape(-1, 1))
            # 逆归一化还原真实值
           #test = torch.tensor(target)
            test=target[:,:,[0]][-1]
            Mse = criterion(output, test)
            Mae = np.abs(output - test)
            test=scaler.inverse_transform(test)
            truth.append(test)
            true_predictions = torch.tensor(true_predictions)
            test_result.append(true_predictions)

            # loss_Mse= criterion(true_predictions, test)
            average_accuracy = 1 - np.abs(np.array(test) - np.array(true_predictions)) / np.array(test)
            total_loss += average_accuracy.item()
            MSE += Mse.item()
            MAE += Mae.item()
    ht = np.array(test_result)
    ht = ht.reshape(-1, 1)
    np.savetxt("yuce.csv", ht, delimiter=",")
    hb = np.array(truth)
    hb = hb.reshape(-1, 1)
    np.savetxt("shiji.csv", hb, delimiter=",")

    yuce = pd.read_csv("yuce.csv", header=None)
    shiji = pd.read_csv("shiji.csv", header=None)  # 读取csv数据'''
    plt.figure(figsize=(12, 4))
    plt.rcParams['font.sans-serif'] = ['Simhei']
    x = [107,108,109,110,111]
    line1, = plt.plot(x, yuce, 'r')
    line2, = plt.plot(x, shiji, 'b')
    plt.legend([line1, line2], ["predict", "truth"], loc="upper right")
    plt.xlabel('环号')
    plt.ylabel('刀盘转矩')
    plt.xticks([107, 108, 109,110,111])
    plt.show()


    return total_loss / 5,MSE/ 5,MAE/5



model1 = LSTMnetwork1().to(device)
model2 = dann_LSTM().to(device)
model3 = LSTMnetwork2().to(device)


criterion = nn.MSELoss()
loss_domain = torch.nn.NLLLoss()
lr = 0.005

from itertools import chain

# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(params=chain(model1.parameters(),
                                           model2.parameters(),model3.parameters()), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

best_val_loss = float("inf")
epochs = 100  # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_source_data, train_target_data, epoch)
    if (epoch % 10 is 0):
        accuaracy, MSE, MAE = plot(model1, model2, model3,test_data,0,scaler_prediction)
        # predict_future(model, val_data,200,epoch,scaler)
    else:
        accuaracy, MSE, MAE = evaluate(model1, model2, model3,test_data,0,scaler_prediction)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s |  accuaracy {:5.5f} |MSE {:5.5f} |MAE {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
                time.time() - epoch_start_time),
                                                                                                  accuaracy,MSE,MAE, math.exp(accuaracy)))
    print('-' * 89)

    # if val_loss < best_val_loss:
    #    best_val_loss = val_loss
    #    best_model = model

    scheduler.step()
