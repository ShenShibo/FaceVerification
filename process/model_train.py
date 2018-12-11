import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import pickle
import torch.utils.data as data
import torchvision


class NaiveNet(nn.Module):
    # input size 148*148, ouput 10575 classes
    def __init__(self):
        super(NaiveNet, self).__init__()
        pass
    def forward(self, x):
        pass


def accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels.data).sum()
    return correct


def validate(net, loader, use_cuda=False):
    correct_count = 0.
    count = 0.
    if use_cuda:
        net = net.cuda()
    for i, (b_x, b_y) in enumerate(loader, 0):
        size = b_x.shape[0]
        b_x = Variable(b_x)
        b_y = Variable(b_y)
        if use_cuda:
            b_x = b_x.cuda()
            b_y = b_y.cuda()
        outputs = net(b_x)
        c = accuracy(outputs, b_y)
        correct_count += c
        count += size
    acc = correct_count.item() / float(count)
    return acc


def train():
    lr = 0.01
    batch_size = 128
    use_cuda = True
    epochs = 50
    if torch.cuda.is_available() is False:
        use_cuda = False
    # 这里改成任意的网络
    net = NaiveNet()
    if use_cuda:
        net = net.cuda()
    # 定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.00001)
    # 定义loss函数
    loss_function = nn.CrossEntropyLoss()
    # 训练集
    train_path = '../data/train'
    train_set = torchvision.datasets.ImageFolder(train_path)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # 学习速率调节器
    lr_scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
    # 测试数据集
    validating_path = '../data/test'
    validating_set = torchvision.datasets.ImageFolder(validating_path)
    validation_loader = DataLoader(validating_set, batch_size=batch_size)
    # 开始训练
    loss_save = []
    tacc_save = []
    vacc_save = []
    for epoch in range(epochs):
        lr_scheduler.step()
        running_loss = 0.0
        correct_count = 0.
        count = 0
        for i, (b_x, b_y) in enumerate(train_loader):
            size = b_x.shape[0]
            b_x = Variable(b_x)
            b_y = Variable(b_y)
            if use_cuda:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            # 前向
            outputs = net(b_x)
            # 计算误差
            loss = loss_function(outputs, b_y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 参数更新
            optimizer.step()
            # 计算loss
            running_loss += loss.item()
            count += size
            correct_count += accuracy(outputs, b_y).item()
            if (i + 1) % 10 == 0:
                acc = validate(net, validation_loader, True)
                print('[ %d-%d ] loss: %.9f, \n'
                      'training accuracy: %.6f, \n'
                      'validating accuracy: %.6f' % (
                      epoch + 1, i + 1, running_loss / count, correct_count / count, acc))
                tacc_save.append(correct_count / count)
                loss_save.append(running_loss / count)
                vacc_save.append(acc)
        if (epoch + 1) % 5 == 0:
            print("save")
            torch.save(net.state_dict(), '../model/face_verification_net{}.p'.format(epoch + 1))
    dic = {}
    dic['loss'] = loss_save
    dic['training_accuracy'] = tacc_save
    dic['validating_accuracy'] = vacc_save
    with open('../model/record.p', 'wb') as f:
        pickle.dump(dic, f)






