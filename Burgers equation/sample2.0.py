'''
@author: XiaotingHan
https://blog.csdn.net/qq_24211837/article/details/124402998
'''

import torch
import torch.nn as nn
import torch
import torch.nn as nn
import numpy as np
import random
import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from matplotlib import cm
from early_stopping import EarlyStopping
from dataset import get_dataloader
from dataset import get_testloader
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# 模型搭建
class Net(nn.Module):
    def __init__(self, NN): # NL n个l（线性，全连接）隐藏层， NN 输入数据的维数， 128 256
        # NL是有多少层隐藏层
        # NN是每层的神经元数量
        super(Net, self).__init__()

        self.input_layer = nn.Linear(2, NN)
        self.hidden_layer1 = nn.Linear(NN,int(NN/2)) ## 原文这里用NN，我这里用的下采样，经过实验验证，“等采样”更优
        self.hidden_layer2 = nn.Linear(int(NN/2), int(NN/2))  ## 原文这里用NN，我这里用的下采样，经过实验验证，“等采样”更优
        self.output_layer = nn.Linear(int(NN / 2), 1)

    def forward(self, x): # 一种特殊的方法 __call__() 回调
        out = torch.tanh(self.input_layer(x))
        out = torch.tanh(self.hidden_layer1(out))
        out = torch.tanh(self.hidden_layer2(out))
        out_final = self.output_layer(out)
        return out_final

def pde(x, net):
    u = net(x)  # 网络得到的数据
    u_tx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(net(x)),
                               create_graph=True, allow_unused=True)[0]  # 求偏导数
    d_t = u_tx[:, 0].unsqueeze(-1)
    d_x = u_tx[:, 1].unsqueeze(-1)
    u_xx = torch.autograd.grad(d_x, x, grad_outputs=torch.ones_like(d_x),
                               create_graph=True, allow_unused=True)[0][:,1].unsqueeze(-1)  # 求偏导数

    w = torch.tensor(0.01 / np.pi)

    return d_t + u * d_x - w * u_xx  # 公式（1）

def setup():
    seed = 1445
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def draw_loss(Loss_list,epoch):
    # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
    plt.cla()
    x1 = range(1, epoch+1)
    y1 = Loss_list
    # print(y1)
    plt.title('Train loss vs. epoches', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoches', fontsize=20)
    plt.ylabel('Train loss', fontsize=20)
    plt.grid()
    plt.savefig("burgers_trainloss.png")
    plt.show()

def train():
    #setup()
    net = Net(30)
    iterations = 250
    dataloader = get_dataloader()
    testloader = get_testloader()
    mse_cost_function = torch.nn.MSELoss(reduction='mean')  # Mean squared error
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    train_losses = []

    eval_losses = []

    save_path = ".\\"
    early_stopping = EarlyStopping(save_path)

    for epoch in range(iterations):
        train_loss = 0
        train_acc = 0
        net.train()

        for index, pt_x_bc_var, pt_t_bc_zeros, pt_u_bc_sin, pt_x_in_pos_one, pt_x_in_neg_one, pt_t_in_var, \
           pt_u_in_zeros, pt_x_collocation, pt_t_collocation, pt_all_zeros in tqdm.tqdm(dataloader):
            optimizer.zero_grad()  # 梯度归0

            net_bc_out = net(torch.cat([pt_t_bc_zeros, pt_x_bc_var], 1))  # u(x,t)的输出
            mse_u_2 = mse_cost_function(net_bc_out, pt_u_bc_sin)  # e = u(x,t)-(-sin(pi*x))  公式（2）

            net_bc_inr = net(torch.cat([pt_t_in_var, pt_x_in_pos_one], 1))  # 0=u(t,1) 公式（3)
            net_bc_inl = net(torch.cat([pt_t_in_var, pt_x_in_neg_one], 1))  # 0=u(t,-1) 公式（4）

            mse_u_3 = mse_cost_function(net_bc_inr, pt_u_in_zeros)  # e = 0-u(t,1) 公式(3)
            mse_u_4 = mse_cost_function(net_bc_inl, pt_u_in_zeros)  # e = 0-u(t,-1) 公式（4）

            # 求PDE函数式的误差
            # 将变量x,t带入公式（1）
            f_out = pde(torch.cat([pt_t_collocation, pt_x_collocation], 1), net)  # output of f(x,t) 公式（1）
            mse_f_1 = mse_cost_function(f_out, pt_all_zeros)

            # 将误差(损失)累加起来
            loss = mse_f_1 + mse_u_2 + mse_u_3 + mse_u_4

            loss.backward()  # 反向传播
            optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

            train_loss += loss.item()

        train_losses.append(train_loss/2000)

        eval_loss = 0
        net.eval()
        for index, pt_x_bc_var, pt_t_bc_zeros, pt_u_bc_sin, pt_x_in_pos_one, pt_x_in_neg_one, pt_t_in_var, \
           pt_u_in_zeros, pt_x_collocation, pt_t_collocation, pt_all_zeros in tqdm.tqdm(testloader):
            net_bc_out = net(torch.cat([pt_t_bc_zeros, pt_x_bc_var], 1))  # u(x,t)的输出
            mse_u_2 = mse_cost_function(net_bc_out, pt_u_bc_sin)  # e = u(x,t)-(-sin(pi*x))  公式（2）

            net_bc_inr = net(torch.cat([pt_t_in_var, pt_x_in_pos_one], 1))  # 0=u(t,1) 公式（3)
            net_bc_inl = net(torch.cat([pt_t_in_var, pt_x_in_neg_one], 1))  # 0=u(t,-1) 公式（4）

            mse_u_3 = mse_cost_function(net_bc_inr, pt_u_in_zeros)  # e = 0-u(t,1) 公式(3)
            mse_u_4 = mse_cost_function(net_bc_inl, pt_u_in_zeros)  # e = 0-u(t,-1) 公式（4）

            # 求PDE函数式的误差
            # 将变量x,t带入公式（1）
            f_out = pde(torch.cat([pt_t_collocation, pt_x_collocation], 1), net)  # output of f(x,t) 公式（1）
            mse_f_1 = mse_cost_function(f_out, pt_all_zeros)

            # 将误差(损失)累加起来
            loss = mse_f_1 + mse_u_2 + mse_u_3 + mse_u_4

            eval_loss += loss.item()

        eval_losses.append(eval_loss/2000)

        print('epoch: {}, Train Loss: {:.6f}, Eval Loss: {:.6f}'
              .format(epoch, train_loss / 2000, eval_loss / 2000,))
        # 早停止
        early_stopping(eval_loss, net)
        # 达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练

        with torch.autograd.no_grad():
            if epoch % 100 == 0:
                print(epoch, "Traning Loss:", loss.data)
    draw_loss(train_losses, iterations)

    ## 画图 ##
    t = np.linspace(0, 1, 100)
    x = np.linspace(-1, 1, 256)
    ms_t, ms_x = np.meshgrid(t, x)
    x = np.ravel(ms_x).reshape(-1, 1)
    t = np.ravel(ms_t).reshape(-1, 1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True)
    pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True)
    pt_u0 = net(torch.cat([pt_t, pt_x], 1))
    u = pt_u0.data.cpu().numpy()

    ti = np.zeros(25600)
    ti = np.ravel(ti).reshape(-1, 1)
    ti = Variable(torch.from_numpy(ti).float(), requires_grad=True)
    pt_ui = net(torch.cat([ti, pt_x], 1))
    ui = pt_ui.data.cpu().numpy()
    u_real_i = -np.sin(np.pi * x)
    mse_i = mean_absolute_error(ui, u_real_i)

    x0 = -np.ones(25600)
    x0 = np.ravel(x0).reshape(-1, 1)
    x0 = Variable(torch.from_numpy(x0).float(), requires_grad=True)
    x1 = np.ones(25600)
    x1 = np.ravel(x1).reshape(-1, 1)
    x1 = Variable(torch.from_numpy(x1).float(), requires_grad=True)
    pt_ub0 = net(torch.cat([pt_t, x0], 1))
    pt_ub1 = net(torch.cat([pt_t, x1], 1))
    ub0 = pt_ub0.data.cpu().numpy()
    ub1 = pt_ub1.data.cpu().numpy()
    u_real_b = np.zeros(25600)
    mse_b0 = mean_absolute_error(ub0, u_real_b)
    mse_b1 = mean_absolute_error(ub1, u_real_b)
    print(mse_i)
    print(mse_b0)
    print(mse_b1)

    pt_u0 = u.reshape(256, 100)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_zlim([-1, 1])
    ax.plot_surface(ms_t, ms_x, pt_u0, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    plt.savefig('burgers5.png')
    plt.close(fig)


if __name__ == "__main__":
    train()

