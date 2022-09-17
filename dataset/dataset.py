import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


def make_traindata():
    t_bc_zeros = np.zeros((2000, 1))
    x_in_pos_one = np.ones((2000, 1))
    x_in_neg_one = -np.ones((2000, 1))
    u_in_zeros = np.zeros((2000, 1))
    x_collocation = np.random.uniform(low=-1.0, high=1.0, size=(2000, 1))
    t_collocation = np.random.uniform(low=0.0, high=1.0, size=(2000, 1))
    all_zeros = np.zeros((2000, 1))
    t_in_var = np.random.uniform(low=0, high=1.0, size=(2000, 1))
    x_bc_var = np.random.uniform(low=-1.0, high=1.0, size=(2000, 1))
    u_bc_sin = -np.sin(np.pi * x_bc_var)

    # 将数据转化为torch可用的
    pt_x_bc_var = Variable(torch.from_numpy(x_bc_var).float(), requires_grad=False)
    pt_t_bc_zeros = Variable(torch.from_numpy(t_bc_zeros).float(), requires_grad=False)
    pt_u_bc_sin = Variable(torch.from_numpy(u_bc_sin).float(), requires_grad=False)
    pt_x_in_pos_one = Variable(torch.from_numpy(x_in_pos_one).float(), requires_grad=False)
    pt_x_in_neg_one = Variable(torch.from_numpy(x_in_neg_one).float(), requires_grad=False)
    pt_t_in_var = Variable(torch.from_numpy(t_in_var).float(), requires_grad=False)
    pt_u_in_zeros = Variable(torch.from_numpy(u_in_zeros).float(), requires_grad=False)
    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True)
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False)

    #print(torch.Tensor(pt_x_bc_var))
    return pt_x_bc_var, pt_t_bc_zeros, pt_u_bc_sin, pt_x_in_pos_one, pt_x_in_neg_one, pt_t_in_var, \
           pt_u_in_zeros, pt_x_collocation, pt_t_collocation, pt_all_zeros


class MyTrainDataset(Dataset):
    def __init__(self):
        super(MyTrainDataset, self).__init__()
        self.pt_x_bc_var, self.pt_t_bc_zeros, self.pt_u_bc_sin, self.pt_x_in_pos_one, \
        self.pt_x_in_neg_one, self.pt_t_in_var, self.pt_u_in_zeros, self.pt_x_collocation, \
        self.pt_t_collocation, self.pt_all_zeros = make_traindata()

    def __iter__(self):
        # return list(self.pt_x_bc_var), list(self.pt_t_bc_zeros), list(self.pt_u_bc_sin), list(self.pt_x_in_pos_one), \
        #        list(self.pt_x_in_neg_one), list(self.pt_t_in_var), list(self.pt_u_in_zeros), list(self.pt_x_collocation), \
        #        list(self.pt_t_collocation), list(self.pt_all_zeros)
        return list(zip(self.pt_x_bc_var, self.pt_t_bc_zeros, self.pt_u_bc_sin, self.pt_x_in_pos_one,
                        self.pt_x_in_neg_one, self.pt_t_in_var, self.pt_u_in_zeros, self.pt_x_collocation,
                        self.pt_t_collocation,self.pt_all_zeros))

    def __len__(self):
        # return self.pt_x_bc_var.shape[0], self.pt_t_bc_zeros.shape[0], self.pt_u_bc_sin.shape[0], \
        #        self.pt_x_in_pos_one.shape[0], self.pt_x_in_neg_one.shape[0], self.pt_t_in_var.shape[0], \
        #        self.pt_u_in_zeros.shape[0], self.pt_x_collocation.shape[0], self.pt_t_collocation.shape[0], \
        #        self.pt_all_zeros.shape[0]
        return self.pt_x_bc_var.shape[0]

    def __getitem__(self, item):
        return item, self.pt_x_bc_var[item], self.pt_t_bc_zeros[item], self.pt_u_bc_sin[item], \
               self.pt_x_in_pos_one[item], self.pt_x_in_neg_one[item], self.pt_t_in_var[item], self.pt_u_in_zeros[item], \
               self.pt_x_collocation[item], self.pt_t_collocation[item], self.pt_all_zeros[item]


def make_testdata():
    t_bc_zeros = np.zeros((2000, 1))
    x_in_pos_one = np.ones((2000, 1))
    x_in_neg_one = -np.ones((2000, 1))
    u_in_zeros = np.zeros((2000, 1))
    x_collocation = np.random.uniform(low=-1.0, high=1.0, size=(2000, 1))
    t_collocation = np.random.uniform(low=0.0, high=1.0, size=(2000, 1))
    all_zeros = np.zeros((2000, 1))
    t_in_var = np.random.uniform(low=0, high=1.0, size=(2000, 1))
    x_bc_var = np.random.uniform(low=-1.0, high=1.0, size=(2000, 1))
    u_bc_sin = -np.sin(np.pi * x_bc_var)

    # 将数据转化为torch可用的
    pt_x_bc_var = Variable(torch.from_numpy(x_bc_var).float(), requires_grad=False)
    pt_t_bc_zeros = Variable(torch.from_numpy(t_bc_zeros).float(), requires_grad=False)
    pt_u_bc_sin = Variable(torch.from_numpy(u_bc_sin).float(), requires_grad=False)
    pt_x_in_pos_one = Variable(torch.from_numpy(x_in_pos_one).float(), requires_grad=False)
    pt_x_in_neg_one = Variable(torch.from_numpy(x_in_neg_one).float(), requires_grad=False)
    pt_t_in_var = Variable(torch.from_numpy(t_in_var).float(), requires_grad=False)
    pt_u_in_zeros = Variable(torch.from_numpy(u_in_zeros).float(), requires_grad=False)
    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True)
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False)

    #print(torch.Tensor(pt_x_bc_var))
    return pt_x_bc_var, pt_t_bc_zeros, pt_u_bc_sin, pt_x_in_pos_one, pt_x_in_neg_one, pt_t_in_var, \
           pt_u_in_zeros, pt_x_collocation, pt_t_collocation, pt_all_zeros


class MyTestDataset(Dataset):
    def __init__(self):
        super(MyTestDataset, self).__init__()
        self.pt_x_bc_var, self.pt_t_bc_zeros, self.pt_u_bc_sin, self.pt_x_in_pos_one, \
        self.pt_x_in_neg_one, self.pt_t_in_var, self.pt_u_in_zeros, self.pt_x_collocation, \
        self.pt_t_collocation, self.pt_all_zeros = make_testdata()

    def __iter__(self):
        # return list(self.pt_x_bc_var), list(self.pt_t_bc_zeros), list(self.pt_u_bc_sin), list(self.pt_x_in_pos_one), \
        #        list(self.pt_x_in_neg_one), list(self.pt_t_in_var), list(self.pt_u_in_zeros), list(self.pt_x_collocation), \
        #        list(self.pt_t_collocation), list(self.pt_all_zeros)
        return list(zip(self.pt_x_bc_var, self.pt_t_bc_zeros, self.pt_u_bc_sin, self.pt_x_in_pos_one,
                        self.pt_x_in_neg_one, self.pt_t_in_var, self.pt_u_in_zeros, self.pt_x_collocation,
                        self.pt_t_collocation,self.pt_all_zeros))

    def __len__(self):
        # return self.pt_x_bc_var.shape[0], self.pt_t_bc_zeros.shape[0], self.pt_u_bc_sin.shape[0], \
        #        self.pt_x_in_pos_one.shape[0], self.pt_x_in_neg_one.shape[0], self.pt_t_in_var.shape[0], \
        #        self.pt_u_in_zeros.shape[0], self.pt_x_collocation.shape[0], self.pt_t_collocation.shape[0], \
        #        self.pt_all_zeros.shape[0]
        return self.pt_x_bc_var.shape[0]

    def __getitem__(self, item):
        return item, self.pt_x_bc_var[item], self.pt_t_bc_zeros[item], self.pt_u_bc_sin[item], \
               self.pt_x_in_pos_one[item], self.pt_x_in_neg_one[item], self.pt_t_in_var[item], self.pt_u_in_zeros[item], \
               self.pt_x_collocation[item], self.pt_t_collocation[item], self.pt_all_zeros[item]

def get_dataloader():
    dataset = MyTrainDataset()

    return DataLoader(
        dataset,
        batch_size=32,
        #shuffle=True
    )

def get_testloader():
    dataset = MyTestDataset()

    return DataLoader(
        dataset,
        batch_size=32,
        #shuffle=True
    )