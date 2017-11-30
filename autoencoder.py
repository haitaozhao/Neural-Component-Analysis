from datetime import datetime
import numpy as np
import torch
import torch.utils.data
from sklearn import preprocessing
import util


class MLP(torch.nn.Module):
    def __init__(self, i, h, o):
        super(MLP, self).__init__()
        self.h = torch.nn.Linear(i, h)
        self.a = torch.nn.Tanh()
        self.o = torch.nn.Linear(h, o)
        
    def forward(self, x):
        x = self.o(self.a(self.h(x)))
        return x

    def predict(self, data_loader):
        pred = []
        for data in data_loader:
            x = torch.autograd.Variable(data.cuda())
            o = self.h(x)
            pred.append(o.data.cpu().numpy())
        return np.vstack(pred)


class Metirc(object):

    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, pred, target):
        self._sum += np.sum(np.mean(np.power(target - pred, 2), axis=1))
        self._count += pred.shape[0]

    def get(self):
        return self._sum / self._count


def train(model, optimizer, train_loader):
    model.train()
    l2norm = Metirc()
    for data in train_loader:
        x = torch.autograd.Variable(data.cuda())
        y = model(x)

        loss = torch.mean(torch.pow(x - y, 2))
        l2norm.update(x.data.cpu().numpy(), y.data.cpu().numpy())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return l2norm.get()


def validate(model, val_loader):
    model.eval()
    l2norm = Metirc()
    for data in val_loader:
        x = torch.autograd.Variable(data.cuda())
        y = model(x)
        l2norm.update(y.data.cpu().numpy(), x.data.cpu().numpy())
    return l2norm.get()


def get_train_data():
    train_data, _ = util.read_data(error=0, is_train=True)
    train_data = preprocessing.StandardScaler().fit_transform(train_data)
    return train_data


def get_test_data():
    test_data = []
    for i in range(22):
        data, _ = util.read_data(error=i, is_train=False)
        test_data.append(data)
    test_data = np.concatenate(test_data)
    train_data, _ = util.read_data(error=0, is_train=True)
    scaler = preprocessing.StandardScaler().fit(train_data)
    test_data = scaler.transform(test_data)
    return test_data


def main():
    train_dataset = torch.from_numpy(get_train_data()).cuda()
    test_dataset = torch.from_numpy(get_test_data()).cuda()
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = MLP(52, 27, 52)
    model.cuda()
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    
    for i in range(500):
        train_acc = train(model, optimizer, train_loader)
        test_acc = validate(model, test_loader)
        print('{}\tepoch = {}\ttrain accuracy: {:0.3f}\ttest accuracy: {:0.3f}' \
                .format(datetime.now(), i, train_acc, test_acc))


if __name__ == '__main__':
    main()
