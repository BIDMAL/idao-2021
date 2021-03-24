import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

# добавил строчку
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Print(nn.Module):
    """Debugging only"""

    def forward(self, x):
        print(x.size())
        return x


class Clamp(nn.Module):
    """Clamp energy output"""

    def forward(self, x):
        x = torch.clamp(x, min=0, max=30)
        return x

# Добавил класс модели для классификации
class ConvNetClassification(nn.Module): 
        def __init__(self):
            super(ConvNetClassification, self).__init__()
            self.c1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
            self.c2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
            self.c3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
            self.fc1 = nn.Linear(in_features=2048, out_features=1024)
            self.fc2 = nn.Linear(in_features=1024, out_features=512)
            self.fc3 = nn.Linear(in_features=512, out_features=256)
            self.fc4 = nn.Linear(in_features=256, out_features=2)
        
        def forward(self, x):
            x = F.relu(F.max_pool2d(self.c1(x), 3))
            x = F.relu(F.max_pool2d(self.c2(x), 3))
            x = F.relu(F.max_pool2d(self.c3(x), 3))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)

            return F.log_softmax(x, dim=1)

# Добавил класс модели для регрессии
class ConvNetRegression(nn.Module): 
        def __init__(self):
            super(ConvNetRegression, self).__init__()
            self.c1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
            self.c2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
            self.c3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
            self.fc1 = nn.Linear(in_features=2048, out_features=1024)
            self.fc2 = nn.Linear(in_features=1024, out_features=512)
            self.fc3 = nn.Linear(in_features=512, out_features=256)
            self.fc4 = nn.Linear(in_features=256, out_features=6)
        
        def forward(self, x):
            x = F.relu(F.max_pool2d(self.c1(x), 3))
            x = F.relu(F.max_pool2d(self.c2(x), 3))
            x = F.relu(F.max_pool2d(self.c3(x), 3))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)

            return F.log_softmax(x, dim=1)

# загрузка моделей
#net_regr = torch.load('./idao/IDAO_nn_ENERJY_v_2_1.pt')
#net_regr = net_regr.to(device)

#net_class = torch.load('./idao//IDAO_nn_NR_ER_v_2_1.pt')
#net_class = net_class.to(device)