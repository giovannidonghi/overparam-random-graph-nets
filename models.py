import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
import torch.nn.functional as F

class UNTRAINEDGCN(torch.nn.Module):
    def __init__(self, n_feat, hidden_dim=1000, out="ROCKET", act=None, theta=1):
        super().__init__()
        self.out = out
        self.act=act
        self.lin= Linear(n_feat,hidden_dim)
        self.conv1 = GCNConv(n_feat, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.conv1.lin.weight, gain=theta) 
            torch.nn.init.xavier_uniform_(self.conv2.lin.weight, gain=theta)
            torch.nn.init.xavier_uniform_(self.conv3.lin.weight, gain=theta)
            torch.nn.init.xavier_uniform_(self.conv4.lin.weight, gain=theta)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_l= self.lin(x)
        if self.act=="TANH":
            x_l=torch.tanh(x_l)
            h1 = self.conv1(x, edge_index)
            h1_tanh = torch.tanh(h1)
            h2 = self.conv2(h1_tanh, edge_index)
            h2_tanh = torch.tanh(h2)
            h3 = self.conv3(h2_tanh, edge_index)
            h3_tanh = torch.tanh(h3)
            h4 = self.conv4(h3_tanh, edge_index)
            h4_tanh = torch.tanh(h4)
            h=torch.cat((h1_tanh,h2_tanh,h3_tanh,h4_tanh),1)
        else:
            h1 = self.conv1(x, edge_index)
            h2 = self.conv2(h1, edge_index)
            h3 = self.conv3(h2, edge_index)
            h4 = self.conv4(h3, edge_index)
            h= torch.cat((h1,h2,h3,h4),1) 

        h_pooled=global_max_pool(h, batch)

        if self.out=="ROCKET":
            h_pos=torch.gt(h,0).float()
            h_pos_pooled=global_mean_pool(h_pos, batch)
            return torch.cat((h_pooled,h_pos_pooled),1)
        elif self.out=="MAX":
            return h_pooled
        elif self.out=="PPV":
            h_pos=torch.gt(h,0).float()
            h_pos_pooled=global_mean_pool(h_pos, batch)
            return h_pos_pooled
        else:
            print("ERROR! Unrecognized network output selection")

def transform_with_untrained_GCN(model, train_loader,device):
          model.eval()
          out=[]
          y=[]
          for data in train_loader:
                data = data.to(device)

                out.append(model(data))
                y.append(data.y.to(torch.float).unsqueeze(1))
          repr= torch.cat(out)
          y_tot= torch.cat(y)

          return repr.detach().to("cpu"), y_tot.detach().to("cpu")
