import numpy as np
import torch
from torch.nn.init import xavier_normal_


class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, device, **kwargs):
        """
        :param d: 数据集
        :param d1: 实体嵌入维度 200
        :param d2: 关系嵌入维度 200
        :param kwargs: 更多的参数
        """

        super(TuckER, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1)  # len(d.entities)为字典中词的数量，d1为embedding的维度
        self.R = torch.nn.Embedding(len(d.relations), d2)
        # W为网络训练参数
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)), 
                                    dtype=torch.float, device=device, requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        
        self.init()  # 调用附加的初始化函数，用于初始化权值

    def init(self):
        xavier_normal_(self.E.weight.data)  # 高斯初始化
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):
        """
        理解: pred=e1*r*W_r*E
        e1_idx即head
        r_idx即relation

        """
        
        # 主体的向量表示，后经batch norm和dropout存到x中
        e1 = self.E(e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
        # self.W.view(r.size(1), -1)为w_r, 它为来自R的关系表示，从模型参数W中用r取得w_r
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))  # 矩阵乘法：r @ w_r
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)  # 张量的后两维的矩阵乘法 e1 @ (r @ w_r)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1,0))  # 矩阵乘法: e1 @ (r @ w_r) @ W_E
        pred = torch.sigmoid(x)
        return pred

