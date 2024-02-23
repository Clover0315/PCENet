import torch.nn as nn
import torch
from lib.utils.snake import snake_config

N_ADJ = snake_config.adj_num
class BaseConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=1, dilation=1):
        """
        :param state_dim:
        :param out_state_dim:
        """
        super(BaseConv, self).__init__()

        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=2 * n_adj + 1, dilation=dilation, padding=n_adj)

    def forward(self, input):
        return self.fc(input)


class CircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=N_ADJ, dilation=None):
        """
        圆环卷积
        :param state_dim:
        :param out_state_dim:
        :param n_adj: 邻居节点的个数，卷积核是-n_adj ~ n_adj 所以，卷积核大小 = 2*n_adj + 1
        """
        super(CircConv, self).__init__()

        self.n_adj = n_adj
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj * 2 + 1)

    def forward(self, input):
        input = torch.cat([input[..., -self.n_adj:], input, input[..., :self.n_adj]], dim=2)
        return self.fc(input)


class DilatedCircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=N_ADJ, dilation=1):
        """
        膨胀圆环卷积
        :param state_dim:
        :param out_state_dim:
        :param n_adj:
        :param dilation:
        """
        super(DilatedCircConv, self).__init__()

        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj * 2 + 1, dilation=self.dilation)

    def forward(self, input):
        if self.n_adj != 0:
            input = torch.cat(
                [input[..., -self.n_adj * self.dilation:], input, input[..., :self.n_adj * self.dilation]], dim=2)
        return self.fc(input)


_conv_factory = {
    'grid': CircConv,
    'dgrid': DilatedCircConv,
    'base': BaseConv
}


class BasicBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, conv_type, n_adj=N_ADJ, dilation=1):
        super(BasicBlock, self).__init__()

        self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj, dilation)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        return x


class Prediction(nn.Module):
    def __init__(self, dim_arr: tuple or list = snake_config.state_dims):
        super(Prediction, self).__init__()

        self.prediction = nn.Sequential(
            nn.Conv1d(dim_arr[0], dim_arr[1], 1),
            nn.ReLU(inplace=True),

            nn.Conv1d(dim_arr[1], dim_arr[2], 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim_arr[2], 1, 1),
        )

    def forward(self, x):
        x = self.prediction(x)
        return x


class Snake(nn.Module):
    def __init__(self, feature_dim, conv_type='dgrid',
                 dilation=[1, 1, 1, 4, 4, 8, 8],
                 ):
        super(Snake, self).__init__()

        state_dim = snake_config.state_dim
        self.head = BasicBlock(feature_dim, state_dim, conv_type)

        self.res_layer_num = len(dilation)

        for i in range(self.res_layer_num):
            conv = BasicBlock(state_dim, state_dim, conv_type, n_adj=N_ADJ, dilation=dilation[i])
            self.__setattr__('res' + str(i), conv)

        fusion_state_dim = snake_config.fusion_state_dim
        self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1), fusion_state_dim, 1)
        # self.prediction = nn.Sequential(
        #     nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv1d(256, 64, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(64, 2, 1),
        #
        #     # 加dropout
        #     # nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),  # 1028 ->
        #     # nn.ReLU(inplace=True),
        #     # nn.Dropout(),
        #     # nn.Conv1d(256, 64, 1),
        #     # nn.ReLU(inplace=True),
        #     # nn.Dropout(),
        #     # nn.Conv1d(64, 2, 1),
        # )

        self.prediction_x = Prediction()
        self.prediction_y = Prediction()

    def forward(self, x, return_state=False):
        states = []

        x = self.head(x)
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__('res' + str(i))(x) + x
            states.append(x)

        state = torch.cat(states, dim=1)

        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]  # 融合state后取最大池化
        global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
        state = torch.cat([global_state, state], dim=1)

        # state = self.fusion(state)  # 融合
        # x = self.prediction(state)
        # return x

        x = self.prediction_x(state)
        y = self.prediction_y(state)

        state = None if not return_state else state
        return x, y, state


if __name__ == "__main__":
    snake = Snake(64, 128)
    print(snake)
