from torch.nn import GRU, LSTM
from nn.TCN import TemporalConvNet


def get_cnn(config):
    return TemporalConvNet(num_inputs=config.hidden_size,
                           num_channels=config.cnn_num_channels,
                           kernel_size=config.cnn_kernel_size,
                           dropout=config.cnn_dropout)


def get_rnn(config):
    return GRU(config.hidden_size,
               config.hidden_size // 2,
               batch_first=True,
               num_layers=config.rnn_num_layers,
               bidirectional=True)


def get_myrnn(config):
    return LSTM(config.hidden_size,
               config.hidden_size // 2,
               batch_first=True,
               num_layers=8,
               bidirectional=True)