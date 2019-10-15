import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import os
import sys
from collections import OrderedDict
from model.tdcnn.tdcnn import _TDCNN


class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):
    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        # print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                        stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3D(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3D, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        # self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
        #                                                   padding=0)
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 1, 1], stride=(2, 1, 1),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[8, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()



    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    def build(self):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                # print end_point
                self.add_module(end_point, self.end_points[end_point])
        # for k in self.end_points.keys():
        #     # self.features.add_module(k, self.end_points[k])
        #     self.add_module(k, self.end_points[k])
        self._init_weights(self.end_points)

    def _init_weights(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)  # use _modules to work with dataparallel
        # print('before avg', x.size())
        x = self.avg_pool(x)
        # print('after avg, before logit', x.size())
        # x = self.logits(self.dropout(x))
        x = self.logits(x)
        # print('after logit, before spatial squeeze', x.size())
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        # logits is batch X time X classes, which is what we want to work with
        # print('after spatial squeeze', logits.size())
        return logits

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)


def get_i3d_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5 + 1):
        ft_module_names.append('module.Mixed_{}'.format(i))
    ft_module_names.append('module.logits')

    parameters = []
    freeze = []
    tune = []
    names = []
    for k, v in model.named_parameters():
        names.append(k)
        no_grad = True
        for ft_module in ft_module_names:
            if k.startswith(ft_module):
                parameters.append({'params': v})
                tune.append(k)
                no_grad = False
                break
        if no_grad:
            # really need this?
            # parameters.append({'params': v, 'lr': 0.0})
            v.requires_grad = False
            freeze.append(k)
    print('fine_tune:', len(tune), tune)
    print('freeze', len(freeze), freeze)
    print('all', len(names))
    return parameters

# class i3d_tdcnn(_TDCNN):
#     def __init__(self, pretrained=False):
#         self.model_path = 'data/pretrained_model/rgb_imagenet.pkl'
#         self.dout_base_model = 832
#         self.pretrained = pretrained
#         _TDCNN.__init__(self)
#     def _init_modules(self):
#         i3d = I3D()
#         if self.pretrained:
#             print("Loading pretrained weights from %s" %(self.model_path))
#             state_dict = torch.load(self.model_path)
#             i3d.load_state_dict({k:v for k,v in state_dict.items() if k in i3d.state_dict()})
#
#         # Using
#         self.RCNN_base = nn.Sequential(*list(i3d.features._modules.values())[:-5])
#         # Using
#         self.RCNN_top = nn.Sequential(*list(i3d.features._modules.values())[-5:-3])
#
#         # not using the last maxpool layer
#         self.RCNN_cls_score = nn.Linear(1024, self.n_classes)
#         self.RCNN_twin_pred = nn.Linear(1024, 2 * self.n_classes)
#
#         # Fix blocks:
#         # TODO: fix blocks optionally
#         for layer in range(5):
#             for p in self.RCNN_base[layer].parameters(): p.requires_grad = False
#
#         def set_bn_fix(m):
#             classname = m.__class__.__name__
#             if classname.find('BatchNorm') != -1:
#                 for p in m.parameters(): p.requires_grad=False
#
#         self.RCNN_base.apply(set_bn_fix)
#         self.RCNN_top.apply(set_bn_fix)
#
#     def train(self, mode=True):
#         # Override train so that the training mode is set as we want
#         nn.Module.train(self, mode)
#         if mode:
#             # Set fixed blocks to be in eval mode
#             self.RCNN_base.eval()
#             self.RCNN_base[5].train()
#             self.RCNN_base[6].train()
#             self.RCNN_base[8].train()
#             self.RCNN_base[9].train()
#             self.RCNN_base[10].train()
#             self.RCNN_base[11].train()
#             self.RCNN_base[12].train()
#
#         def set_bn_eval(m):
#             classname = m.__class__.__name__
#             if classname.find('BatchNorm') != -1:
#                 m.eval()
#
#         self.RCNN_base.apply(set_bn_eval)
#         self.RCNN_top.apply(set_bn_eval)
#
#     def _head_to_tail(self, pool5):
#         fc6 = self.RCNN_top(pool5).mean(4).mean(3).mean(2)
#         return fc6

class i3d_tdcnn(_TDCNN):
    def __init__(self, pretrained=False):
        self.model_path = 'data/pretrained_model/rgb_imagenet.pth'
        # self.dout_base_model = 832
        self.dout_base_model = 1024
        self.pretrained = pretrained
        self.num_channels = 3
        self.num_classes = 400
        self.dropout_keep_prob = 1.0
        self.linear_dim = 1024
        _TDCNN.__init__(self)
    def _init_modules(self):

        model = InceptionI3D(
            # num_classes=157, # charades 157
            num_classes=self.num_classes,
            spatial_squeeze=True,
            final_endpoint='Logits',
            in_channels=self.num_channels,
            dropout_keep_prob=self.dropout_keep_prob)

        if self.pretrained:
            print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)
            model.load_state_dict({k:v for k,v in state_dict.items() if k in model.state_dict()})
        #
        # RCNN_base_keys = ['Conv3d_1a_7x7', 'MaxPool3d_2a_3x3', 'Conv3d_2b_1x1', 'Conv3d_2c_3x3', 'MaxPool3d_3a_3x3',
        #                   'Mixed_3b', 'Mixed_3c', 'MaxPool3d_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
        #                   'Mixed_4f', 'MaxPool3d_5a_2x2', 'Mixed_5b', 'Mixed_5c']
        RCNN_base_keys = ['Conv3d_1a_7x7', 'MaxPool3d_2a_3x3', 'Conv3d_2b_1x1', 'Conv3d_2c_3x3', 'MaxPool3d_3a_3x3',
                          'Mixed_3b', 'Mixed_3c', 'MaxPool3d_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
                          'Mixed_4f', 'MaxPool3d_5a_2x2','Mixed_5b', 'Mixed_5c']
        # RCNN_top_keys = ['Mixed_5b', 'Mixed_5c']

        RCNN_base = OrderedDict()
        # RCNN_top = OrderedDict()

        for key in RCNN_base_keys:
            RCNN_base[key] = model._modules[key]
        # for key in RCNN_top_keys:
        #     RCNN_top[key] = model._modules[key]

        self.RCNN_base = nn.Sequential(*list(RCNN_base.values()))
        # self.RCNN_top = nn.Sequential(*list(RCNN_top.values()))
        self.RCNN_top = nn.Sequential(
            # nn.AvgPool3d(kernel_size=[4, 2, 2], stride=(1, 1, 1)),
            nn.Linear(1024*4, self.linear_dim),
            nn.ReLU(True),
            nn.Dropout(inplace=False),
        )

        # # Using
        # self.RCNN_base = nn.Sequential(*list(i3d.features._modules.values())[:-5])
        # # Using
        # self.RCNN_top = nn.Sequential(*list(i3d.features._modules.values())[-5:-3])

        # not using the last maxpool layer
        self.RCNN_cls_score = nn.Linear(1024, self.n_classes)
        self.RCNN_twin_pred = nn.Linear(1024, 2 * self.n_classes)

        # self.avg = nn.AvgPool3d(kernel_size=[4, 2, 2], stride=(1, 1, 1))
        #
        # # Fix blocks:
        # # TODO: fix blocks optionally
        # for layer in range(5):
        #     for p in self.RCNN_base[layer].parameters(): p.requires_grad = False
        #
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False

        self.RCNN_base.apply(set_bn_fix)
        self.RCNN_top.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        # print()
        # print(len(self.RCNN_base))
        if mode:
            # Set fixed blocks to be in eval mode
            # self.RCNN_base.train()
            self.RCNN_base.eval()
            self.RCNN_base[8].train()
            self.RCNN_base[9].train()
            self.RCNN_base[10].train()
            self.RCNN_base[11].train()
            self.RCNN_base[12].train()
            self.RCNN_base[13].train()
            self.RCNN_base[14].train()
            self.RCNN_base[15].train()

        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.RCNN_base.apply(set_bn_eval)
        self.RCNN_top.apply(set_bn_eval)

    def _head_to_tail(self, pooled_feat):

        pool5 = pooled_feat.mean(4).mean(3)
        pool5_flat = pool5.view(pool5.size(0), -1)
        # fc6 = self.RCNN_top(pool5).mean(4).mean(3).mean(2)
        fc6 = self.RCNN_top(pool5_flat)
        return fc6


