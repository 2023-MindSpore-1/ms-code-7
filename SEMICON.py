import cv2
import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore import context
import mindspore.ops as ops
from mindspore.ops import functional as F
from resnet import resnet50
from mindspore.common.initializer import initializer, HeNormal
from torch.hub import load_state_dict_from_url

def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)

def conv3x3(in_channel, out_channel, stride=1, use_se=False):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def conv1x1(in_channel, out_channel, stride=1, use_se=False):
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=stride, padding=0, pad_mode='same', weight_init=weight)

def _fc(in_channel, out_channel, use_se=False):
    weight_shape = (out_channel, in_channel)
    weight = _weight_variable(weight_shape)
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


"""
Two stage
"""
class ChannelTransformer(nn.Cell):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim
        self.norm = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(axis=-1)

        self.qkv = nn.Conv2d(dim, dim * 3, 1, group=num_heads)
        self.qkv2 = nn.Conv2d(dim, dim * 3, 1, group=head_dim)
        
    def construct(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W).transpose(1,0,2,3,4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = ms.numpy.matmul(q, k.transpose(0,1,3,2)) * self.scale
        attn = ms.numpy.sign(attn) * ms.numpy.sqrt(ms.numpy.abs(attn) + 1e-5)
        attn = self.softmax(attn)

        x = (ms.numpy.matmul(attn, v).reshape(B, C, H, W) + x).reshape(B, self.num_heads, self.head_dim, H, W).transpose(0,2,1,3,4).reshape(B, C, H, W)
        y = self.norm(x)
        x = self.relu(y)
        qkv2 = self.qkv2(x).reshape(B, 3, self.head_dim, self.num_heads, H * W).transpose(1,0,2,3,4)
        q, k, v = qkv2[0], qkv2[1], qkv2[2]

        attn = (ms.numpy.matmul(q, k.transpose(0,1,3,2))) * (self.num_heads ** -0.5)
        attn = ms.numpy.sign(attn) * ms.numpy.sqrt(ms.numpy.abs(attn) + 1e-5)
        attn = self.softmax(attn)
        x = ms.numpy.matmul(attn, v).reshape(B, self.head_dim, self.num_heads, H, W).transpose(0,2,1,3,4).reshape(B, C, H, W) + y
        return x


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_Backbone(nn.Cell):
    
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet_Backbone, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=0, pad_mode='same')
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.SequentialCell(layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def construct(self, x):
        return self._forward_impl(x)


def SEMICON_backbone(pretrained=True, progress=True, **kwargs):
    model = ResNet_Backbone(Bottleneck, [3, 4, 6], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-0676ba61.pth",
                                              progress=progress)
        for name in list(state_dict.keys()):
            if 'fc' in name or 'layer4' in name:
                state_dict.pop(name)
        for name in list(state_dict.keys()):
            params = state_dict[name].data.numpy()
            params = ms.Tensor(params)
            if 'bn' in name or 'downsample.1' in name:
                state_dict.pop(name)
                name = name.replace('weight', 'gamma').replace('bias', 'beta') \
                                    .replace('running_mean', 'moving_mean').replace('running_var', 'moving_variance')
            state_dict[name] = ms.Parameter(params, name=name)
        ms.load_param_into_net(model, state_dict, True)
    return model


class ResNet_Refine(nn.Cell):

    def __init__(self, block, layer, is_local=True, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet_Refine, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 1024
        self.dilation = 1

        self.is_local = is_local
        self.groups = groups
        self.base_width = width_per_group
        self.layer4 = self._make_layer(block, 512, layer, stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = ops.AdaptiveAvgPool2D((1, 1))
        self.flatten = nn.Flatten()


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
            layers.append(ChannelTransformer(planes * block.expansion, max(planes * block.expansion // 64, 16)))
        return nn.SequentialCell(*layers)

    def _forward_impl(self, x):
        x = self.layer4(x)

        pool_x = self.avgpool(x)
        pool_x = self.flatten(pool_x)
        if self.is_local:
            return x, pool_x
        else:
            return pool_x

    def construct(self, x):
        return self._forward_impl(x)


def SEMICON_refine(is_local=True, pretrained=True, progress=True, **kwargs):
    model = ResNet_Refine(Bottleneck, 3, is_local, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-0676ba61.pth",
                                              progress=progress)
        for name in list(state_dict.keys()):
            if 'fc' in name or 'layer4' not in name:
                state_dict.pop(name)
        for name in list(state_dict.keys()):
            params = state_dict[name].data.numpy()
            params = ms.Tensor(params)
            if 'bn' in name or 'downsample.1' in name:
                state_dict.pop(name)
                name = name.replace('weight', 'gamma').replace('bias', 'beta') \
                                    .replace('running_mean', 'moving_mean').replace('running_var', 'moving_variance')
            state_dict[name] = ms.Parameter(params, name=name)
        ms.load_param_into_net(model, state_dict, True)
    return model

class SEM(nn.Cell):

    def __init__(self, block, layer, att_size=4, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(SEM, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 1024
        self.dilation = 1
        self.att_size = att_size
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer4 = self._make_layer(block, 512, layer, stride=1)

        self.feature1 = nn.SequentialCell(
            conv1x1(self.inplanes, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.feature2 = nn.SequentialCell(
            conv1x1(self.inplanes, 1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.feature3 = nn.SequentialCell(
            conv1x1(self.inplanes, 1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.rm_op = ops.ReduceMean()
        self.concat_op = ops.Concat(axis=1)
        self.mul = ops.Mul()
        self.softmax = nn.Softmax(axis=1)
        

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        att_expansion = 0.25
        layers = []
        layers.append(block(self.inplanes, int(self.inplanes * att_expansion), stride,
                            downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        for _ in range(1, blocks):
            layers.append(nn.SequentialCell(
                conv1x1(self.inplanes, int(self.inplanes * att_expansion)),
                nn.BatchNorm2d(int(self.inplanes * att_expansion))
            ))
            self.inplanes = int(self.inplanes * att_expansion)
            layers.append(block(self.inplanes, int(self.inplanes * att_expansion), groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.SequentialCell(*layers)

    def _mask(self, feature, x):
        """
        Sample:
        import mindspore as ms
        import mindspore.nn as nn
        import mindspore.ops as ops
        from mindspore.common.tensor import Tensor
        softmax = nn.Softmax(axis=1)
        rm_op = ops.ReduceMean()
        mul = ops.Mul()
        fea = Tensor([[[[0.,1.,10.],[-3.,9.,2.],[-5.,-6.,-1.]]],[[[-12,3,4],[-4,-2,-1],[1,3,9]]]])
        cam = fea.mean(1)
        attn = softmax(cam.view(2, 3 * 3))
        std = ms.numpy.std(attn)
        mean = rm_op(attn)
        attn = (attn - mean)/ (std ** 0.3) + 1
        attn = attn.view((2, 1, 3, 3))
        attn = ops.clip_by_value(attn, 0, 2)
        attn = 2-attn
        fea2 = mul(fea, attn)
        """
        cam1 = ms.numpy.mean(feature, 1)
        attn = self.softmax(cam1.view(x.shape[0], x.shape[2] * x.shape[3]))#B,H,W
        # std, mean = self.rm_op(attn)
        std = ms.numpy.std(attn)
        mean = self.rm_op(attn)
        attn = (attn - mean) / (std ** 0.3) + 1
        attn = attn.view((x.shape[0], 1, x.shape[2], x.shape[3]))
        attn = ops.clip_by_value(attn, 0, 2)
        return attn

    def _forward_impl(self, x):
        x = self.layer4(x)#bs*64*14*14
        fea1 = self.feature1(x) #bs*1*14*14
        attn = 2-self._mask(fea1, x)

        attn = ms.numpy.tile(attn, (1, self.inplanes, 1, 1))
        x = self.mul(x, attn)
        fea2 = self.feature2(x)
        attn = 2-self._mask(fea2, x)

        attn = ms.numpy.tile(attn, (1, self.inplanes, 1, 1))
        x = self.mul(x, attn)
        fea3 = self.feature3(x)

        # x = torch.cat([fea1,fea2,fea3], dim=1)
        x = self.concat_op((fea1, fea2, fea3))
        return x

    def construct(self, x):
        return self._forward_impl(x)


def SEMICON_attention(att_size=3, pretrained=False, progress=True, **kwargs):
    model = SEM(Bottleneck, 3, att_size=att_size, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-0676ba61.pth",
                                              progress=progress)
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        
        model.load_state_dict(state_dict)
    return model


"""
Visual
"""
class SEMICON(nn.Cell):
    def __init__(self, code_length=12, num_classes=200, att_size=3, feat_size=2048, pretrained=True):
        super(SEMICON, self).__init__()
        self.backbone = SEMICON_backbone(pretrained=pretrained)
        self.refine_global = SEMICON_refine(is_local=False, pretrained=pretrained)
        self.refine_local = SEMICON_refine(pretrained=pretrained)
        self.attention = SEMICON_attention(att_size=att_size)

        self.hash_layer_active = nn.Tanh()
        
        self.code_length = code_length
        self.feat_size = feat_size
        self.concat_op = ops.Concat(axis=1)
        self.cast_op = ops.Cast()

        #global
        if self.code_length != 32:
            # self.W_G = ms.Parameter(Tensor(np.zeros((self.code_length//2, self.feat_size)), ms.float32), name="W_G", requires_grad=True)
            #initializer(W_G, [1, 2, 3], mindspore.float32)
            # self.W_G = initializer(HeNormal(), [self.code_length//2, self.feat_size], ms.float32)
            self.Linear_G = nn.Dense(self.feat_size, self.code_length//2)
        else:
            # self.W_G = ms.Parameter(Tensor(np.zeros((self.code_length//2 + 1, self.feat_size)), ms.float32), name="W_G", requires_grad=True)
            # self.W_G = initializer(HeNormal(), [self.code_length//2 + 1, self.feat_size], ms.float32)
            self.Linear_G = nn.Dense(self.feat_size, self.code_length//2 + 1)
        
        #local
        # self.W_L1 = ms.Parameter(Tensor(np.zeros((code_length//6, feat_size)), ms.float32), name="W_L1", requires_grad=True)
        self.W_L1 = initializer(HeNormal(), [self.code_length//6, self.feat_size], ms.float32)
        # self.W_L2 = ms.Parameter(Tensor(np.zeros((code_length//6, feat_size)), ms.float32), name="W_L2", requires_grad=True)
        self.W_L2 = initializer(HeNormal(), [self.code_length//6, self.feat_size], ms.float32)
        # self.W_L3 = ms.Parameter(Tensor(np.zeros((code_length//6, feat_size)), ms.float32), name="W_L3", requires_grad=True)
        self.W_L3 = initializer(HeNormal(), [self.code_length//6, self.feat_size], ms.float32)
        
        self.Linear_L1 = nn.Dense(self.feat_size, self.code_length//6)
        self.Linear_L2 = nn.Dense(self.feat_size, self.code_length//6)
        self.Linear_L3 = nn.Dense(self.feat_size, self.code_length//6)
        
        self.expand_dims = ops.ExpandDims()
        self.mul = ops.Mul()

    def construct(self, x):
        out = self.backbone(x)#.detach()
        batch_size, channels, h, w = out.shape
        global_f = self.refine_global(out)
        att_map = self.attention(out)#batchsize * att-size * 14 * 14
        att_size = att_map.shape[1]
        # att_map_rep = att_map.unsqueeze(axis=2)
        att_map_rep = self.expand_dims(att_map, 2) #batchsize * att-size * 1 * 14 * 14
        att_map_rep = ms.numpy.tile(att_map_rep, (1, 1, channels, 1, 1)) #batchsize * att-size * channels * 14 * 14

        # out_rep = out.unsqueeze(axis=1)
        out_rep = self.expand_dims(out, 1)#batchsize * 1 * channels * 14 * 14
        out_rep = ms.numpy.tile(out_rep, (1, att_size, 1, 1, 1))#batchsize * att_size * channels * 14 * 14
        
        # out_local = att_map_rep.mul(out_rep)
        out_local = self.mul(att_map_rep, out_rep)#batchsize * att_size * channels * 14 * 14
        #batchsize * channels * 14 * 14
        out_local1 = out_local[:,:att_size//3,:,:].reshape(batch_size * att_size//3, channels, h, w)
        out_local2 = out_local[:,att_size//3:att_size*2//3,:,:].reshape(batch_size * att_size//3, channels, h, w)
        out_local3 = out_local[:,att_size*2//3:att_size*3//3,:,:].reshape(batch_size * att_size//3, channels, h, w)
        
        #batchsize * 2048 * 7 * 7; batchsize*2048
        local_f1, avg_local_f1 = self.refine_local(out_local1)
        local_f2, avg_local_f2 = self.refine_local(out_local2)
        local_f3, avg_local_f3 = self.refine_local(out_local3)

        # deep_S_G = F.linear(global_f, self.W_G)
        deep_S_G = self.Linear_G(global_f)

        # deep_S_1 = F.linear(avg_local_f1, self.W_L1)
        # deep_S_2 = F.linear(avg_local_f2, self.W_L2)
        # deep_S_3 = F.linear(avg_local_f3, self.W_L3)
        deep_S_1 = self.Linear_L1(avg_local_f1)
        deep_S_2 = self.Linear_L1(avg_local_f2)
        deep_S_3 = self.Linear_L1(avg_local_f3)
        # deep_S = torch.cat([deep_S_G, deep_S_1, deep_S_2, deep_S_3], dim = 1)
        deep_S = self.concat_op((deep_S_G, deep_S_1, deep_S_2, deep_S_3))

        ret = self.hash_layer_active(deep_S)
        return ret#, local_f1

def semicon(code_length, num_classes, att_size, feat_size, pretrained=False, **kwargs):
    model = SEMICON(code_length, num_classes, att_size, feat_size, pretrained, **kwargs)
    return model


if __name__ == '__main__':
    device = 'GPU'
    context.set_context(mode=context.PYNATIVE_MODE, device_target=device)
    # context.set_context(mode=context.GRAPH_MODE, device_target=device)
    x = np.random.uniform(-1, 1, (4, 3, 224, 224)).astype(np.float32)
    x = ms.Tensor(x)
    net = semicon(code_length=12, num_classes=200, att_size=3, feat_size=2048, pretrained=False)
    y = net(x)
    print(y.shape)
