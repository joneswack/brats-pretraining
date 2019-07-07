from torch import nn
import torch
import torchvision.models as models

def conv2d_to_conv3d(layer):
    in_channels = layer.in_channels
    out_channels = layer.out_channels
    new_weight = layer.weight.unsqueeze(2)
    kernel_size = tuple([1] + list(layer.kernel_size))
    stride = tuple([1] + list(layer.stride))
    padding = tuple([0] + list(layer.padding))

    new_layer = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
    
    new_weight = layer.weight.unsqueeze(2)
    new_layer.weight.data = new_weight.data
    if layer.bias is not None:
        new_layer.bias.data = layer.bias.data
    
    return new_layer

def batch2d_to_batch3d(layer):
    num_features = layer.num_features
    eps = layer.eps
    momentum = layer.momentum
    affine = layer.affine
    
    new_bn = nn.BatchNorm3d(num_features, eps, momentum, affine)
    new_bn.load_state_dict(layer.state_dict())
    
    return new_bn

def pool2d_to_pool3d(layer):
    kernel_size = tuple([1, layer.kernel_size, layer.kernel_size])
    stride = tuple([1, layer.stride, layer.stride])
    padding = layer.padding
    dilation = layer.dilation
    return_indices = layer.return_indices
    ceil_mode = layer.ceil_mode
    
    return nn.MaxPool3d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

def transform_module_list(module_list):
    transformed_list = []
    
    for item in module_list:
        if isinstance(item, nn.Conv2d): 
            transformed_list.append(conv2d_to_conv3d(item))
        elif isinstance(item, nn.BatchNorm2d):
            transformed_list.append(batch2d_to_batch3d(item))
        elif isinstance(item, nn.MaxPool2d):
            transformed_list.append(pool2d_to_pool3d(item))
        elif isinstance(item, nn.ReLU):
            transformed_list.append(item)
        elif isinstance(item, nn.Sequential):
            transformed_list += transform_module_list(item.children())
        else:
            transformed_list.append(item)
            
    return transformed_list

def translate_block(block, downsample=False):
    block.conv1 = conv2d_to_conv3d(block.conv1)
    block.bn1 = batch2d_to_batch3d(block.bn1)
    
    block.conv2 = conv2d_to_conv3d(block.conv2)
    block.bn2 = batch2d_to_batch3d(block.bn2)
    
    if block.downsample is not None:
        conv = conv2d_to_conv3d(block.downsample[0])
        bn = batch2d_to_batch3d(block.downsample[1])
        block.downsample = nn.Sequential(*[conv, bn])
    
    return block

def conv1x3x3(in_, out):
    return nn.Conv3d(in_, out, kernel_size=(1,3,3), padding=(0,1,1))

def conv3x1x1(in_, out):
    # padding=(0,0,0) when depth reduction, otherwise (1,0,0)
    # return nn.Conv3d(in_, out, kernel_size=(3,1,1), padding=(0,0,0))
    return nn.Conv3d(in_, out, kernel_size=(3,1,1), padding=(1,0,0))

class ConvRelu(nn.Module):
    def __init__(self, in_, out, depth=False):
        super().__init__()
        if depth:
            self.conv = conv3x1x1(in_, out)
        else:
            self.conv = conv1x3x3(in_, out)
        self.bn = nn.BatchNorm3d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
class ConvReluFromEnc(nn.Module):
    def __init__(self, conv, bn, relu):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class Upsample2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose3d(middle_channels, out_channels, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class JonasNet(nn.Module):
    def __init__(self, num_classes=1, cut_layer=14, max_filters=128, pretrained=False, is_deconv=True):
        super().__init__()
        base_model = models.vgg16_bn(pretrained=pretrained)
        self.num_classes = num_classes
        self.encoder = nn.Sequential(*transform_module_list(list(base_model.features.children())[:cut_layer]))
        # pick filters from batchnorm: max_filters = encoder[-3].num_features
        
        self.enc1 = ConvReluFromEnc(self.encoder[0], self.encoder[1], self.encoder[2])
        self.enc2 = ConvReluFromEnc(self.encoder[3], self.encoder[4], self.encoder[5])
        self.pool1 = self.encoder[6]
        self.enc3 = ConvReluFromEnc(self.encoder[7], self.encoder[8], self.encoder[9])
        self.enc4 = ConvReluFromEnc(self.encoder[10], self.encoder[11], self.encoder[12])
        self.pool2 = self.encoder[13]
    
        self.depth1 = ConvRelu(max_filters, max_filters, depth=True)
        self.center = ConvRelu(max_filters, max_filters // 2, depth=False)
        # out1 starts here
        
        # join center and pool2
        self.dec1 = Upsample2D(max_filters//2 + max_filters, max_filters//2 + max_filters, max_filters//2)
        
        # join dec1 and enc4
        self.dec2 = ConvRelu(max_filters//2 + max_filters, max_filters//2)
        
        # second depth component
        self.depth2 = ConvRelu(max_filters//2, max_filters//2, depth=True)
        
        # join depth2 and enc3
        self.dec3 = Upsample2D(max_filters//2 + max_filters, max_filters//2 + max_filters, max_filters//2)
        
        # join dec3 and enc2
        self.dec4 = ConvRelu(max_filters//2 + max_filters//2, max_filters//4)
        
        self.depth3 = ConvRelu(max_filters//4, max_filters//4, depth=True)
        
        # join dec4 and enc1
        self.final = nn.Conv3d(max_filters//4 + max_filters//2, num_classes, kernel_size=1)
        
        
#         # out2 starts here
        
#         self.dec2 = ConvRelu(max_filters//2, max_filters//4, depth=False)
#         self.depth2 = ConvRelu(max_filters//4, max_filters//4, depth=True)
#         # out3 starts here
        
#         # out1
#         self.up1_1 = Upsample2D(max_filters, max_filters//2, is_deconv=is_deconv)
#         self.out1 = ConvRelu(max_filters//2, max_filters//4, depth=False)
#         self.up1_2 = Upsample2D(max_filters//4, max_filters//8, is_deconv=is_deconv)
#         self.res1 = ConvRelu(max_filters//8, num_classes, depth=False)
        
#         # out2
#         self.up2 = Upsample2D(max_filters//2, max_filters//4, is_deconv=is_deconv)
#         self.out2 = ConvRelu(max_filters//4, max_filters//8, depth=False)
#         self.res2 = Upsample2D(max_filters//8, num_classes, is_deconv=is_deconv)
        
#         # out3
#         self.up3 = Upsample2D(max_filters//4, max_filters//8, is_deconv=is_deconv)
#         self.out3 = ConvRelu(max_filters//8, num_classes, depth=False)
#         self.res3 = Upsample2D(num_classes, num_classes, is_deconv=is_deconv)
        
#         # final prediction
#         self.final = ConvRelu(num_classes*3, 1, depth=False)
        
    def forward(self, x):
        # encoder
        # out = self.encoder.forward(x)
        conv1 = self.enc1(x) # 64
        conv2 = self.enc2(conv1) # 64
        conv3 = self.enc3(self.pool1(conv2)) # 128
        conv4 = self.enc4(conv3) # 128
        conv5 = self.pool2(conv4) # 128
        
        center = self.center(self.depth1(conv5)) # 64
        
        print('Conv1', conv1.shape)
        print('Conv2', conv2.shape)
        print('Conv3', conv3.shape)
        print('Conv4', conv4.shape)
        print('Conv5', conv5.shape)
        print('Center', center.shape)
        
        dec1 = self.dec1(torch.cat([center, conv5], 1)) # 64
        print('Dec1', dec1.shape)
        dec2 = self.dec2(torch.cat([dec1, conv4], 1)) # 64
        print('Dec2', dec2.shape)
        dec2 = self.depth2(dec2)
        dec3 = self.dec3(torch.cat([dec2, conv3], 1)) # 32
        print('Dec3', dec3.shape)
        dec4 = self.dec4(torch.cat([dec3, conv2], 1)) # 32
        print('Dec4', dec4.shape)
        dec5 = self.depth3(dec4)
        print('Dec5', dec5.shape)
        out = self.final(torch.cat([dec5, conv1], 1))
        print('Out', out.shape)
        
#         interm1 = self.depth1.forward(out)
#         interm2 = self.dec1.forward(interm1)
#         out = self.dec2.forward(interm2)
#         interm3 = self.depth2.forward(out)
        
#         final1 = self.up1_1.forward(interm1)
#         final1 = self.out1.forward(final1)
#         final1 = self.up1_2.forward(final1)
#         final1 = self.res1.forward(final1)
        
#         final2 = self.up2.forward(interm2)
#         final2 = self.out2.forward(final2)
#         final2 = self.res2.forward(final2)
        
#         final3 = self.up3.forward(interm3)
#         final3 = self.out3.forward(final3)
#         final3 = self.res3.forward(final3)
        
#         print(final1.shape)
#         print(final2.shape)
#         print(final3.shape)
        
#         out = self.final.forward(torch.cat([final1, final2, final3], 1))
        
        return out
    
class AlbuNet3D18(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False, is_deconv=False):
        super(AlbuNet3D18, self).__init__()
        # resnet18 has only 2 blocks per layer group
        self.num_classes = num_classes
        self.encoder = models.resnet18(pretrained=pretrained)
        
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        conv0_modules = [self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool]
        self.conv0 = nn.Sequential(*transform_module_list(conv0_modules))
        
        self.conv1_0 = translate_block(self.encoder.layer1[0])
        self.conv1_1 = translate_block(self.encoder.layer1[1])
        
        # number output filters: 64
        
        self.conv2_0 = translate_block(self.encoder.layer2[0])
        self.conv2_1 = translate_block(self.encoder.layer2[1])
        
        # number output filters: 128
        
        self.conv3_0 = translate_block(self.encoder.layer3[0])
        self.conv3_1 = translate_block(self.encoder.layer3[1])
        
        # number output filters: 256
        
        self.conv4_0 = translate_block(self.encoder.layer4[0])
        self.conv4_1 = translate_block(self.encoder.layer4[1])

        # number output filters: 512

        self.center = ConvRelu(512, num_filters * 8) # 128

        self.dec5 = Upsample2D(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = Upsample2D(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = Upsample2D(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = Upsample2D(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2)
        self.dec1 = Upsample2D(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv3d(num_filters, num_classes, kernel_size=1)
        
        self.depth1 = ConvRelu(512, 512, depth=True)
        self.depth2 = ConvRelu(num_filters * 8, num_filters * 8, depth=True)
        self.depth3 = ConvRelu(num_filters * 2 * 2, num_filters * 2 * 2, depth=True)
        self.depth4 = ConvRelu(num_filters, num_filters, depth=True)
        
    def forward(self, x):
        # print(x.shape)
        
        conv0 = self.conv0(x)
        # conv1 = self.conv1_2(self.conv1_1(self.conv1_0(conv0)))
        conv1 = self.conv1_1(self.conv1_0(conv0))
        # conv2 = self.conv2_3(self.conv2_2(self.conv2_1(self.conv2_0(conv1))))
        conv2 = self.conv2_1(self.conv2_0(conv1))
        # conv3 = self.conv3_5(self.conv3_4(self.conv3_3(self.conv3_2(self.conv3_1(self.conv3_0(conv2))))))
        conv3 = self.conv3_1(self.conv3_0(conv2))
        # conv4 = self.conv4_2(self.conv4_1(self.conv4_0(conv3)))
        conv4 = self.conv4_1(self.conv4_0(conv3))
        
        conv4 = self.depth1(conv4)
        
        # center = self.center(self.pool(conv4))
        center = self.center(conv4)
        
#         print('Center', center.shape)
#         print('Conv4', conv4.shape)

        dec5 = self.dec5(torch.cat([center, conv4], 1))
        dec4 = self.dec4(torch.cat([dec5, conv3], 1))
        dec4 = self.depth2(dec4)
        
        dec3 = self.dec3(torch.cat([dec4, conv2], 1))
        dec2 = self.dec2(torch.cat([dec3, conv1], 1))
        dec2 = self.depth3(dec2)
        
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)
        dec0 = self.depth4(dec0)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)

        return x_out

    
class AlbuNet3D34(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False, is_deconv=False):
        super(AlbuNet3D34, self).__init__()
        # resnet34 has a lot of blocks
        self.num_classes = num_classes
        self.encoder = models.resnet34(pretrained=pretrained)
        
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        conv0_modules = [self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool]
        self.conv0 = nn.Sequential(*transform_module_list(conv0_modules))
        
        self.conv1_0 = translate_block(self.encoder.layer1[0])
        self.conv1_1 = translate_block(self.encoder.layer1[1])
        self.conv1_2 = translate_block(self.encoder.layer1[2])
        
        # number output filters: 64
        
        self.conv2_0 = translate_block(self.encoder.layer2[0])
        self.conv2_1 = translate_block(self.encoder.layer2[1])
        self.conv2_2 = translate_block(self.encoder.layer2[2])
        self.conv2_3 = translate_block(self.encoder.layer2[3])
        
        # number output filters: 128
        
        self.conv3_0 = translate_block(self.encoder.layer3[0])
        self.conv3_1 = translate_block(self.encoder.layer3[1])
        self.conv3_2 = translate_block(self.encoder.layer3[2])
        self.conv3_3 = translate_block(self.encoder.layer3[3])
        self.conv3_4 = translate_block(self.encoder.layer3[4])
        self.conv3_5 = translate_block(self.encoder.layer3[5])
        
        # number output filters: 256
        
        self.conv4_0 = translate_block(self.encoder.layer4[0])
        self.conv4_1 = translate_block(self.encoder.layer4[1])
        self.conv4_2 = translate_block(self.encoder.layer4[2])
        
        # number output filters: 512
        
        # self.center = Upsample2D(512, num_filters * 8 * 2, num_filters * 8)
        self.center = ConvRelu(512, num_filters * 8) # 128

        self.dec5 = Upsample2D(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        # self.dec5 = Upsample2D(512, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = Upsample2D(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = Upsample2D(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = Upsample2D(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2)
        self.dec1 = Upsample2D(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv3d(num_filters, num_classes, kernel_size=1)
        
        self.depth1 = ConvRelu(512, 512, depth=True)
        self.depth2 = ConvRelu(num_filters * 8, num_filters * 8, depth=True)
        self.depth3 = ConvRelu(num_filters * 2 * 2, num_filters * 2 * 2, depth=True)
        self.depth4 = ConvRelu(num_filters, num_filters, depth=True)
        
    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1_2(self.conv1_1(self.conv1_0(conv0)))
        conv2 = self.conv2_3(self.conv2_2(self.conv2_1(self.conv2_0(conv1))))
        conv3 = self.conv3_5(self.conv3_4(self.conv3_3(self.conv3_2(self.conv3_1(self.conv3_0(conv2))))))
        conv4 = self.conv4_2(self.conv4_1(self.conv4_0(conv3)))
        
        conv4 = self.depth1(conv4)
        
        center = self.center(conv4)

        dec5 = self.dec5(torch.cat([center, conv4], 1))
        dec4 = self.dec4(torch.cat([dec5, conv3], 1))
        dec4 = self.depth2(dec4)
        
        dec3 = self.dec3(torch.cat([dec4, conv2], 1))
        dec2 = self.dec2(torch.cat([dec3, conv1], 1))
        dec2 = self.depth3(dec2)
        
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)
        dec0 = self.depth4(dec0)

        if self.num_classes > 1:
            # x_out = F.log_softmax(self.final(dec0), dim=1)
            x_out = self.final(dec0) # softmax is moved to loss metric
        else:
            x_out = self.final(dec0)

        return x_out