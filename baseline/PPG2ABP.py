# https://github.com/nibtehaz/PPG2ABP/tree/master
"""
    Models used in experiments
"""

# Standard library imports
# None

# Third-party imports
import torch
import torch.nn as nn

class UNetDS64(nn.Module):
    """
    Deeply supervised U-Net with kernels multiples of 64
    
    Args:
        length (int): length of the input signal
        n_channel (int): number of channels in the input (default: 1)
    """
    def __init__(self, n_channel=1, conv_channel = 64):
        super(UNetDS64, self).__init__()

        x = conv_channel
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_channel, x, 3, padding=1),
            nn.BatchNorm1d(x),
            nn.ReLU(),
            nn.Conv1d(x, x, 3, padding=1),
            nn.BatchNorm1d(x),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(x, x*2, 3, padding=1),
            nn.BatchNorm1d(x*2),
            nn.ReLU(),
            nn.Conv1d(x*2, x*2, 3, padding=1),
            nn.BatchNorm1d(x*2),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Sequential(
            nn.Conv1d(x*2, x*4, 3, padding=1),
            nn.BatchNorm1d(x*4),
            nn.ReLU(),
            nn.Conv1d(x*4, x*4, 3, padding=1),
            nn.BatchNorm1d(x*4),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Sequential(
            nn.Conv1d(x*4, x*8, 3, padding=1),
            nn.BatchNorm1d(x*8),
            nn.ReLU(),
            nn.Conv1d(x*8, x*8, 3, padding=1),
            nn.BatchNorm1d(x*8),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool1d(2)

        self.conv5 = nn.Sequential(
            nn.Conv1d(x*8, x*16, 3, padding=1),
            nn.BatchNorm1d(x*16),
            nn.ReLU(),
            nn.Conv1d(x*16, x*16, 3, padding=1),
            nn.BatchNorm1d(x*16),
            nn.ReLU()
        )
        
        # Deep supervision outputs
        self.level4 = nn.Conv1d(x*16, 1, 1)

        # Decoder
        self.up6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6 = nn.Sequential(
            nn.Conv1d(x*16 + x*8, x*8, 3, padding=1),
            nn.BatchNorm1d(x*8),
            nn.ReLU(),
            nn.Conv1d(x*8, x*8, 3, padding=1),
            nn.BatchNorm1d(x*8),
            nn.ReLU()
        )
        self.level3 = nn.Conv1d(x*8, 1, 1)

        self.up7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7 = nn.Sequential(
            nn.Conv1d(x*8 + x*4, x*4, 3, padding=1),
            nn.BatchNorm1d(x*4),
            nn.ReLU(),
            nn.Conv1d(x*4, x*4, 3, padding=1),
            nn.BatchNorm1d(x*4),
            nn.ReLU()
        )
        self.level2 = nn.Conv1d(x*4, 1, 1)

        self.up8 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8 = nn.Sequential(
            nn.Conv1d(x*4 + x*2, x*2, 3, padding=1),
            nn.BatchNorm1d(x*2),
            nn.ReLU(),
            nn.Conv1d(x*2, x*2, 3, padding=1),
            nn.BatchNorm1d(x*2),
            nn.ReLU()
        )
        self.level1 = nn.Conv1d(x*2, 1, 1)

        self.up9 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv9 = nn.Sequential(
            nn.Conv1d(x*2 + x, x, 3, padding=1),
            nn.BatchNorm1d(x),
            nn.ReLU(),
            nn.Conv1d(x, x, 3, padding=1),
            nn.BatchNorm1d(x),
            nn.ReLU()
        )
        self.out = nn.Conv1d(x, 1, 1)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.conv5(pool4)
        level4 = self.level4(conv5)

        # Decoder
        up6 = self.up6(conv5)
        merge6 = torch.cat([up6, conv4], dim=1)
        conv6 = self.conv6(merge6)
        level3 = self.level3(conv6)

        up7 = self.up7(conv6)
        merge7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.conv7(merge7)
        level2 = self.level2(conv7)

        up8 = self.up8(conv7)
        merge8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.conv8(merge8)
        level1 = self.level1(conv8)

        up9 = self.up9(conv8)
        merge9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.conv9(merge9)
        out = self.out(conv9)

        return out, level1, level2, level3, level4


class MultiResUNet1D(nn.Module):
    """
    1D MultiResUNet
    
    Args:
        length (int): length of the input signal
        n_channel (int): number of channels in the input (default: 1)
    """
    def __init__(self, alpha=2.5, n_channel=1):
        super(MultiResUNet1D, self).__init__()
        self.alpha = alpha

        def conv_bn(in_channels, out_channels, num_row=3, num_col=3, activation=True):
            # Match Keras conv2d_bn signature
            kernel_size = num_row  # For 1D, we use num_row
            padding = kernel_size // 2
            layers = [
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels)
            ]
            if activation:
                layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        class TransConvBN(nn.Module):
            def __init__(self, in_channels, skip_channels):
                super().__init__()
                self.up = nn.Upsample(scale_factor=2, mode='nearest')
                # BatchNorm after concatenation needs in_channels + skip_channels
                self.bn = nn.BatchNorm1d(in_channels + skip_channels)
            
            def forward(self, x):
                x = self.up(x)
                return x  # Don't apply BatchNorm here, will be applied after concatenation

        def MultiResBlock(U, in_channels):
            W = int(self.alpha * U)
            
            out_3x3 = int(W*0.167)
            out_5x5 = int(W*0.333)
            out_7x7 = int(W*0.5)
            concat_channels = out_3x3 + out_5x5 + out_7x7

            return nn.ModuleDict({
                'shortcut': conv_bn(in_channels, concat_channels, num_row=1, num_col=1, activation=False),
                'conv3x3': conv_bn(in_channels, out_3x3, num_row=3, num_col=3),
                'conv5x5': nn.Sequential(
                    conv_bn(out_3x3, out_5x5, num_row=3, num_col=3),
                    conv_bn(out_5x5, out_5x5, num_row=3, num_col=3)
                ),
                'conv7x7': nn.Sequential(
                    conv_bn(out_5x5, out_7x7, num_row=3, num_col=3),
                    conv_bn(out_7x7, out_7x7, num_row=3, num_col=3),
                    conv_bn(out_7x7, out_7x7, num_row=3, num_col=3)
                ),
                'bn_concat': nn.BatchNorm1d(concat_channels),
                'final_bn': nn.BatchNorm1d(concat_channels),
                'proj': nn.Conv1d(concat_channels, U, 1)  # Project back to U channels
            })

        def ResPath(filters, length):
            layers = []
            for i in range(length):
                layers.append(nn.ModuleDict({
                    'shortcut': conv_bn(filters, filters, num_row=1, num_col=1, activation=False),
                    'conv': conv_bn(filters, filters, num_row=3, num_col=3),
                    'bn': nn.BatchNorm1d(filters)
                }))
            return nn.ModuleList(layers)

        # Encoder path
        self.mresblock1 = MultiResBlock(32, n_channel)
        self.pool1 = nn.MaxPool1d(2)
        self.respath1 = ResPath(32, 4)

        self.mresblock2 = MultiResBlock(64, 32)
        self.pool2 = nn.MaxPool1d(2)
        self.respath2 = ResPath(64, 3)

        self.mresblock3 = MultiResBlock(128, 64)
        self.pool3 = nn.MaxPool1d(2)
        self.respath3 = ResPath(128, 2)

        self.mresblock4 = MultiResBlock(256, 128)
        self.pool4 = nn.MaxPool1d(2)
        self.respath4 = ResPath(256, 1)

        self.mresblock5 = MultiResBlock(512, 256)

        # Decoder path with correct channel counts for BatchNorm
        self.up6 = TransConvBN(512, 256)  # mres5(512) + mres4(256) channels
        self.mresblock6 = MultiResBlock(256, 512 + 256)

        self.up7 = TransConvBN(256, 128)  # mres6(256) + mres3(128) channels
        self.mresblock7 = MultiResBlock(128, 256 + 128)

        self.up8 = TransConvBN(128, 64)   # mres7(128) + mres2(64) channels
        self.mresblock8 = MultiResBlock(64, 128 + 64)

        self.up9 = TransConvBN(64, 32)    # mres8(64) + mres1(32) channels
        self.mresblock9 = MultiResBlock(32, 64 + 32)

        self.conv10 = nn.Conv1d(32, 1, 1)

    def forward(self, x):
        def mres_forward(block, x):
            shortcut = block['shortcut'](x)
            
            conv3x3 = block['conv3x3'](x)
            conv5x5 = block['conv5x5'](conv3x3)
            conv7x7 = block['conv7x7'](conv5x5)
            
            concat = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)  # Channel dimension
            concat = block['bn_concat'](concat)
            
            out = concat + shortcut
            out = torch.relu(out)
            out = block['final_bn'](out)
            out = block['proj'](out)  # Project to U channels before ResPath
            
            return out

        def respath_forward(path, x):
            out = x
            for block in path:
                shortcut = block['shortcut'](out)
                
                conv = block['conv'](out)
                
                out = conv + shortcut
                out = torch.relu(out)
                out = block['bn'](out)
                
            return out

        # Encoder path with direct ResPath assignment like Keras
        mres1 = mres_forward(self.mresblock1, x)
        pool1 = self.pool1(mres1)
        mres1 = respath_forward(self.respath1, mres1)  # Direct assignment

        mres2 = mres_forward(self.mresblock2, pool1)
        pool2 = self.pool2(mres2)
        mres2 = respath_forward(self.respath2, mres2)  # Direct assignment

        mres3 = mres_forward(self.mresblock3, pool2)
        pool3 = self.pool3(mres3)
        mres3 = respath_forward(self.respath3, mres3)  # Direct assignment

        mres4 = mres_forward(self.mresblock4, pool3)
        pool4 = self.pool4(mres4)
        mres4 = respath_forward(self.respath4, mres4)  # Direct assignment

        mres5 = mres_forward(self.mresblock5, pool4)

        # Decoder path matching Keras order
        up6 = self.up6.up(mres5)  # Just upsample
        merge6 = torch.cat([up6, mres4], dim=1)  # Concatenate
        merge6 = self.up6.bn(merge6)  # BatchNorm after concatenation
        mres6 = mres_forward(self.mresblock6, merge6)

        up7 = self.up7.up(mres6)
        merge7 = torch.cat([up7, mres3], dim=1)
        merge7 = self.up7.bn(merge7)
        mres7 = mres_forward(self.mresblock7, merge7)

        up8 = self.up8.up(mres7)
        merge8 = torch.cat([up8, mres2], dim=1)
        merge8 = self.up8.bn(merge8)
        mres8 = mres_forward(self.mresblock8, merge8)

        up9 = self.up9.up(mres8)
        merge9 = torch.cat([up9, mres1], dim=1)
        merge9 = self.up9.bn(merge9)
        mres9 = mres_forward(self.mresblock9, merge9)

        out = self.conv10(mres9)
        return out