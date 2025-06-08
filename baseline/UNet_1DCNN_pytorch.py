# Standard library imports
# None

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """1D Convolutional Block"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class TransConv1D(nn.Module):
    """1D Transposed Convolutional Block"""
    def __init__(self, in_channels, out_channels):
        super(TransConv1D, self).__init__()
        self.trans_conv = nn.ConvTranspose1d(in_channels, out_channels, 2, stride=2, padding=0)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn(self.trans_conv(x)))

class FeatureExtractionBlock(nn.Module):
    """Feature Extraction Block for the AutoEncoder Mode"""
    def __init__(self, input_size, feature_number, model_width):
        super(FeatureExtractionBlock, self).__init__()
        self.model_width = model_width
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, feature_number)
        self.fc2 = nn.Linear(feature_number, input_size)
        
    def forward(self, x, feature_extraction_only=False):
        shape = x.shape
        x = self.flatten(x)
        x = self.fc1(x)
        
        # Return features directly if feature_extraction_only is True
        if feature_extraction_only:
            return x
            
        # Otherwise continue with normal behavior
        x = self.fc2(x)
        return x.view(shape[0], self.model_width, -1)

class AttentionBlock(nn.Module):
    """Attention Block"""
    def __init__(self, in_channels, num_filters, is_transconv=True):
        super(AttentionBlock, self).__init__()
        # First conv for skip connection
        self.conv1x1_1 = nn.Conv1d(in_channels, num_filters, 1, stride=2)
        # Second conv for gating signal - adjust based on upsampling method
        gating_channels = in_channels if is_transconv else in_channels * 2
        self.conv1x1_2 = nn.Conv1d(gating_channels, num_filters, 1, stride=2)
        self.conv_final = nn.Conv1d(num_filters, 1, 1, stride=1)
        
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.bn_final = nn.BatchNorm1d(1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.up_conv = UpConvBlock()
        self.trans_conv = TransConv1D(1, 1)
        
    def forward(self, skip_connection, gating_signal):
        # Process skip connection
        conv1x1_1 = self.conv1x1_1(skip_connection)
        conv1x1_1 = self.bn1(conv1x1_1)
        
        # Process gating signal
        conv1x1_2 = self.conv1x1_2(gating_signal)
        conv1x1_2 = self.bn2(conv1x1_2)
        
        # Add the features
        conv1_2 = conv1x1_1 + conv1x1_2
        
        conv1_2 = self.relu(conv1_2)
        conv1_2 = self.conv_final(conv1_2)
        conv1_2 = self.bn_final(conv1_2)
        
        conv1_2 = self.sigmoid(conv1_2)
        
        # Upsample back to original size
        resampler1 = self.up_conv(conv1_2)
        
        resampler2 = self.trans_conv(conv1_2)
        
        resampler = resampler1 + resampler2
        
        result = skip_connection * resampler
        
        return result

class ConcatBlock(nn.Module):
    """Concatenation Block"""
    def __init__(self):
        super(ConcatBlock, self).__init__()
        
    def forward(self, input1, *argv):
        cat = input1
        for arg in argv:
            cat = torch.cat([cat, arg], dim=1)
        return cat

class UpConvBlock(nn.Module):
    """1D UpSampling Block"""
    def __init__(self, size=2):
        super(UpConvBlock, self).__init__()
        self.size = size
        
    def forward(self, inputs):
        return F.interpolate(inputs, scale_factor=self.size, mode='nearest')

class ConvLSTM1D(nn.Module):
    """1D Convolutional LSTM"""
    def __init__(self, input_size, hidden_size, kernel_size=3, padding='same', go_backwards=True):
        super(ConvLSTM1D, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.go_backwards = go_backwards
        
        # Input gate
        self.conv_i = nn.Conv1d(input_size, hidden_size, kernel_size, padding=padding)
        self.bn_i = nn.BatchNorm1d(hidden_size)
        
        # Forget gate
        self.conv_f = nn.Conv1d(input_size, hidden_size, kernel_size, padding=padding)
        self.bn_f = nn.BatchNorm1d(hidden_size)
        
        # Cell gate
        self.conv_c = nn.Conv1d(input_size, hidden_size, kernel_size, padding=padding)
        self.bn_c = nn.BatchNorm1d(hidden_size)
        
        # Output gate
        self.conv_o = nn.Conv1d(input_size, hidden_size, kernel_size, padding=padding)
        self.bn_o = nn.BatchNorm1d(hidden_size)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        
        # Initialize hidden state and cell state
        h_t = torch.zeros(batch_size, self.hidden_size, seq_len).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size, seq_len).to(x.device)
        
        # Reverse sequence if go_backwards
        if self.go_backwards:
            x = torch.flip(x, [2])
            
        # Gates
        i = self.sigmoid(self.bn_i(self.conv_i(x)))  # Input gate
        f = self.sigmoid(self.bn_f(self.conv_f(x)))  # Forget gate
        c = self.tanh(self.bn_c(self.conv_c(x)))     # Cell gate
        o = self.sigmoid(self.bn_o(self.conv_o(x)))  # Output gate
        
        # Update cell state
        c_t = f * c_t + i * c
        # Update hidden state
        h_t = o * self.tanh(c_t)
        
        return h_t

class UNet(nn.Module):
    def __init__(self, length, model_depth, num_channel, model_width, kernel_size, 
                 problem_type='Regression', output_nums=1, ds=1, ae=0, ag=0, lstm=0,
                 alpha=1, feature_number=1024, is_transconv=True):
        super(UNet, self).__init__()
        
        # Store other configuration parameters
        self.length = length
        self.model_depth = model_depth
        self.num_channel = num_channel
        self.model_width = model_width
        self.kernel_size = kernel_size
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.D_S = ds
        self.A_E = ae
        self.A_G = ag
        self.LSTM = lstm
        self.alpha = alpha
        self.feature_number = feature_number
        self.is_transconv = is_transconv
        
        # Input validation
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        # Initialize encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for i in range(1, self.model_depth + 1):
            in_channels = num_channel if i == 1 else model_width * (2 ** (i-2))
            out_channels = model_width * (2 ** (i-1))
            double_conv = nn.Sequential(
                ConvBlock(in_channels, out_channels, kernel_size),
                ConvBlock(out_channels, out_channels, kernel_size)
            )
            self.encoder_blocks.append(double_conv)
            
        self.pool = nn.MaxPool1d(2)
        
        # Bottleneck channels should match the last encoder block output
        bottleneck_in_channels = model_width * (2 ** (model_depth-1))
        bottleneck_out_channels = model_width * (2 ** model_depth)
        
        # AutoEncoder feature extraction
        if self.A_E:
            # Calculate the correct input size for the feature extraction
            input_size = bottleneck_in_channels * (length // (2 ** model_depth))
            self.feature_extraction = FeatureExtractionBlock(
                input_size, 
                feature_number, 
                bottleneck_in_channels
            )
            
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(bottleneck_in_channels, bottleneck_out_channels, kernel_size),
            ConvBlock(bottleneck_out_channels, bottleneck_out_channels, kernel_size)
        )
        
        # Initialize decoder components
        self.decoder_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.ds_convs = nn.ModuleList() if self.D_S else None
        self.attention_blocks = nn.ModuleList() if self.A_G else None
        self.lstm_layers = nn.ModuleList() if self.LSTM else None
        
        # Utility blocks
        self.concat_block = ConcatBlock()
            
        for i in range(model_depth):
            # Calculate channels
            in_channels = model_width * (2 ** (model_depth-i))
            out_channels = model_width * (2 ** (model_depth-i-1))
            
            # 1. Deep Supervision
            if self.D_S:
                self.ds_convs.append(nn.Conv1d(in_channels, 1, 1))
                
            # 2. Upsampling blocks
            if is_transconv:
                self.up_blocks.append(TransConv1D(in_channels, out_channels))
            else:
                self.up_blocks.append(UpConvBlock())
            
            # 3. Attention
            if self.A_G:
                self.attention_blocks.append(
                    AttentionBlock(out_channels, out_channels, is_transconv=is_transconv)
                )
            
            # 4. LSTM
            if self.LSTM:
                lstm_in_channels = in_channels + out_channels if not is_transconv else out_channels * 2
                self.lstm_layers.append(
                    ConvLSTM1D(
                        input_size=lstm_in_channels,  # Adjusted based on upsampling method
                        hidden_size=out_channels,
                        kernel_size=3,
                        padding='same',
                        go_backwards=True
                    )
                )
            
            # 5. Decoder Convolutions
            if is_transconv:
                decoder_in_channels = out_channels * 2  # out_channels from TransConv + out_channels from skip
            else:
                decoder_in_channels = in_channels + out_channels  # in_channels from upsampled + out_channels from skip

            if self.LSTM:
                decoder_in_channels = out_channels  # LSTM output channels

            self.decoder_blocks.append(nn.Sequential(
                ConvBlock(decoder_in_channels, out_channels, kernel_size),
                ConvBlock(out_channels, out_channels, kernel_size)
            ))

        # Output layer (moved after decoder part)
        self.final_conv = nn.Conv1d(model_width, output_nums, 1)
        self.final_activation = nn.Softmax(dim=1) if problem_type == 'Classification' else nn.Identity()

    def forward(self, x, feature_extraction_only=False):        
        # Validate feature extraction settings
        if feature_extraction_only and not self.A_E:
            raise ValueError("feature_extraction_only requires ae (AutoEncoder) to be enabled")
        
        # Store skip connections
        skips = []
        
        # Encoder path
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            skips.append(x)
            x = self.pool(x)
            
        # AutoEncoder feature extraction
        if self.A_E:
            x = self.feature_extraction(x, feature_extraction_only=feature_extraction_only)
            # Return early if only feature extraction is needed
            if feature_extraction_only:
                return x
                
        # Bottleneck
        x = self.bottleneck(x)
        
        # Store deep supervision outputs
        ds_outputs = []
        
        # Decoder path
        skips = skips[::-1]  # Reverse for easier access
        
        for i in range(self.model_depth):
            # 1. Deep supervision
            if self.D_S:
                ds_outputs.append(self.ds_convs[i](x))
                
            # 2. Upsampling
            x = self.up_blocks[i](x)
            
            # 3. Get skip connection
            skip = skips[i]
            
            # 4. Apply attention if enabled
            if self.A_G:
                skip = self.attention_blocks[i](skip, x)
            
            # 5. LSTM if enabled
            if self.LSTM:
                combined = self.concat_block(x, skip)
                x = self.lstm_layers[i](combined)
            else:
                x = self.concat_block(x, skip)
            
            # 6. Convolutions
            x = self.decoder_blocks[i](x)
        
        # Final output
        x = self.final_conv(x)
        outputs = self.final_activation(x)
        
        if self.D_S:
            ds_outputs.append(outputs)
            return ds_outputs[::-1]
            
        return outputs