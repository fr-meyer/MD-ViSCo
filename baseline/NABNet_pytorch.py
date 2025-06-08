# Standard library imports

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F

def Conv_Block(in_channels, out_channels, kernel_size, padding='same'):
    """1D Convolutional Block"""
    if padding == 'same':
        padding = kernel_size // 2
        
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    )

def trans_conv1D(in_channels, out_channels):
    """1D Transposed Convolutional Block"""
    return nn.Sequential(
        nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    )

class Feature_Extraction_Block(nn.Module):
    """Feature Extraction Block for AutoEncoder Mode"""
    def __init__(self, in_channels, feature_number, seq_length):
        super().__init__()
        self.in_channels = in_channels
        self.seq_length = seq_length
        
        # Calculate input features size
        self.input_size = in_channels * seq_length
        print(f"Feature extraction block initialized with: channels={in_channels}, length={seq_length}")
        
        self.net = nn.Sequential(
            nn.Flatten(),  # Flatten all dimensions except batch
            nn.Linear(self.input_size, feature_number),
            nn.ReLU(),
            nn.Linear(feature_number, self.input_size)
        )
        
    def to(self, device):
        """Moves all model parameters to the given device"""
        super().to(device)
        self.net = self.net.to(device)
        return self
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Ensure proper flattening and device placement
        x = x.reshape(batch_size, -1)  # Flatten to (batch_size, channels*length)
        if x.size(1) != self.input_size:
            raise ValueError(
                f"Input size mismatch. Got tensor with {x.size(1)} features, "
                f"expected {self.input_size} features. Input shape was {x.shape}"
            )
            
        x = self.net(x)
        return x.reshape(batch_size, self.in_channels, self.seq_length)

class Attention_Block(nn.Module):
    def __init__(self, channels, multiplier):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels * multiplier, 1, stride=1)
        self.bn1 = nn.BatchNorm1d(channels * multiplier)
        self.conv2 = nn.Conv1d(channels, channels * multiplier, 1, stride=1)
        self.bn2 = nn.BatchNorm1d(channels * multiplier)
        self.conv3 = nn.Conv1d(channels * multiplier, 1, 1)
        self.bn3 = nn.BatchNorm1d(1)
        
    def forward(self, skip, gating):
        # Ensure same spatial dimensions
        if skip.size(2) != gating.size(2):
            gating = F.interpolate(gating, size=skip.size(2), mode='linear', align_corners=False)
            
        x1 = self.bn1(self.conv1(skip))
        x2 = self.bn2(self.conv2(gating))
        x = F.relu(x1 + x2)
        x = self.bn3(self.conv3(x))
        x = torch.sigmoid(x)
        return skip * x

class Attention_LSTM_Block(nn.Module):
    def __init__(self, channels, multiplier, lstm_multiplier):
        super().__init__()
        self.channels = channels
        self.lstm_hidden = int(channels * lstm_multiplier)
        
        self.lstm_skip = nn.LSTM(channels, self.lstm_hidden, 
                                bidirectional=True, batch_first=True)
        self.lstm_up = nn.LSTM(channels, self.lstm_hidden, 
                              bidirectional=True, batch_first=True)
        
        # Adjust attention to match LSTM output dimensions
        self.attention = nn.MultiheadAttention(self.lstm_hidden * 2, 1)
        
        # Adjust output projection to match input channels
        self.out_proj = nn.Sequential(
            nn.Linear(self.lstm_hidden * 4, channels),
            nn.ReLU()
        )
        
    def to(self, device):
        """Moves all model parameters to the given device"""
        super().to(device)
        self.lstm_skip = self.lstm_skip.to(device)
        self.lstm_up = self.lstm_up.to(device)
        self.attention = self.attention.to(device)
        self.out_proj = self.out_proj.to(device)
        return self

    def forward(self, skip, up):
        batch_size = skip.size(0)
        seq_len = skip.size(2)
        
        # Transpose for LSTM (batch, channels, length) -> (batch, length, channels)
        skip = skip.transpose(1, 2)
        up = up.transpose(1, 2)
        
        # Process through LSTMs
        skip_feat, _ = self.lstm_skip(skip)  # Output: [batch, seq_len, lstm_hidden*2]
        up_feat, _ = self.lstm_up(up)        # Output: [batch, seq_len, lstm_hidden*2]
        
        # Apply attention
        attn_out, _ = self.attention(skip_feat, up_feat, up_feat)
        
        # Combine features
        combined = torch.cat([attn_out, skip_feat], dim=-1)  # [batch, seq_len, lstm_hidden*4]
        
        # Project back to original channel dimension
        output = self.out_proj(combined)  # [batch, seq_len, channels]
        
        # Transpose back to channel-first format
        output = output.transpose(1, 2)  # [batch, channels, seq_len]
        
        return output

class NABNet(nn.Module):
    def __init__(self, length, model_depth, num_channel, model_width, kernel_size,
                 problem_type='Regression', output_nums=1, ds=1, ae=0, ag=0, lstm=0,
                 feature_number=1024, is_transconv=True):
        super().__init__()
        self.model_depth = model_depth
        self.D_S = ds
        self.A_E = ae
        self.A_G = ag
        self.LSTM = lstm
        
        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        in_channels = num_channel
        
        for i in range(model_depth):
            out_channels = model_width * (2 ** i)
            self.encoder_blocks.append(
                nn.Sequential(
                    Conv_Block(in_channels, out_channels, kernel_size),
                    Conv_Block(out_channels, out_channels, kernel_size)
                )
            )
            in_channels = out_channels
            
        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Bridge
        bridge_channels = model_width * (2 ** model_depth)
        self.bridge = nn.Sequential(
            Conv_Block(model_width * (2 ** (model_depth - 1)), bridge_channels, kernel_size),
            Conv_Block(bridge_channels, bridge_channels, kernel_size)
        )
        
        # Decoder path with transposed convolution
        self.decoder_blocks = nn.ModuleList()
        for i in range(model_depth):
            in_channels = model_width * (2 ** (model_depth - i))
            out_channels = model_width * (2 ** (model_depth - i - 1))
            self.decoder_blocks.append(trans_conv1D(in_channels, out_channels))
            
        # Attention blocks if enabled
        if ag:
            self.attention_blocks = nn.ModuleList()
            # Initialize attention blocks in reverse order to match decoder path
            for i in range(model_depth - 1, -1, -1):
                channels = model_width * (2 ** i)
                self.attention_blocks.append(Attention_Block(channels, 2))
                
        # Final convolution
        self.final_conv = nn.Conv1d(model_width, output_nums, 1)
        self.final_activation = nn.Identity() if problem_type == 'Regression' else nn.Sigmoid()
        
        # Initialize feature extraction if autoencoder mode is enabled
        if ae:
            bottleneck_channels = model_width * (2 ** (model_depth - 1))
            bottleneck_length = length // (2 ** model_depth)
            
            self.feature_extraction = Feature_Extraction_Block(
                bottleneck_channels,
                feature_number,
                bottleneck_length
            )
            
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
            self.feature_extraction.apply(init_weights)
        
        # Initialize decoder conv blocks
        self.decoder_conv_blocks = nn.ModuleList()
        for i in range(model_depth):
            curr_width = model_width * (2 ** (model_depth - i - 1))
            in_channels = curr_width * 2  # *2 for concatenation
            
            self.decoder_conv_blocks.append(
                nn.Sequential(
                    Conv_Block(in_channels, curr_width, kernel_size),
                    Conv_Block(curr_width, curr_width, kernel_size)
                )
            )
        
        # Initialize deep supervision convs
        if ds:
            self.ds_convs = nn.ModuleList([
                nn.Conv1d(model_width * (2 ** i), output_nums, 1)
                for i in range(model_depth)
            ][::-1])  # Reverse order to match decoder path

    def forward(self, x):
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoding path
        for block in self.encoder_blocks:
            x = block(x)
            encoder_outputs.append(x)
            x = self.pool(x)
            
        # Bridge
        x = self.bridge(x)
        
        # Decoding path
        encoder_outputs.reverse()
        decoder_outputs = []
        
        for i in range(self.model_depth):
            # Upsampling
            x = self.decoder_blocks[i](x)
            enc_feat = encoder_outputs[i]
            
            if self.A_G:
                enc_feat = self.attention_blocks[i](enc_feat, x)
                
            # Concatenate and apply convolutions
            x = torch.cat([x, enc_feat], dim=1)
            x = self.decoder_conv_blocks[i](x)
            
            if self.D_S:
                decoder_outputs.append(self.ds_convs[i](x))
        
        # Output
        x = self.final_conv(x)
        x = self.final_activation(x)
        
        if self.D_S:
            decoder_outputs.append(x)
            return decoder_outputs[::-1]  # Reverse order to match TF implementation
        return x

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)

class NABNetV2(NABNet):
    """NABNet Version 2 implementation"""
    def __init__(self, length, model_depth, num_channel, model_width, kernel_size,
                 problem_type='Regression', output_nums=1, ds=1, ae=0, ag=0, lstm=0,
                 feature_number=1024, is_transconv=True):
        super().__init__(length, model_depth, num_channel, model_width, kernel_size,
                        problem_type, output_nums, ds, ae, ag, lstm, feature_number, is_transconv)
        
        # Initialize feature extraction block if autoencoder mode is enabled
        if ae:
            # Calculate bottleneck dimensions - use the encoder output size
            bottleneck_channels = model_width * (2 ** (model_depth - 1))  # Last encoder block channels
            bottleneck_length = length // (2 ** model_depth)
            
            self.feature_extraction = Feature_Extraction_Block(
                bottleneck_channels,
                feature_number,
                bottleneck_length
            )
            
            # Ensure feature extraction block is properly initialized
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
            self.feature_extraction.apply(init_weights)
        
        # Initialize additional conv blocks for decoder path
        self.decoder_conv_blocks = nn.ModuleList()
        for i in range(model_depth):
            # Calculate channels for each decoder level
            curr_width = model_width * (2 ** (model_depth - i - 1))
            in_channels = curr_width * 2  # *2 for concatenation
            
            self.decoder_conv_blocks.append(
                nn.Sequential(
                    Conv_Block(in_channels, curr_width, kernel_size),
                    Conv_Block(curr_width, curr_width, kernel_size)
                )
            )
        
        # Initialize output convs for deep supervision
        if ds:
            self.ds_convs = nn.ModuleList([
                nn.Conv1d(model_width * (2 ** i), output_nums, 1)
                for i in range(model_depth)
            ][::-1])  # Reverse order to match decoder path

    def forward(self, x):
        # Store encoder outputs for skip connections
        encoder_outputs = []
        device = x.device
        
        # Encoding path
        for block in self.encoder_blocks:
            x = block(x)
            encoder_outputs.append(x)
            x = self.pool(x)
            
        # AutoEncoder feature extraction if enabled
        if self.A_E:
            self.feature_extraction = self.feature_extraction.to(device)
            x = self.feature_extraction(x)
            
        # Bridge
        x = self.bridge(x)
        
        # Decoding path
        encoder_outputs.reverse()
        decoder_outputs = []
        
        for i in range(self.model_depth):
            # Upsampling
            x = self.decoder_blocks[i](x)
            enc_feat = encoder_outputs[i]
            
            if self.A_G:
                # Create attention block with matching dimensions
                attention_block = Attention_LSTM_Block(
                    enc_feat.size(1),  # Use encoder feature channels
                    1,                 # No multiplier needed now
                    1
                ).to(device)
                enc_feat = attention_block(enc_feat, x)
            
            # Concatenate and apply convolutions
            x = torch.cat([x, enc_feat], dim=1)
            x = self.decoder_conv_blocks[i](x)
            
            if self.D_S:
                decoder_outputs.append(self.ds_convs[i](x))
        
        # Output
        x = self.final_conv(x)
        x = self.final_activation(x)
        
        if self.D_S:
            decoder_outputs.append(x)
            return decoder_outputs[::-1]  # Reverse order to match TF implementation
        return x
