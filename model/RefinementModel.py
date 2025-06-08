# Standard library imports
from typing import Optional

# Third-party imports
import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    DistilBertConfig,
    DistilBertModel,
    PatchTSMixerConfig,
    PatchTSMixerModel
)

class Mlp_BP(nn.Module):
    """Multi-layer perceptron (MLP) for blood pressure prediction.
    
    This class implements a two-layer MLP with dropout and activation functions,
    specifically designed for blood pressure prediction tasks. The network can be
    configured with different input, hidden, and output dimensions.
    
    Architecture:
        Input -> Dropout -> Linear1 -> Activation -> Dropout -> Linear2 -> Output
    
    Args:
        in_features (int): Number of input features
        hidden_features (int, optional): Number of hidden features. If None, uses in_features
        out_features (int, optional): Number of output features. If None, uses in_features
        act_layer (nn.Module, optional): Activation function to use. Defaults to nn.GELU
        drop (float, optional): Dropout probability. Defaults to 0.0
    
    Example:
        >>> model = Mlp_BP(in_features=512, hidden_features=256, out_features=1)
        >>> x = torch.randn(32, 512)  # batch_size=32, features=512
        >>> output = model(x)  # shape: (32, 1)
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # Set default values if not provided
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # First fully connected layer
        self.fc1 = nn.Linear(in_features, hidden_features)
        # Activation function (default: GELU)
        self.act = act_layer()
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout layer for regularization
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Apply dropout to input
        x = self.drop(x)
        # First linear transformation
        x = self.fc1(x)
        # Apply activation function
        x = self.act(x)
        # Apply dropout after activation
        x = self.drop(x)
        # Final linear transformation
        x = self.fc2(x)
        return x

class WaveformEncoder(nn.Module):
    """Encoder for processing waveform data (ECG/PPG signals).
    
    This class implements a waveform encoder using the PatchTSMixer architecture,
    which is specifically designed for processing time series data like ECG and PPG signals.
    The encoder can be configured with different context lengths and model dimensions.
    
    Architecture:
        Uses PatchTSMixer model with configurable:
        - Context length (sequence length)
        - Number of input channels
        - Model dimension
        - Number of layers
        - Expansion factor
    
    Args:
        model_name (str, optional): Name of the model. Defaults to None
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True
        trainable (bool, optional): Whether to make model parameters trainable. Defaults to True
        context_length (int, optional): Length of input sequences. Defaults to 1280
        num_input_channels (int, optional): Number of input channels. Defaults to 1
        d_model (int, optional): Dimension of the model. Defaults to 64
    
    Example:
        >>> encoder = WaveformEncoder(context_length=1024, d_model=64)
        >>> x = torch.randn(32, 1, 1024)  # batch_size=32, channels=1, seq_len=1024
        >>> output = encoder(x)  # Returns PatchTSMixer output
    """
    def __init__(self, model_name=None, pretrained=True, 
                 trainable=True, context_length=1280, num_input_channels=1, d_model=64):
        super().__init__()
        # Initialize PatchTSMixer model with specified configuration
        self.model = PatchTSMixerModel(PatchTSMixerConfig(
            context_length=context_length,  # Length of input sequences
            num_input_channels=num_input_channels,  # Number of input channels
            d_model=d_model,  # Model dimension
            num_layers=15,  # Number of transformer layers
            expansion_factor=5  # Expansion factor for feed-forward networks
        ))
        
        # Configure parameter trainability
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        """Forward pass through the waveform encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_input_channels, context_length)
            
        Returns:
            torch.Tensor: Output from the PatchTSMixer model
        """
        return self.model(x)

class TextEncoder(nn.Module):
    """Encoder for processing text data using DistilBERT.
    
    This class implements a text encoder using the DistilBERT architecture,
    which is a distilled version of BERT that maintains most of the performance
    while being more efficient. The encoder processes text input and extracts
    contextual embeddings.
    
    Architecture:
        Uses DistilBERT model with configurable:
        - Model name/version
        - Pretrained weights
        - Parameter trainability
    
    Args:
        model_name (str, optional): Name of the pretrained model to use.
            Defaults to "distilbert-base-uncased"
        pretrained (bool, optional): Whether to use pretrained weights.
            Defaults to True
        trainable (bool, optional): Whether to make model parameters trainable.
            Defaults to True
    
    Example:
        >>> encoder = TextEncoder(pretrained=True)
        >>> input_ids = torch.randint(0, 1000, (32, 128))  # batch_size=32, seq_len=128
        >>> attention_mask = torch.ones(32, 128)
        >>> output = encoder(input_ids, attention_mask)  # shape: (32, 768)
    """
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=True):
        super().__init__()
        # Initialize DistilBERT model with specified configuration
        if pretrained:
            # Load pretrained model if specified
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            # Initialize model with default configuration
            self.model = DistilBertModel(config=DistilBertConfig())
        
        # Index of the target token to extract from the output
        # Using [CLS] token (index 0) as the default representation
        self.target_token_idx = 0
        
        # Configure parameter trainability
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, input_ids, attention_mask):
        """Forward pass through the text encoder.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, sequence_length)
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, sequence_length)
                where 1 indicates tokens to attend to and 0 indicates padding tokens
            
        Returns:
            torch.Tensor: Extracted embeddings from the target token position
                of shape (batch_size, hidden_size)
        """
        # Process input through DistilBERT model
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Extract embeddings from the target token position (default: [CLS] token)
        return output.last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    """Projection head for transforming embeddings to a common space.
    
    This class implements a projection head that transforms input embeddings into
    a common representation space. It uses a combination of linear layers,
    activation functions, dropout, and layer normalization to create robust
    embeddings suitable for downstream tasks.
    
    Architecture:
        Input -> Linear -> GELU -> Linear -> Dropout -> Residual -> LayerNorm -> Output
    
    The architecture includes:
        - Initial linear projection
        - GELU activation
        - Second linear transformation
        - Dropout for regularization
        - Residual connection
        - Layer normalization
    
    Args:
        embedding_dim (int): Dimension of input embeddings
        projection_dim (int, optional): Dimension of projected embeddings.
            Defaults to 256
        dropout (float, optional): Dropout probability for regularization.
            Defaults to 0.1
    
    Example:
        >>> projection = ProjectionHead(embedding_dim=768, projection_dim=256)
        >>> x = torch.randn(32, 768)  # batch_size=32, embedding_dim=768
        >>> output = projection(x)  # shape: (32, 256)
    """
    def __init__(self, embedding_dim, projection_dim=256, dropout=0.1):
        super().__init__()
        # Initial linear projection layer
        self.projection = nn.Linear(embedding_dim, projection_dim)
        # GELU activation function
        self.gelu = nn.GELU()
        # Second linear transformation
        self.fc = nn.Linear(projection_dim, projection_dim)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        # Layer normalization for stabilizing training
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        """Forward pass through the projection head.
        
        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, embedding_dim)
            
        Returns:
            torch.Tensor: Projected embeddings of shape (batch_size, projection_dim)
                with residual connection and layer normalization applied
        """
        # Initial projection
        projected = self.projection(x)
        # Apply GELU activation
        x = self.gelu(projected)
        # Second linear transformation
        x = self.fc(x)
        # Apply dropout
        x = self.dropout(x)
        # Add residual connection
        x = x + projected
        # Apply layer normalization
        x = self.layer_norm(x)
        return x

class BPModel(nn.Module):
    """Unified blood pressure estimation model.
    
    This class implements a comprehensive blood pressure estimation model that can
    process both waveform data (ECG/PPG) and patient information to predict blood
    pressure values. The model can operate in two modes:
    1. With patient information (PI): Uses both waveform and text data
    2. Without patient information: Uses only waveform data
    
    Architecture Components:
        - Waveform Encoders: Process ECG and PPG signals
        - Text Encoder: Process patient information (when PI is enabled)
        - Projection Heads: Transform embeddings to common space
        - BP Prediction Heads: Predict SBP and DBP values
        - Loss Functions: L1 loss and weighted contrastive loss (WCL)
    
    The model uses a combination of:
        - PatchTSMixer for waveform processing
        - DistilBERT for text processing
        - Multi-layer perceptrons for BP prediction
        - Weighted contrastive learning for better representations
    
    Args:
        pi (bool, optional): Whether to use patient information. Defaults to True
        temperature (float, optional): Temperature parameter for contrastive loss.
            Defaults to 4.0
        image_embedding (int, optional): Size of image embedding dimension.
            Defaults to None (computed from d_model and w_length)
        text_embedding (int, optional): Size of text embedding dimension.
            Defaults to 768 (DistilBERT default)
        wcl_age_threshold (float, optional): Threshold for age in weighted contrastive loss.
            Defaults to 0.2231
        wcl (bool, optional): Whether to use weighted contrastive loss.
            Defaults to True
        w_length (int, optional): Length of waveform input.
            Defaults to 1024
        normalized_bp (bool, optional): Whether BP values are normalized.
            Defaults to True
        projection_dim (int, optional): Dimension of projected embeddings.
            Defaults to 512
        dropout (float, optional): Dropout probability for regularization.
            Defaults to 0.1
        d_model (int, optional): Dimension of the model.
            Defaults to 64
    
    Example:
        >>> model = BPModel(pi=True, wcl=True)
        >>> batch = {
        ...     "ecg": torch.randn(32, 1, 1024),
        ...     "ppg": torch.randn(32, 1, 1024),
        ...     "input_ids": torch.randint(0, 1000, (32, 128)),
        ...     "attention_mask": torch.ones(32, 128)
        ... }
        >>> ecg_loss, ppg_loss = model(batch)
    """
    def __init__(self, 
                 pi: bool = True,
                 temperature: float = 4.0,
                 image_embedding: Optional[int] = None,
                 text_embedding: int = 768,  # Default DistilBERT embedding size
                 wcl_age_threshold: Optional[float] = 0.2231,
                 wcl: bool = True,
                 w_length: int = 1024,
                 normalized_bp: bool = True,
                 projection_dim: int = 512,
                 dropout: float = 0.1,
                 d_model: int = 64
                 ) -> None:
        super().__init__()
            
        # Store configuration parameters
        self.pi = pi  # Whether to use patient information
        self.wcl = wcl  # Whether to use weighted contrastive loss
        self.temperature = temperature  # Temperature for contrastive loss
        self.normalized_bp = normalized_bp  # Whether BP values are normalized

        # Calculate image embedding size based on model dimension and waveform length
        image_embedding = (d_model * w_length) // 8
            
        # Initialize waveform encoders based on whether using patient information
        if pi:
            # With patient information: Initialize encoders for combined processing
            self.ppg_encoder = WaveformEncoder(context_length=w_length, d_model=d_model)
            self.ecg_encoder = WaveformEncoder(context_length=w_length, d_model=d_model)
            self.text_encoder = TextEncoder()
            self.ppg_text_encoder = WaveformEncoder(context_length=projection_dim, num_input_channels=2)
            self.ecg_text_encoder = WaveformEncoder(context_length=projection_dim, num_input_channels=2)
        else:
            # Without patient information: Initialize encoders for direct processing
            self.ppg_encoder = WaveformEncoder(context_length=w_length, d_model=d_model)
            self.ppg_bp_encoder = WaveformEncoder(context_length=projection_dim, d_model=d_model)
            self.ecg_encoder = WaveformEncoder(context_length=w_length, d_model=d_model)
            self.ecg_bp_encoder = WaveformEncoder(context_length=projection_dim, d_model=d_model)

        # Initialize projection heads for transforming embeddings
        self.ppg_projection = ProjectionHead(embedding_dim=image_embedding, 
                                           projection_dim=projection_dim, 
                                           dropout=dropout)
        self.ecg_projection = ProjectionHead(embedding_dim=image_embedding, 
                                           projection_dim=projection_dim, 
                                           dropout=dropout)
        
        # Initialize text projection head if using patient information
        if pi:
            self.text_projection = ProjectionHead(embedding_dim=text_embedding, 
                                                projection_dim=projection_dim, 
                                                dropout=dropout)

        # Calculate input features size for BP prediction heads
        # Different sizes for PI and non-PI modes
        in_features = (projection_dim * 2 * d_model) // 8 if pi else (projection_dim * d_model) // 8
        
        # Initialize BP prediction heads
        self.ppg_SBP_head = Mlp_BP(in_features=in_features, 
                                  hidden_features=in_features // 2, 
                                  out_features=1, 
                                  drop=0.2)
        self.ppg_DBP_head = Mlp_BP(in_features=in_features, 
                                  hidden_features=in_features // 2, 
                                  out_features=1, 
                                  drop=0.2)
        self.ecg_SBP_head = Mlp_BP(in_features=in_features, 
                                  hidden_features=in_features // 2, 
                                  out_features=1, 
                                  drop=0.2)
        self.ecg_DBP_head = Mlp_BP(in_features=in_features, 
                                  hidden_features=in_features // 2, 
                                  out_features=1, 
                                  drop=0.2)

        # Initialize L1 loss for BP prediction
        self.l1 = torch.nn.L1Loss()
        
        # Store age threshold for weighted contrastive loss
        self.wcl_age_threshold = wcl_age_threshold

    def forward(self, batch):
        """Forward pass of the blood pressure estimation model.
        
        This method implements the complete forward pass of the model, processing both
        waveform data (ECG/PPG) and optionally patient information to predict blood
        pressure values. The method can operate in two modes:
        1. With patient information (PI): Uses both waveform and text data
        2. Without patient information: Uses only waveform data
        
        The processing pipeline:
        1. Process waveform data:
           - Extract features from ECG and PPG signals
           - Project features to embedding space
        2. If using PI:
           - Process text data
           - Combine waveform and text features
           - Process combined features
        3. If not using PI:
           - Process waveform embeddings directly
        4. Predict blood pressure values
        5. Compute losses:
           - L1 losses for BP predictions
           - Weighted contrastive losses (if enabled)
           - Text-based WCL (if using PI)
        
        Args:
            batch (dict): Dictionary containing:
                - "ecg" (torch.Tensor): ECG signal data of shape (batch_size, channels, sequence_length)
                - "ppg" (torch.Tensor): PPG signal data of shape (batch_size, channels, sequence_length)
                - "input_ids" (torch.Tensor, optional): Text token IDs if using PI
                - "attention_mask" (torch.Tensor, optional): Text attention mask if using PI
                - "SBP_n" (torch.Tensor): Normalized systolic BP values
                - "DBP_n" (torch.Tensor): Normalized diastolic BP values
                - "SBP" (torch.Tensor): Raw systolic BP values for WCL
                - "DBP" (torch.Tensor): Raw diastolic BP values for WCL
                - "gender" (torch.Tensor, optional): Gender values for text WCL if using PI
                - "age" (torch.Tensor, optional): Age values for text WCL if using PI
        
        Returns:
            tuple: (ecg_loss, ppg_loss)
                - ecg_loss (torch.Tensor): Combined loss for ECG-based predictions
                - ppg_loss (torch.Tensor): Combined loss for PPG-based predictions
        """
        # Process ECG features
        # Shape: (batch_size, sequence_length, d_model)
        ecg_features = self.ecg_encoder(batch["ecg"].view(batch["ecg"].shape[0], batch["ecg"].shape[1], -1))
        
        # Process PPG features
        # Shape: (batch_size, sequence_length, d_model)
        ppg_features = self.ppg_encoder(batch["ppg"].view(batch["ppg"].shape[0], batch["ppg"].shape[1], -1))
        
        # Project ECG features to embedding space
        # Shape: (batch_size, projection_dim)
        ecg_embeddings = self.ecg_projection(ecg_features[0].view(ecg_features[0].shape[0], -1))
        
        # Project PPG features to embedding space
        # Shape: (batch_size, projection_dim)
        ppg_embeddings = self.ppg_projection(ppg_features[0].view(ppg_features[0].shape[0], -1))

        # Initialize text_embeddings as None
        text_embeddings = None

        if self.pi:
            # Process text data when using patient information
            # Shape: (batch_size, text_embedding_dim)
            text_features = self.text_encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            # Project text features
            # Shape: (batch_size, projection_dim)
            text_embeddings = self.text_projection(text_features)

            # Combine waveform and text features
            # Shape: (batch_size, 2, projection_dim)
            ecg_text_features = torch.cat((ecg_embeddings.unsqueeze(2), text_embeddings.unsqueeze(2)), dim=2)
            ppg_text_features = torch.cat((ppg_embeddings.unsqueeze(2), text_embeddings.unsqueeze(2)), dim=2)

            # Process combined features
            # Shape: (batch_size, sequence_length, d_model)
            ppg_text_features = self.ppg_text_encoder(ppg_text_features)
            ecg_text_features = self.ecg_text_encoder(ecg_text_features)

            # Get final features for BP prediction
            # Shape: (batch_size, flattened_features)
            ppg_final = ppg_text_features[0].view(ppg_text_features[0].shape[0], -1)
            ecg_final = ecg_text_features[0].view(ecg_text_features[0].shape[0], -1)
        else:
            # Process waveform embeddings directly when not using patient information
            # Shape: (batch_size, sequence_length, d_model)
            ppg_bp_features = self.ppg_bp_encoder(ppg_embeddings.unsqueeze(-1))
            ecg_bp_features = self.ecg_bp_encoder(ecg_embeddings.unsqueeze(-1))

            # Get final features for BP prediction
            # Shape: (batch_size, flattened_features)
            ppg_final = ppg_bp_features[0].view(ppg_bp_features[0].shape[0], -1)
            ecg_final = ecg_bp_features[0].view(ecg_bp_features[0].shape[0], -1)

        # Predict blood pressure values
        # Shape: (batch_size, 1) for each
        y_ppg_SBP = self.ppg_SBP_head(ppg_final)
        y_ppg_DBP = self.ppg_DBP_head(ppg_final)
        y_ecg_SBP = self.ecg_SBP_head(ecg_final)
        y_ecg_DBP = self.ecg_DBP_head(ecg_final)

        # Calculate L1 losses for BP predictions
        loss_ppg_SBP = self.l1(batch["SBP_n"], y_ppg_SBP)
        loss_ppg_DBP = self.l1(batch["DBP_n"], y_ppg_DBP)
        loss_ecg_SBP = self.l1(batch["SBP_n"], y_ecg_SBP)
        loss_ecg_DBP = self.l1(batch["DBP_n"], y_ecg_DBP)

        # Initialize base losses
        ecg_loss = loss_ecg_SBP + loss_ecg_DBP
        ppg_loss = loss_ppg_SBP + loss_ppg_DBP

        # Add weighted contrastive losses if enabled
        if self.wcl:
            # Compute WCL for ECG and PPG embeddings
            ecg_wcl = self._compute_wcl_losses(ecg_embeddings, batch)
            ppg_wcl = self._compute_wcl_losses(ppg_embeddings, batch)
            
            # Add WCL to base losses
            ecg_loss += ecg_wcl
            ppg_loss += ppg_wcl

            # Add text WCL only if both WCL and PI are enabled
            if self.pi and text_embeddings is not None:
                # Compute text-based WCL
                text_wcl = self._compute_text_wcl(text_embeddings, batch)
                # Add text WCL to both ECG and PPG losses
                ecg_loss += text_wcl
                ppg_loss += text_wcl

        return ecg_loss, ppg_loss

    def _compute_wcl_losses(self, embeddings, batch):
        """Compute weighted contrastive losses for waveform embeddings.
        
        This method computes weighted contrastive losses for waveform embeddings (ECG/PPG)
        based on blood pressure values (SBP and DBP). It helps learn waveform representations
        that are similar for similar blood pressure values.
        
        The loss computation:
        1. Compute SBP-based WCL using regression parameters
        2. Compute DBP-based WCL using regression parameters
        3. Combine losses with appropriate scaling based on BP normalization
        
        Args:
            embeddings (torch.Tensor): Waveform embeddings of shape (batch_size, embedding_dim)
            batch (dict): Dictionary containing:
                - "SBP" (torch.Tensor): Systolic blood pressure values of shape (batch_size,)
                - "DBP" (torch.Tensor): Diastolic blood pressure values of shape (batch_size,)
        
        Returns:
            torch.Tensor: Combined weighted contrastive loss for waveform embeddings
            
        Note:
            - Both SBP and DBP WCL use temperature=4.0 and threshold=0.0235 for regression
            - Losses are scaled by 1e-3 when BP values are normalized
            - The same parameters are used for both SBP and DBP to ensure consistent learning
        """
        # Compute SBP-based weighted contrastive loss
        # Uses regression parameters (temperature=4.0, threshold=0.0235)
        wcl_SBP = self.weighted_constrastive_loss(embeddings, self.temperature, 
                                           batch["SBP"], 4, 0.0235)
        
        # Compute DBP-based weighted contrastive loss
        # Uses regression parameters (temperature=4.0, threshold=0.0235)
        wcl_DBP = self.weighted_constrastive_loss(embeddings, self.temperature, 
                                           batch["DBP"], 4, 0.0235)
        
        # Combine SBP and DBP losses with appropriate scaling
        if self.normalized_bp:
            # Scale losses by 1e-3 when BP values are normalized
            return wcl_SBP * 1e-3 + wcl_DBP * 1e-3
        # Use raw losses when BP values are not normalized
        return wcl_SBP + wcl_DBP

    def _compute_text_wcl(self, text_embeddings, batch):
        """Compute weighted contrastive loss for text embeddings.
        
        This method computes weighted contrastive losses for text embeddings based on
        demographic information (gender and age). It helps learn text representations
        that are similar for similar demographic characteristics.
        
        The loss computation:
        1. Compute gender-based WCL using binary classification
        2. If age threshold is set:
           - Compute age-based WCL using regression
           - Combine gender and age losses with appropriate scaling
        3. If no age threshold:
           - Use only gender-based WCL
        4. Apply normalization if BP values are normalized
        
        Args:
            text_embeddings (torch.Tensor): Text embeddings of shape (batch_size, embedding_dim)
            batch (dict): Dictionary containing:
                - "gender" (torch.Tensor): Gender values of shape (batch_size,)
                - "age" (torch.Tensor): Age values of shape (batch_size,)
        
        Returns:
            torch.Tensor: Combined weighted contrastive loss for text embeddings
            
        Note:
            - Gender WCL uses temperature=1.0 and threshold=1.0 for binary classification
            - Age WCL uses temperature=4.0 and threshold=0.0235 for regression
            - Losses are scaled by 1e-2 when BP values are normalized
        """
        # Compute gender-based weighted contrastive loss
        # Uses binary classification parameters (temperature=1.0, threshold=1.0)
        text_wcl_gender = self.weighted_constrastive_loss(text_embeddings, self.temperature, 
                                                   batch["gender"], 1, 1)
        
        if self.wcl_age_threshold is not None:
            # Compute age-based weighted contrastive loss
            # Uses regression parameters (temperature=4.0, threshold=0.0235)
            text_wcl_age = self.weighted_constrastive_loss(text_embeddings, self.temperature,
                                                           batch["age"], 4, 0.0235)
            
            # Combine gender and age losses with appropriate scaling
            if self.normalized_bp:
                # Scale losses by 1e-2 when BP values are normalized
                text_wcl = text_wcl_age * 1e-2 + text_wcl_gender * 1e-2
            else:
                # Use raw losses when BP values are not normalized
                text_wcl = text_wcl_age + text_wcl_gender
        else:
            # Use only gender-based loss if no age threshold is set
            if self.normalized_bp:
                # Scale loss by 1e-2 when BP values are normalized
                text_wcl = text_wcl_gender * 1e-2
            else:
                # Use raw loss when BP values are not normalized
                text_wcl = text_wcl_gender

        return text_wcl

    def get_SBP_DBP_fromPPG(self, batch):
        """Get blood pressure predictions from PPG signal.
        
        This method processes Photoplethysmogram (PPG) signals to predict both systolic (SBP)
        and diastolic (DBP) blood pressure values. It can operate in two modes:
        1. With patient information (PI): Uses both PPG and text data
        2. Without patient information: Uses only PPG data
        
        The processing pipeline:
        1. Extract PPG features using the PPG encoder
        2. Project features to embedding space
        3. If using PI:
           - Process text data
           - Combine PPG and text features
           - Process combined features
        4. If not using PI:
           - Process PPG embeddings directly
        5. Predict SBP and DBP values
        
        Args:
            batch (dict): Dictionary containing:
                - "ppg" (torch.Tensor): PPG signal data of shape (batch_size, channels, sequence_length)
                - "input_ids" (torch.Tensor, optional): Text token IDs if using PI
                - "attention_mask" (torch.Tensor, optional): Text attention mask if using PI
        
        Returns:
            tuple: (SBP_predictions, DBP_predictions)
                - SBP_predictions (torch.Tensor): Predicted systolic blood pressure values
                - DBP_predictions (torch.Tensor): Predicted diastolic blood pressure values
        
        Example:
            >>> batch = {
            ...     "ppg": torch.randn(32, 1, 1024),  # batch_size=32, channels=1, seq_len=1024
            ...     "input_ids": torch.randint(0, 1000, (32, 128)),  # if using PI
            ...     "attention_mask": torch.ones(32, 128)  # if using PI
            ... }
            >>> sbp, dbp = model.get_SBP_DBP_fromPPG(batch)
        """
        # Extract features from PPG signal
        # Shape: (batch_size, sequence_length, d_model)
        ppg_features = self.ppg_encoder(batch["ppg"].view(batch["ppg"].shape[0], batch["ppg"].shape[1], -1))
        
        # Project PPG features to embedding space
        # Shape: (batch_size, projection_dim)
        ppg_embeddings = self.ppg_projection(ppg_features[0].view(ppg_features[0].shape[0], -1))

        if self.pi:
            # Process text data when using patient information
            # Shape: (batch_size, text_embedding_dim)
            text_features = self.text_encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            # Project text features
            # Shape: (batch_size, projection_dim)
            text_embeddings = self.text_projection(text_features)

            # Combine PPG and text features
            # Shape: (batch_size, 2, projection_dim)
            ppg_text_features = torch.cat((ppg_embeddings.unsqueeze(2), text_embeddings.unsqueeze(2)), dim=2)
            
            # Process combined features
            # Shape: (batch_size, sequence_length, d_model)
            ppg_text_features = self.ppg_text_encoder(ppg_text_features)
            
            # Get final features for BP prediction
            # Shape: (batch_size, flattened_features)
            ppg_final = ppg_text_features[0].view(ppg_text_features[0].shape[0], -1)
        else:
            # Process PPG embeddings directly when not using patient information
            # Shape: (batch_size, sequence_length, d_model)
            ppg_bp_features = self.ppg_bp_encoder(ppg_embeddings.unsqueeze(-1))
            
            # Get final features for BP prediction
            # Shape: (batch_size, flattened_features)
            ppg_final = ppg_bp_features[0].view(ppg_bp_features[0].shape[0], -1)

        # Predict SBP and DBP values
        # Shape: (batch_size, 1) for each
        y_ppg_SBP = self.ppg_SBP_head(ppg_final)
        y_ppg_DBP = self.ppg_DBP_head(ppg_final)

        return (y_ppg_SBP, y_ppg_DBP)

    def get_SBP_DBP_fromECG(self, batch):
        """Get blood pressure predictions from ECG signal.
        
        This method processes ECG signals to predict both systolic (SBP) and diastolic (DBP)
        blood pressure values. It can operate in two modes:
        1. With patient information (PI): Uses both ECG and text data
        2. Without patient information: Uses only ECG data
        
        The processing pipeline:
        1. Extract ECG features using the ECG encoder
        2. Project features to embedding space
        3. If using PI:
           - Process text data
           - Combine ECG and text features
           - Process combined features
        4. If not using PI:
           - Process ECG embeddings directly
        5. Predict SBP and DBP values
        
        Args:
            batch (dict): Dictionary containing:
                - "ecg" (torch.Tensor): ECG signal data of shape (batch_size, channels, sequence_length)
                - "input_ids" (torch.Tensor, optional): Text token IDs if using PI
                - "attention_mask" (torch.Tensor, optional): Text attention mask if using PI
        
        Returns:
            tuple: (SBP_predictions, DBP_predictions)
                - SBP_predictions (torch.Tensor): Predicted systolic blood pressure values
                - DBP_predictions (torch.Tensor): Predicted diastolic blood pressure values
        
        Example:
            >>> batch = {
            ...     "ecg": torch.randn(32, 1, 1024),  # batch_size=32, channels=1, seq_len=1024
            ...     "input_ids": torch.randint(0, 1000, (32, 128)),  # if using PI
            ...     "attention_mask": torch.ones(32, 128)  # if using PI
            ... }
            >>> sbp, dbp = model.get_SBP_DBP_fromECG(batch)
        """
        # Extract features from ECG signal
        # Shape: (batch_size, sequence_length, d_model)
        ecg_features = self.ecg_encoder(batch["ecg"].view(batch["ecg"].shape[0], batch["ecg"].shape[1], -1))
        
        # Project ECG features to embedding space
        # Shape: (batch_size, projection_dim)
        ecg_embeddings = self.ecg_projection(ecg_features[0].view(ecg_features[0].shape[0], -1))

        if self.pi:
            # Process text data when using patient information
            # Shape: (batch_size, text_embedding_dim)
            text_features = self.text_encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            # Project text features
            # Shape: (batch_size, projection_dim)
            text_embeddings = self.text_projection(text_features)

            # Combine ECG and text features
            # Shape: (batch_size, 2, projection_dim)
            ecg_text_features = torch.cat((ecg_embeddings.unsqueeze(2), text_embeddings.unsqueeze(2)), dim=2)
            
            # Process combined features
            # Shape: (batch_size, sequence_length, d_model)
            ecg_text_features = self.ecg_text_encoder(ecg_text_features)
            
            # Get final features for BP prediction
            # Shape: (batch_size, flattened_features)
            ecg_final = ecg_text_features[0].view(ecg_text_features[0].shape[0], -1)
        else:
            # Process ECG embeddings directly when not using patient information
            # Shape: (batch_size, sequence_length, d_model)
            ecg_bp_features = self.ecg_bp_encoder(ecg_embeddings.unsqueeze(-1))
            
            # Get final features for BP prediction
            # Shape: (batch_size, flattened_features)
            ecg_final = ecg_bp_features[0].view(ecg_bp_features[0].shape[0], -1)

        # Predict SBP and DBP values
        # Shape: (batch_size, 1) for each
        y_ecg_SBP = self.ecg_SBP_head(ecg_final)
        y_ecg_DBP = self.ecg_DBP_head(ecg_final)

        return (y_ecg_SBP, y_ecg_DBP)

    def weighted_constrastive_loss(self, embeddings, temperature_embeddings, weight, temperature_weight, threshold):
        """Compute weighted contrastive loss for learning discriminative embeddings.
        
        This method implements a weighted contrastive loss that considers both embedding
        similarities and target value similarities. It helps learn embeddings that are
        similar for similar target values while being different for dissimilar ones.
        
        The loss is computed as:
        1. Calculate weight similarity matrix using exponential of negative absolute differences
        2. Apply threshold to filter out low similarities
        3. Calculate embedding similarities using dot product
        4. Compute log probabilities of embedding similarities
        5. Weight the log probabilities by the weight similarities
        6. Normalize by the sum of weight similarities
        
        Args:
            embeddings (torch.Tensor): Feature embeddings of shape (batch_size, embedding_dim)
            temperature_embeddings (float): Temperature parameter for embedding similarities.
                Higher values make the distribution more uniform
            weight (torch.Tensor): Target values of shape (batch_size,) used for weighting
            temperature_weight (float): Temperature parameter for weight similarities.
                Higher values make the distribution more uniform
            threshold (float): Minimum similarity threshold for weight matrix.
                Values below this are set to zero
            
        Returns:
            torch.Tensor: Mean weighted contrastive loss value
            
        Example:
            >>> embeddings = torch.randn(32, 256)  # batch_size=32, embedding_dim=256
            >>> weights = torch.randn(32)  # batch_size=32
            >>> loss = weighted_constrastive_loss(embeddings, 0.5, weights, 0.1, 0.1)
        """
        # Calculate weight similarity matrix using exponential of negative absolute differences
        # Shape: (batch_size, batch_size)
        weight_similarity = torch.exp(-(torch.abs(weight - weight.T) / temperature_weight))
        
        # Zero out similarities below threshold to focus on more similar pairs
        weight_similarity = torch.where(weight_similarity >= threshold, 
                                      weight_similarity, 
                                      torch.zeros_like(weight_similarity))

        # Calculate normalization term for weight similarities
        # Shape: (batch_size, 1)
        weight_similarity_norm = weight_similarity.sum(dim=-1, keepdim=True)

        # Calculate embedding similarities using dot product and temperature scaling
        # Shape: (batch_size, batch_size)
        emb_similarity = torch.matmul(embeddings, embeddings.T) / temperature_embeddings

        # Calculate log probabilities of embedding similarities
        # Shape: (batch_size, batch_size)
        log_prob = F.log_softmax(emb_similarity, dim=-1)

        # Compute normalized weighted contrastive loss
        # 1. Multiply log probabilities by weight similarities
        # 2. Sum over the batch dimension
        # 3. Normalize by the sum of weight similarities
        # 4. Add small epsilon to avoid division by zero
        loss = -torch.sum(weight_similarity * log_prob, dim=-1) / (weight_similarity_norm.squeeze(-1) + 1e-8)

        # Return mean loss over the batch
        return loss.mean()