'''Network modules for pytorch models. 

Edited by: Kirsten Mayer

Functions
---------
dense_couplet(in_features, out_features, act_fun, *args, **kwargs)
dense_block(out_features, act_fun)

Classes
---------
NeuralNetwork()

'''

import numpy as np
import torch
import torch.nn as nn
from base.base_model import BaseModel


class VisionTransformer(BaseModel):
    def __init__(self,config):
        """
        Parameters:
        - img_height: Height of the input image
        - img_width: Width of the input image
        - patch_height: Height of each patch
        - patch_width: Width of each patch
        - num_outputs: Number of regression outputs
        - dim: Dimension of the token embeddings
        - depth: Number of transformer blocks
        - heads: Number of attention heads in multi-head attention
        - mlp_dim: Hidden dimension of the feed-forward network
        - dropout: Dropout rate
        """
        super().__init__()
        self.config = config
        self.img_height = config["img_height"]
        self.img_width = config["img_width"]
        self.patch_height = config["patch_height"]
        self.patch_width = config["patch_width"]
        self.dim = config["dim"]
        self.num_channels = config["num_channels"]
        
        # Calculate the number of patches
        self.num_patches_h = config["img_height"] // config["patch_height"]
        self.num_patches_w = config["img_width"] // config["patch_width"]
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.patch_dim = (config["patch_height"] * config["patch_width"]) * config["num_channels"]  
        
        # Linear layer to embed patches
        self.patch_embedding = nn.Linear(self.patch_dim, config["dim"])
        
        # Learnable positional embeddings for the patches
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, config["dim"]))
        
        # CLS token (classification token) = patch summary token; also used for regression tasks
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["dim"]))
        
        # Dropout for embeddings
        self.dropout = nn.Dropout(config["dropout"])
        
        # Transformer blocks
        self.transformer = nn.Sequential(
            *[TransformerBlock(config["dim"], config["heads"], config["mlp_dim"], config["dropout"]) for _ in range(config["depth"])]
        )
        
        # Regression head for multiple outputs
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(config["dim"]),
            nn.Linear(config["dim"], config["num_outputs"])
        )

    def forward(self, x):
        """
        Forward pass for the Vision Transformer.
        
        Parameters:
        - x: Input image tensor of shape (batch_size, num_channels, img_height, img_width)
        
        Returns:
        - outputs: Regression outputs of shape (batch_size, num_outputs)
        """
        batch_size = x.size(0)
        
        # Divide the image into patches
        patches = x.unfold(2, self.patch_height, self.patch_height).unfold(self.num_channels, self.patch_width, self.patch_width)
        patches = patches.contiguous().view(batch_size, self.num_channels, -1, self.patch_height * self.patch_width).permute(0, 2, 1, 3)
        patches = patches.contiguous().view(batch_size, self.num_patches, -1)
        
        # Embed patches
        patches = self.patch_embedding(patches)
        
        # Add CLS token to the beginning of the sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, dim)
        x = torch.cat((cls_tokens, patches), dim=1)  # (batch_size, num_patches + 1, dim)
        
        # Add positional embeddings
        x += self.pos_embedding
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through the transformer
        x = self.transformer(x)
        
        # Use the CLS token for regression
        cls_output = x[:, 0]  # (batch_size, dim)
        outputs = self.mlp_head(cls_output)  # (batch_size, num_outputs)
        
        return outputs

    def predict(self, dataset=None, dataloader=None, batch_size=32, device="gpu"):

        if (dataset is None) & (dataloader is None):
            raise ValueError("both dataset and dataloader cannot be none.")

        if (dataset is not None) & (dataloader is not None):
            raise ValueError("dataset and dataloader cannot both be defined. choose one.")

        if dataset is not None:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                )

        self.to(device)
        self.eval()
        with torch.inference_mode():

            output = None
            for batch_idx, (data, target) in enumerate(dataloader):
                input, target = (
                    data.to(device),
                    target.to(device),
                    )

                out = self(input).to("cpu").numpy() # this has to be "cpu" to convert to a numpy
                if output is None:
                    output = out
                else:
                    output = np.concatenate((output, out), axis=0)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        """
        Single Transformer block with:
        - Multi-head Self-Attention
        - Feed-Forward Network (MLP)
        - LayerNorm and Dropout
        
        Parameters:
        - dim: Dimension of the token embeddings
        - heads: Number of attention heads
        - mlp_dim: Hidden dimension of the MLP
        - dropout: Dropout rate
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for a single Transformer block.
        
        Parameters:
        - x: Input tensor of shape (sequence_length, batch_size, dim)
        
        Returns:
        - Output tensor of the same shape
        """
        # Multi-head self-attention with skip connection
        attn_output, _ = self.attn(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-forward network with skip connection
        mlp_output = self.mlp(x)
        x = x + self.dropout2(mlp_output)
        x = self.norm2(x)
        
        return x

