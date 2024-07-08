import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size, output_size, inner_size, num_layers,
                 activation_func=nn.GELU(), dropout=0.1):
        super(MLP, self).__init__()

        # Create a list to hold the layers
        layers = [
            nn.Linear(input_size, inner_size),
            nn.LayerNorm(inner_size),
            activation_func,
            nn.Dropout(dropout),

        ]
        # Hidden layers
        for _ in range(num_layers-1):
            layers.append(nn.Linear(inner_size, inner_size))
            nn.LayerNorm(inner_size),
            layers.append(activation_func)
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(inner_size, output_size))

        # Combine all layers into a Sequential module
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)



class ResidualHead(nn.Module):
    def __init__(
        self,
        dim,
        dropout,
    ):
        super().__init__()
        self.gelu = nn.GELU()
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        out = self.fc(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = x + out
        out = self.layer_norm(out)
        return out


class ProjectionHead(nn.Module):
    """Taken from https://www.kaggle.com/code/moeinshariatnia/openai-clip-simple-implementation"""
    def __init__(
        self,
        embedding_dim,
        output_dim,
        dropout,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, output_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class ClipModel(nn.Module):
    def __init__(self, image_model, text_model, logit_scale=np.log(1/0.07), logit_bias=None):
        super().__init__()

        self.image_model = image_model
        self.text_model = text_model
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale)
        self.logit_bias = (
            nn.Parameter(torch.ones([]) * logit_bias) if logit_bias else None
        )

    def encode_image(self, image):  # DiFuMo
        return self.image_model(image)

    def encode_text(self, text):  # Embeddings
        return self.text_model(text)

    def forward(self, image, text):
        image_embeddings = self.encode_image(image)
        # print(f"image_embeddings shape: {image_embeddings.shape}")
        
        text_embeddings = self.encode_text(text)
        # print(f"text_embeddings shape: {text_embeddings.shape}")

        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        return image_embeddings, text_embeddings
