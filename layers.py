import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class KSparseLayer(nn.Module):
    def __init__(self, k):
        super(KSparseLayer, self).__init__()
        self.k = k

    def forward(self, x):
        # Calculate the energy of each component
        energy = x.pow(2).sum(dim=-1, keepdim=True)
        
        # Find the threshold that keeps the top k components
        sorted_energy, _ = torch.sort(energy, dim=-1, descending=True)
        cumulative_energy = torch.cumsum(sorted_energy, dim=-1)
        threshold = cumulative_energy >= cumulative_energy[:, -1:] * 0.8
        threshold = threshold.float()
        threshold = torch.where(threshold[:, :self.k] > 0, torch.tensor(1.0, device=x.device), torch.tensor(0.0, device=x.device))
        
        # Zero out the components that are not in the top k
        x = x * threshold
        
        return x
    

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


class ResidualHead_autoencoder(nn.Module):
    def __init__(
        self,
        dim,
        dropout,
    ):
        super().__init__()
        self.gelu = nn.GELU()
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        out = self.fc(x)
        out = self.gelu(out)
        out = self.dropout(out)
        # out = x + out
        # out = self.layer_norm(out)
        return out

class ResidualHead_decoder(nn.Module):
    def __init__(
        self,
        dim,
        dropout,
    ):
        super().__init__()
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        out = self.fc(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = x + out
        out = self.layer_norm(out)
        # out = out ** 2  # Ensure non-negative output
        # out = torch.abs(out)  # Ensure non-negative output
        # out = self.relu(out) # Ensure non-negative output
        out = self.sigmoid(out) # Ensure non-negative output
        return out


class ProjectionHead_decoder(nn.Module):
    """Taken from https://www.kaggle.com/code/moeinshariatnia/openai-clip-simple-implementation"""
    def __init__(
        self,
        embedding_dim,
        output_dim,
        dropout,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, output_dim)
        self.elu = nn.ELU()
        self.fc = nn.Linear(output_dim, output_dim)
        # self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x, mask=None):
        projected = self.projection(x)
        x = self.elu(projected)
        x = self.fc(x)
        # x = self.dropout(x)
        # x = x + projected
        # x = self.layer_norm(x)
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

class ClipModel_autoencoder(nn.Module):
    def __init__(self, image_model, text_model, decoder_model, logit_scale=np.log(1/0.07), logit_bias=None):
        super().__init__()

        self.image_model = image_model
        self.text_model = text_model
        self.decoder_model = decoder_model
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale)
        self.logit_bias = (
            nn.Parameter(torch.ones([]) * logit_bias) if logit_bias else None
        )

    def encode_image(self, image):  # DiFuMo
        return self.image_model(image)

    def encode_text(self, text):  # Embeddings
        return self.text_model(text)
    
    def decode_latent(self, latent):  # Embeddings
        return self.decoder_model(latent)

    def forward(self, image, text):
        image_embeddings = self.encode_image(image)
        # print(f"image_embeddings shape: {image_embeddings.shape}")
        
        text_embeddings = self.encode_text(text)
        # print(f"text_embeddings shape: {text_embeddings.shape}")

        # latent = image_embeddings
        latent = text_embeddings
        latent_decoded = self.decode_latent(latent)
        # print(f"latent_decoded shape: {latent_decoded.shape}")

        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        # latent_decoded = latent_decoded / latent_decoded.norm(dim=-1, keepdim=True)

        return image_embeddings, text_embeddings, latent_decoded
    
class ClipModel_withMask(nn.Module):
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


class WeightedAverageTransformer(nn.Module):
    def __init__(self, input_dim, st_embed_dim, num_heads, num_layers, dropout=0.4):
        super(WeightedAverageTransformer, self).__init__()
        self.positional_encoding = PositionalEncoding(st_embed_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=st_embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(st_embed_dim, st_embed_dim)
        self.fc_reduction = nn.Linear(input_dim, st_embed_dim)
        self.layer_norm = nn.LayerNorm(st_embed_dim)
        # self.relu = nn.ReLU()

    def forward(self, x):
        # X: input tensor of shape (batch_size, num_channels, num_rows)
        # mask: boolean tensor of shape (batch_size, num_rows)
        # print(f"#### before mask ####")
        mask = torch.all(x == 0, dim=2)  # (sequence_length, batch_size)
        # mask = ~mask
        # print(f"x: {x[0]}")
        # print(f"mask: {mask[0]}")
        x = self.fc_reduction(x)
        # x = self.relu(x)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # Transformer expects (sequence_length, batch_size, embed_dim)

        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        # x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Back to (batch_size, num_chunks, st_embed_dim)
        x = self.fc(x.mean(dim=1))  # Weighted average
        # x = self.layer_norm(x)        
        # x = x.mean(dim=1)  # Weighted average
        
        # print(f"x.shape: {x.shape}")

        ## Below is to check for only the average chunks
        # zero_rows = (x == 0).all(dim=2)
        # zero_row_counts = zero_rows.sum(dim=1)
        # non_zero_row_counts = x.size(1) - zero_row_counts
        # row_sums = x.sum(dim=1)
        # # print(f"non_zero_row_counts: {non_zero_row_counts}")
        # x = row_sums / non_zero_row_counts.unsqueeze(1)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class WeightedAverageCNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super(WeightedAverageCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size)
        self.conv2 = nn.Conv1d(output_channels, 1, output_channels)

    def forward(self, X, mask):
        # X: input tensor of shape (batch_size, num_channels, num_rows)
        # mask: boolean tensor of shape (batch_size, num_rows)

        mask = torch.all(X == 0, dim=2)  # (sequence_length, batch_size)

        # Apply convolutional layers
        print(f"x shape before conv1: {X.shape}")
        x = self.conv1(X)
        print(f"x shape after conv1: {x.shape}")
        x = F.relu(x)
        x = self.conv2(x)
        print(f"x shape after conv2: {x.shape}")
        weights = F.relu(x).squeeze(1)  # Shape: (batch_size, num_rows)

        # Apply mask to ignore padded rows
        weights = weights * mask
        masked_X = X * mask.unsqueeze(1)  # Shape: (batch_size, num_channels, num_rows)

        # Calculate the weighted sum
        weighted_sum = torch.sum(masked_X * weights.unsqueeze(1), dim=2)  # Shape: (batch_size, num_channels)

        # Normalize by the sum of weights to get the weighted average
        sum_of_weights = torch.sum(weights, dim=1, keepdim=True)  # Shape: (batch_size, 1)
        weighted_avg = weighted_sum / (sum_of_weights + 1e-8)  # Avoid division by zero

        return weighted_avg
