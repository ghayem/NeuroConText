import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class ClipLoss(nn.Module):
    def forward(self, image_embeddings, text_embeddings, logit_scale, *_):
        logits_per_image = logit_scale * image_embeddings @ text_embeddings.T
        logits_per_text = logit_scale * text_embeddings @ image_embeddings.T
        labels = torch.arange(len(logits_per_image), device=image_embeddings.device)

        return (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2
    