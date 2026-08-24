import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
# from AdMSLoss import AdMSoftmaxLoss

class SigLipLoss(nn.Module):
    def forward(self, image_embeddings, text_embeddings, logit_scale, logit_bias):
        logits =  logit_scale * image_embeddings @ text_embeddings.T
        if logit_bias:
            logits += logit_bias

        labels = (
            2 * torch.eye(len(logits)) - np.ones(len(logits))
        ).to(image_embeddings.device)

        return -F.logsigmoid(labels * logits).sum() / len(logits)


class ClipLoss(nn.Module):
    def forward(self, image_embeddings, text_embeddings, logit_scale, *_):
        logits_per_image = logit_scale * image_embeddings @ text_embeddings.T
        logits_per_text = logit_scale * text_embeddings @ image_embeddings.T
        labels = torch.arange(len(logits_per_image), device=image_embeddings.device)

        return (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2
    
class ClipDALoss(nn.Module):
    def forward(self, image_embeddings, text_embeddings_s, text_embeddings_t, logit_scale, *_):
        logits_per_image = logit_scale * image_embeddings @ text_embeddings_s.T
        logits_per_text = logit_scale * text_embeddings_s @ image_embeddings.T
        labels = torch.arange(len(logits_per_image), device=image_embeddings.device)
        cov_s = torch.cov(text_embeddings_s)
        cov_t = torch.cov(text_embeddings_t)
        da_loss = deepcoral_loss(cov_s, cov_t) # other options: dan_loss
        return (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2 + reg * da_loss
    

class AdMClipLoss(nn.Module):

    def forward(self, image_embeddings, text_embeddings, logit_scale, *_):
        logits_per_image = logit_scale * image_embeddings @ text_embeddings.T
        logits_per_text = logit_scale * text_embeddings @ image_embeddings.T
        labels = torch.arange(len(logits_per_image), device=image_embeddings.device)
        criterion = AdMSoftmaxLoss(in_features=len(labels), out_features=len(labels), device=image_embeddings.device, s=1.0, m=0.05) # Default values recommended by [1]

        return (
            criterion(logits_per_image, labels)
            + criterion(logits_per_text, labels)
        ) / 2



class AdaptiveClipLoss(nn.Module):
    def __init__(self):
        super(AdaptiveClipLoss, self).__init__()
        self.running_contrastive_loss = 0
        self.running_mse_loss = 0
        self.decay = 0.99  # Decay factor for running averages

    def forward(self, image_embeddings, text_embeddings, logit_scale, *_):
        # Contrastive loss (cross-entropy)
        logits_per_image = logit_scale * image_embeddings @ text_embeddings.T
        logits_per_text = logit_scale * text_embeddings @ image_embeddings.T
        labels = torch.arange(len(logits_per_image), device=image_embeddings.device)
        
        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) 
            + F.cross_entropy(logits_per_text, labels)
        ) / 2
        
        # MSE loss
        mse_loss = F.mse_loss(image_embeddings, text_embeddings)
        
        # Update running averages
        self.running_contrastive_loss = self.decay * self.running_contrastive_loss + (1 - self.decay) * contrastive_loss.item()
        self.running_mse_loss = self.decay * self.running_mse_loss + (1 - self.decay) * mse_loss.item()
        
        # Calculate mse_weight based on running averages
        if self.running_contrastive_loss > 0:
            mse_weight = self.running_contrastive_loss / self.running_mse_loss
        else:
            mse_weight = 1  # Default to 1 if the running_contrastive_loss is zero (e.g., at initialization)
        
        # Combine the losses
        total_loss = contrastive_loss + mse_weight * mse_loss
        
        return total_loss