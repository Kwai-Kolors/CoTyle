import torch
from typing import List, Tuple
from torch.nn import functional as F
from torch import distributed as tdist, nn as nn

def get_entropy_loss(latent_embed, codebook_embed, inv_entropy_tau):
    E_dist = latent_embed.square().sum(dim=1, keepdim=True) + codebook_embed.square().sum(dim=1, keepdim=False)
    E_dist.addmm_(latent_embed, codebook_embed.T, alpha=-2, beta=1)  # E_dist: (N, vocab_size)
    logits = -E_dist.float().mul_(inv_entropy_tau)
    # calc per_sample_entropy
    prob, log_prob = logits.softmax(dim=-1), logits.log_softmax(dim=-1)  # both are (N, vocab_size)
    per_sample_entropy = torch.mean((-prob * log_prob).sum(dim=-1))
    # calc codebook_entropy
    avg_prob = prob.mean(dim=0)  # (vocab_size,)
    log_avg_prob = torch.log(avg_prob + 1e-7)
    codebook_entropy = (-avg_prob * log_avg_prob).sum()
    # calc entropy_loss
    entropy_loss = per_sample_entropy - codebook_entropy
    return entropy_loss


class NormalizedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    def forward(self, idx):
        return F.embedding(
            idx, F.normalize(self.weight, dim=1), self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )

    def get_norm_weight(self):
        return F.normalize(self.weight, dim=1)


class ResConv(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3 if quant_resi < 0 else 1
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BChw):
        return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        vocab_width: int,
        beta: float = 0.25,
        use_entropy_loss=False,
        entropy_temp=0.01,
    ):
        super().__init__()
        self.beta = beta
        self.vocab_size = vocab_size
        self.vocab_width = vocab_width
        self.vocab_usage_record_times: int = 0
        self.register_buffer('vocab_usage', torch.zeros(self.vocab_size))
        self.codebook = NormalizedEmbedding(self.vocab_size, self.vocab_width)

        self.use_entropy_loss = use_entropy_loss
        self.inv_entropy_tau = 1 / entropy_temp

    def init_vocab(self, eini: float):
        if eini > 0:
            nn.init.trunc_normal_(self.codebook.weight.data, std=eini)
        elif eini < 0:
            base = self.vocab_width ** -0.5
            base /= 36
            self.codebook.weight.data.uniform_(-abs(eini) * base, abs(eini) * base)

    def extra_repr(self) -> str:
        return f'beta={self.beta:g}'

    def forward(self, features):
        B, L, C = features.shape
        
        features = features.reshape(-1, C)
        features = F.normalize(features, dim=-1).to(dtype=torch.bfloat16)

        codebook_embed = self.codebook.get_norm_weight()
        indices = torch.argmax(features.detach() @ codebook_embed.T, dim=1)
        entropy_loss = get_entropy_loss(features, codebook_embed, self.inv_entropy_tau) if self.use_entropy_loss else 0   
        features_hat = self.codebook(indices)

        # calc loss
        vq_loss = F.mse_loss(features_hat.detach(), features).mul_(self.beta) + F.mse_loss(features_hat,
                                                                                           features.detach())
        features_hat = (features_hat.detach() - features.detach()).add_(features)

        # update vocab_usage
        prob_per_class_is_chosen = indices.bincount(minlength=self.vocab_size).to(dtype=torch.bfloat16)
        handler = None
        if handler is not None:
            handler.wait()
        prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
        vocab_usage = (prob_per_class_is_chosen > 0.01 / self.vocab_size).to(dtype=torch.bfloat16).mean().mul_(100)
        if self.vocab_usage_record_times == 0:
            self.vocab_usage.copy_(prob_per_class_is_chosen)
        elif self.vocab_usage_record_times < 100:
            self.vocab_usage.mul_(0.9).add_(prob_per_class_is_chosen, alpha=0.1)
        else:
            self.vocab_usage.mul_(0.99).add_(prob_per_class_is_chosen, alpha=0.01)
        self.vocab_usage_record_times += 1

        return features_hat.view(B, L, C), vq_loss, entropy_loss, vocab_usage

    def f_to_idx(self, features):
        B, L, C = features.shape
        features = features.reshape(-1, C)
        features = F.normalize(features, dim=-1).float()
        codebook_embed = self.codebook.get_norm_weight().float()
        indices = torch.argmax(features.detach() @ codebook_embed.T, dim=1)
        return indices.view(B, L)


class VectorQuantizerM(nn.Module):
    def __init__(
        self,
        vocab_size,
        vocab_width,
        beta=0.25,
        use_entropy_loss=False,
        entropy_temp=0.01,
        num_codebooks=16
    ):
        super().__init__()
        self.num_codebooks = num_codebooks 
        self.codebooks = nn.ModuleList()    
        
        for _ in range(num_codebooks):
            codebook = VectorQuantizer(
                vocab_size=vocab_size // num_codebooks, 
                vocab_width=vocab_width // num_codebooks, 
                beta=beta,
                use_entropy_loss=use_entropy_loss,
                entropy_temp=entropy_temp,
            )
            self.codebooks.append(codebook)

    def init_vocab(self, eini: float):
        for codebook in self.codebooks:
            codebook.init_vocab(eini)

    def f_to_idx(self, features):
        indices = []
    
        chunk_size = features.shape[-1] // self.num_codebooks   # features.shape [1, 256, 64]  -->   8     
        splited_features = features.split(chunk_size, dim=-1)   # tuple, len=8, [1, 256, 8]          64是embedding的dim，从dim维度分割
        for i, codebook in enumerate(self.codebooks):
            indices.append(codebook.f_to_idx(splited_features[i]))  # idx: [1, 256]
        
        indices = torch.stack(indices, dim=1)   # 8*[1,256] --> [1, 8, 256]
        
        return indices

    def idx_to_f(self, indices):

        assert indices.shape[1] == self.num_codebooks
        latent_features = []
        for i, codebook in enumerate(self.codebooks):
            sub_indices = indices[:, i].flatten(start_dim=1)
            latent_feature = codebook.codebook(sub_indices) # [1, 256, 8]
            latent_features.append(latent_feature)
        latent_features = torch.cat(latent_features, dim=-1)    # [1, 256, 64]
        
        return latent_features

    def forward(self, features):
        
        latent_features = []
        global_vq_loss = 0.
        global_entropy_loss = 0.
        global_vocab_usage = 0.
        chunk_size = features.shape[-1] // self.num_codebooks
        splited_features = features.split(chunk_size, dim=-1)
        # 
        for i, codebook in enumerate(self.codebooks):
            latent_feature, vq_loss, entropy_loss, vocab_usage = codebook(splited_features[i])
            latent_features.append(latent_feature)
            global_vq_loss += vq_loss
            global_entropy_loss += entropy_loss
            global_vocab_usage += vocab_usage
        # 
        latent_features = torch.cat(latent_features, dim=-1)
        global_entropy_loss /= self.num_codebooks
        global_vq_loss /= self.num_codebooks
        global_vocab_usage /= self.num_codebooks
        return latent_features, global_vq_loss, global_entropy_loss, global_vocab_usage
