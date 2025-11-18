import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext


from .quant import VectorQuantizerM
from .vqvae import AttnProjection
def is_rank0():
    return (not dist.is_initialized()) or (dist.get_rank() == 0)

class UniTok(nn.Module):
    def __init__(self, args):
        super().__init__()
        try:
            embed_dim = args.unitok_embed_dim
            vocab_width = args.unitok_vocab_width
            vocab_size = args.unitok_vocab_size
            e_temp =args.unitok_e_temp
            num_codebooks = args.unitok_num_codebooks
            le = args.unitok_le 
        except:
            embed_dim = args['unitok_embed_dim']
            vocab_width = args['unitok_vocab_width']
            vocab_size = args['unitok_vocab_size']
            e_temp =args['unitok_e_temp']
            num_codebooks = args['unitok_num_codebooks']
            le = args['unitok_le']
    
        self.quant_proj = AttnProjection(embed_dim, vocab_width, embed_dim // vocab_width)


        self.quantizer = VectorQuantizerM(
            vocab_size=vocab_size, 
            vocab_width=vocab_width,
            beta=0.25,  
            use_entropy_loss=le > 0,
            entropy_temp=e_temp, 
            num_codebooks=num_codebooks, 
        )

        self.post_quant_proj = AttnProjection(vocab_width, embed_dim, embed_dim // vocab_width)
        

        self.fc_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  
        self.vocab_size = vocab_size


    def forward(self, input_feature):
        with torch.amp.autocast(device_type="cuda", enabled=False):
            img_tokens = self.quant_proj(input_feature)
            img_tokens, vq_loss, entropy_loss, usages = self.quantizer(img_tokens)
            img_tokens = self.post_quant_proj(img_tokens)

        output_dict = {
            "img_rec": img_tokens,
            "vq_loss": vq_loss,
            "entropy_loss": entropy_loss,
            "codebook_usages": usages,
            "logit_scale": self.logit_scale.exp()
        }
        return output_dict

