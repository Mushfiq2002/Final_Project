"""
Models for music clustering.
"""
from .ae_mlp import MLPAE, train_ae
from .vae_mlp import MLPVAE, train_vae_mlp
from .vae_conv import ConvVAE, train_vae_conv
from .vae_beta import BetaVAE, train_beta_vae
from .cvae import CVAE, train_cvae

__all__ = [
    'MLPAE', 'train_ae',
    'MLPVAE', 'train_vae_mlp',
    'ConvVAE', 'train_vae_conv',
    'BetaVAE', 'train_beta_vae',
    'CVAE', 'train_cvae'
]


