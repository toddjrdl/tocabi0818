# utils/actor_critic_dyros.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DyrosActorCritic(nn.Module):
    """
    GRU-based asymmetric A2C network for DWL
      • Encoder  : GRU(61→256) + FC(256→24) → z
      • Actor    : z → π(a)
      • Decoder  : z → ê(priv_vec)          (denoising head)
      • Critic   : priv_vec → V(s)
    forward()  → logits, value, z, recon, new_hidden
    """

    def __init__(self,
                 num_obs: int = 61,
                 priv_dim: int = 175,      # passed in from YAML later
                 num_actions: int = 12,
                 rnn_hidden: int = 256,
                 z_dim: int = 24):
        super().__init__()
        # ---------- Encoder ----------
        self.gru = nn.GRU(input_size=num_obs,
                          hidden_size=rnn_hidden,
                          batch_first=True)
        self.emb = nn.Sequential(
            nn.Linear(rnn_hidden, rnn_hidden),
            nn.ELU(inplace=True),
            nn.Linear(rnn_hidden, z_dim)
        )
        # ---------- Policy ----------
        self.policy_net = nn.Sequential(
            nn.Linear(z_dim, 48),
            nn.ELU(inplace=True),
            nn.Linear(48, num_actions)
        )
        # ---------- Decoder ----------
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, priv_dim)
        )
        # ---------- Critic ----------
        self.value_net = nn.Sequential(
            nn.Linear(priv_dim, 512), nn.ELU(inplace=True),
            nn.Linear(512, 512),     nn.ELU(inplace=True),
            nn.Linear(512, 256),     nn.ELU(inplace=True),
            nn.Linear(256, 1)
        )
        # --- global log σ (fixed, learnable) ---
        self.log_std = nn.Parameter(torch.full((num_actions,), -2.9957))  # ln 0.05

    # ----------------------------------------------------
    def forward(self,
                obs_seq: torch.Tensor,     # (B,T,61)
                priv_seq: torch.Tensor,    # (B,T,priv_dim)
                hidden: torch.Tensor = None):
        # GRU encoding
        gru_out, h_new = self.gru(obs_seq, hidden)       # (B,T,256)
        h_t = gru_out[:, -1]                             # last step
        z   = self.emb(h_t)                              # (B,24)

        mu = self.policy_net(z)                      # actor
        sigma = self.log_std.exp().expand_as(mu)

        recon  = self.decoder(z)                         # decoder
        value  = self.value_net(priv_seq[:, -1]).squeeze(-1)         # critic

        return mu, sigma, value, z, recon, h_new