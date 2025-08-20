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
                 num_obs: int = 126,
                 priv_dim: int = 144,      # passed in from YAML later
                 num_actions: int = 13,
                 rnn_hidden: int = 256,
                 z_dim: int = 32):
        super().__init__()
        if hasattr(self, '_factory_kwargs'):
            self._factory_kwargs = {
                k: v for k, v in self._factory_kwargs.items() if v is not None
            }
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
        nn.init.xavier_uniform_(self.decoder[-1].weight, gain=0.5)  # 작은 gain
        nn.init.zeros_(self.decoder[-1].bias)
        # ---------- Critic ----------
        self.value_net = nn.Sequential(
            nn.Linear(priv_dim, 512), nn.ELU(inplace=True),
            nn.Linear(512, 512),     nn.ELU(inplace=True),
            nn.Linear(512, 256),     nn.ELU(inplace=True),
            nn.Linear(256, 1)
        )
        # --- global log σ (fixed, learnable) ---
        self.log_std = nn.Parameter(torch.full((num_actions,), -1.609))  # ln 0.05s

    # ----------------------------------------------------
    def forward(self,
                obs_seq: torch.Tensor,     # (B,T,61)
                priv_seq: torch.Tensor,    # (B,T,priv_dim)
                hidden=None):

        # ── 0) 디바이스 통일 ─────────────────────────────────────────────
        # obs_seq가 이미 GPU에 올라와 있으니, 같은 디바이스로 priv_seq와 hidden을 옮겨 줍니다.
        device = obs_seq.device
        priv_seq = priv_seq.to(device)
        if isinstance(hidden, (list, tuple)):
            # list/tuple 형태로 온 RNN state도 마찬가지로
            hidden = [h.to(device) for h in hidden]
        elif hidden is not None:
            hidden = hidden.to(device)

        # 1) hidden이 tuple/list로 왔을 때 첫 번째 요소만 사용
        if isinstance(hidden, (list, tuple)):
            hid_in = hidden[0]
        else:
            hid_in = hidden

        # 2) 메모리 레이아웃 보장
        obs_seq = obs_seq.contiguous()
        if hid_in is not None:
            hid_in = hid_in.contiguous()

        # 3) cuDNN 파라미터 캐시 초기화
        self.gru.flatten_parameters()

        # 4) hidden 배치 크기 확장 (1 → B) - 수정된 부분
        if hid_in is not None:
            if hid_in.size(1) == 1 and hid_in.size(1) != obs_seq.size(0):
                hid_in = hid_in.expand(-1, obs_seq.size(0), -1).contiguous()
        else:
            # hidden이 None이면 기본값 생성
            num_layers = self.gru.num_layers
            hidden_size = self.gru.hidden_size
            hid_in = torch.zeros(num_layers, obs_seq.size(0), hidden_size, 
                                device=device, dtype=obs_seq.dtype)

        # 5) GRU 인코딩
        #from torch.backends import cudnn
        #with cudnn.flags(enabled=False):
            # 반드시 hid_in 을 넘겨야 올바른 hidden state 사용
        #    gru_out, h_new = self.gru(obs_seq, hid_in)
        try:
            gru_out, h_new = self.gru(obs_seq, hid_in)
        except RuntimeError as e:
            print(f"GRU forward error: {e}")
            print(f"obs_seq shape: {obs_seq.shape}, hid_in shape: {hid_in.shape if hid_in is not None else None}")
            raise

        # 6) 마지막 타임스텝만 꺼내 z 생성
        h_t = gru_out[:, -1]                             # (B,256)
        z   = self.emb(h_t)                              # (B,24)

        # 7) Actor head
        mu    = self.policy_net(z)                       # (B, num_actions)
        sigma = self.log_std.exp().expand_as(mu)         # (B, num_actions)

        # 8) Decoder
        recon = self.decoder(z)                          # (B, priv_dim)
  
        target_priv = priv_seq[:, -1]  # 마지막 시점의 privileged state
        if recon.shape[1] != target_priv.shape[1]:
            print(f"Warning: Decoder output size {recon.shape[1]} != target size {target_priv.shape[1]}")
            # 크기를 맞추기 위해 잘라내거나 패딩
            min_dim = min(recon.shape[1], target_priv.shape[1])
            recon = recon[:, :min_dim]

        # 9) Critic head (privileged state 마지막 시점)
        value = self.value_net(priv_seq[:, -1])          # (B,1)
        #value = value.squeeze(-1)                        # (B,)

        return mu, sigma, value, z, recon, h_new

    def is_rnn(self) -> bool:
        # GRU를 사용하므로 항상 True
        return True

    def get_default_rnn_state(self):
        """
        GRU의 기본 hidden state를 반환.
        shape = (num_layers, 1, hidden_size) 인 텐서 1개를 tuple로 감싸서 반환.
        """
        num_layers  = self.gru.num_layers
        hidden_size = self.gru.hidden_size
        # 첫 루프 때 seq_len(=1) 만큼만 할당하고, 이후 RL-Games에서
        # 실제 batch×horizon 길이에 맞춰 복제해 사용합니다.
        return (torch.zeros(num_layers, 1, hidden_size),)