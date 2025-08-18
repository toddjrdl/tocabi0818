from rl_games.algos_torch.models import BaseModel
import torch
from isaacgymenvs.utils.actor_critic_dyros import DyrosActorCritic
import torch.nn as nn
import numpy as np

class ModelA2CContinuousLogStdDYROS(BaseModel):
    """
    Wraps the DyrosActorCritic network for RL-Games, providing
    a callable interface, mode switching, noise scheduling, and build API.
    """
    def __init__(self, network: DyrosActorCritic, **kwargs):
        super().__init__()
        # Core paper-based actor-critic network
        self.net = network
        # Placeholder for the inner wrapper
        self.wrapper = None

    def build(self, config=None):
        """
        RL-Games expects a build() returning a nn.Module.
        Create and store the inner Network wrapper once.
        """
        if self.wrapper is None:
            self.wrapper = ModelA2CContinuousLogStdDYROS.Network(self.net)
        return self.wrapper

    def __call__(self, input_dict):
        # Delegate to the built wrapper
        if self.wrapper is None:
            # Ensure build has been called
            self.build()
        return self.wrapper(input_dict)

    def update_action_noise(self, progress_remaining: float):
        """
        Linearly scale the global log_std parameter by training progress.
        """
        if hasattr(self.net, 'log_std'):
            with torch.no_grad():
                ratio = float(progress_remaining)
                self.net.log_std.data.mul_(ratio)

    def eval(self):
        """Switch to evaluation mode for the wrapped network."""
        if self.wrapper is not None:
            self.wrapper.eval()

    def train(self, mode: bool = True):
        """Switch to training mode for the wrapped network."""
        if self.wrapper is not None:
            self.wrapper.train(mode)
        return self

    class Network(nn.Module):
        """
        Inner wrapper exposing the forward API expected by RL-Games.
        """
        def __init__(self, a2c_network: DyrosActorCritic):
            super().__init__()
            self.a2c_network = a2c_network

        def is_rnn(self):
            return getattr(self.a2c_network, 'is_rnn', False)

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)

            # 1) 입력 꺼내기
            obs_seq  = input_dict['obs']          # (B,L,obs) or (B,obs)
            priv_seq = input_dict['states']       # (B,L,priv) or (B,priv)
            hidden   = input_dict.get('rnn_states', None)
            if isinstance(hidden, (list, tuple)):
                hidden = hidden[0]

            # 2) 2D → 3D(L=1)
            if obs_seq.dim()  == 2: obs_seq  = obs_seq.unsqueeze(1)
            if priv_seq.dim() == 2: priv_seq = priv_seq.unsqueeze(1)

            # 3) [B,L,·] 보장. L=1일 때 전치 금지
            if obs_seq.dim() == 3 and obs_seq.size(0) == 1 and obs_seq.size(1) > 1:
                obs_seq = obs_seq.transpose(0, 1).contiguous()
                if priv_seq.dim() == 3 and priv_seq.size(0) == 1 and priv_seq.size(1) > 1:
                    priv_seq = priv_seq.transpose(0, 1).contiguous()

            # 4) hidden 배치 정렬
            if hidden is not None:
                num_layers, b_old, H = hidden.shape
                B_new = obs_seq.size(0)
                if b_old == 1 and B_new > 1:
                    hidden = hidden.expand(num_layers, B_new, H).contiguous()
                else:
                    assert b_old == B_new, f"RNN batch mismatch: hidden={b_old}, obs={B_new}"

            # (선택) cuDNN 최적화
            if hasattr(self.a2c_network, 'gru'):
                try: self.a2c_network.gru.flatten_parameters()
                except Exception: pass

            # 5) 본 네트워크 호출
            mu, sigma, value, z, recon, new_hidden = self.a2c_network(obs_seq, priv_seq, hidden)

            # === NEW: 항상 [B,L,A], [B,L,1] 형태로 맞춤 ===
            B, L = obs_seq.size(0), obs_seq.size(1)
            # mu, sigma
            if mu.dim() == 2:          # (B,A) → (B,L,A)
                mu    = mu.unsqueeze(1).expand(B, L, -1)
            if sigma.dim() == 2:       # (B,A) → (B,L,A)
                sigma = sigma.unsqueeze(1).expand(B, L, -1)

            # value: (B,1) or (B,) → (B,L,1)
            if value.dim() == 1:       # (B,) → (B,L,1)
                value = value.view(B, 1, 1).expand(B, L, 1)
            elif value.dim() == 2:     # (B,1) → (B,L,1)  (두 번째 축이 반드시 1이어야 함)
                assert value.size(1) == 1, f"unexpected value shape {tuple(value.shape)}"
                value = value.unsqueeze(1).expand(B, L, 1)
            elif value.dim() == 3:
                if value.size(1) != L:
                    # (B,1,1) 같은 경우만 허용
                    assert value.size(1) == 1 and value.size(2) == 1, f"value shape {tuple(value.shape)}"
                    value = value.expand(B, L, 1)
            else:
                raise RuntimeError(f"unsupported value shape {tuple(value.shape)}")

            distr = torch.distributions.Normal(mu, sigma)

            if is_train:
                # 6-A) 학습 경로
                entropy = distr.entropy().sum(dim=-1)     # (B,L)

                # prev_actions 정규화
                prev_actions = input_dict.get('prev_actions', None)
                if prev_actions is not None:
                    A = mu.size(-1)
                    pa = prev_actions
                    if not torch.is_tensor(pa):
                        pa = torch.as_tensor(pa, device=mu.device, dtype=mu.dtype)
                    else:
                        pa = pa.to(mu.device)

                    if pa.dim() == 2:
                        if pa.size(0) == B * L and pa.size(1) == A:     # (B*L, A)
                            pa = pa.view(B, L, A)
                        elif pa.size(0) == B and pa.size(1) == A:       # (B, A)
                            pa = pa.unsqueeze(1).expand(B, L, A)
                        else:
                            raise RuntimeError(f"prev_actions shape {tuple(pa.shape)} incompatible with (B,L,A)=({B},{L},{A})")
                    elif pa.dim() == 3:
                        if not (pa.size(0) == B and pa.size(1) == L and pa.size(2) == A):
                            raise RuntimeError(f"prev_actions 3D shape {tuple(pa.shape)} must be (B,L,A)=({B},{L},{A})")
                    else:
                        raise RuntimeError(f"prev_actions must be 2D or 3D, got {pa.dim()}D")

                    # === NEW: (B·L,A)로 펴서 neglogp 계산 후 (B,L)로 복원 ===
                    A = mu.size(-1)
                    prev_neglogp = self.neglogp(
                        pa.reshape(-1, A),
                        mu.reshape(-1, A),
                        sigma.reshape(-1, A),
                        sigma.log().reshape(-1, A),
                    ).view(B, L)
                else:
                    prev_neglogp = None

                return {
                    'values':     value.squeeze(-1),   # (B,L)
                    'entropy':    entropy,             # (B,L)
                    'rnn_states': (new_hidden,),
                    'mus':        mu,                  # (B,L,A)
                    'sigmas':     sigma,               # (B,L,A)
                    'z':          z,
                    'recon':      recon,
                    **({'prev_neglogp': prev_neglogp} if prev_neglogp is not None else {})
                }

            else:
                # 6-B) 롤아웃(환경 step용) — L은 보통 1
                A = mu.size(-1)
                action = distr.sample()  # (B, L, A)

                # neglogp를 (B·L,A) → (B,L) 로 계산
                neg = self.neglogp(
                    action.reshape(-1, A),
                    mu.reshape(-1, A),
                    sigma.reshape(-1, A),
                    sigma.log().reshape(-1, A),
                ).view(B, L)

                if L == 1:
                    action_out = action.squeeze(1)          # (B, A)     ← OK
                    value_out  = value.squeeze(1)           # (B, 1)     ← **여기! 마지막 축은 남긴다**
                    neg_out    = neg.squeeze(1)  # (B, 1)  ← **여기! (B,)가 아니라 (B,1)로**
                    mu_out     = mu.squeeze(1)              # (B, A)
                    sigma_out  = sigma.squeeze(1)           # (B, A)
                else:
                    action_out = action                      # (B, L, A) – 잘 안 쓰이지만 안전하게
                    value_out  = value.squeeze(-1)           # (B, L)
                    neg_out    = neg                         # (B, L)
                    mu_out, sigma_out = mu, sigma

                return {
                    'neglogpacs': neg_out,   # (B,1)
                    'values':     value_out, # (B,1)
                    'actions':    action_out,# (B,A)
                    'rnn_states': (new_hidden,),
                    'mus':        mu_out,
                    'sigmas':     sigma_out,
                    'z':          z,
                    'recon':      recon,
                }

        def neglogp(self, x, mean, std, logstd):
            return (0.5 * (((x - mean) / std) ** 2).sum(dim=-1)
                    + 0.5 * np.log(2.0 * np.pi) * x.size()[-1]
                    + logstd.sum(dim=-1))

        def update_action_noise(self, progress_remaining: float):
            """
            Ramp noise down after halfway through training.
            """
            if progress_remaining > 0.5:
                pr = 2 * progress_remaining - 1
            else:
                pr = 0.0
            with torch.no_grad():
                self.a2c_network.log_std.data.mul_(pr)
                