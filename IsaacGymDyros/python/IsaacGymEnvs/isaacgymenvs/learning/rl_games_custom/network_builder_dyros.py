from isaacgymenvs.utils.actor_critic_dyros import DyrosActorCritic

import torch
import torch.nn as nn

from isaacgymenvs.utils.actor_critic_dyros import DyrosActorCritic
from rl_games.algos_torch.network_builder import NetworkBuilder


class A2CDYROSBuilder(NetworkBuilder):
    """
    Custom NetworkBuilder for Dyros actor-critic networks.
    Overrides build and load to separate network construction and weight loading.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def build(self, name, *args, **kwargs):
        # 전달된 환경 및 모델 설정을 params로 저장
        kwargs.pop('device', None)
        kwargs.pop('dtype',  None)
        self.params = kwargs

        if name == 'dyros_actor_critic':
            # Runner가 계산해 주입한 input_shape/priv_dim 우선 사용
            env_cfg   = kwargs.get('env', {})
            model_cfg = kwargs.get('model', {})

            num_obs  = kwargs.get('input_shape', (env_cfg.get('NumSingleStepObs'),))[0]
            priv_dim = kwargs.get('priv_dim', env_cfg.get('priv_dim', 187))
            act_dim  = kwargs.get('actions_num', env_cfg.get('NumAction'))
            rnn_hid  = kwargs.get('rnn_hidden', model_cfg.get('rnn_hidden', 256))
            zdim     = kwargs.get('z_dim', model_cfg.get('latent_dim', 24))

            return DyrosActorCritic(
                num_obs     = num_obs,
                priv_dim    = priv_dim,
                num_actions = act_dim,
                rnn_hidden  = rnn_hid,
                z_dim       = zdim
            )

    def load(self, params):
        # ─── load() 시작부에도 stray factory-kwargs 제거 ───────────
        if isinstance(params, dict):
            params.pop('device', None)
            params.pop('dtype',  None)
            if 'env' in params and isinstance(params['env'], dict):
                params['env'].pop('device', None)
                params['env'].pop('dtype',  None)
            if 'model' in params and isinstance(params['model'], dict):
                params['model'].pop('device', None)
                params['model'].pop('dtype',  None)

        # 전체 설정 저장
        #self.params = params
        net_cfg = params.get('network', {})
        name = net_cfg.get('name', 'dyros_actor_critic')

        # Build network instance with env/model configs
        # ─── Build: 필요한 키만 골라 전달 ────────────────────────
        build_kwargs = {
            'env'          : params.get('env'),
            'model'        : params.get('model'),
            'input_shape'  : params.get('input_shape'),
            'priv_dim'     : params.get('priv_dim'),
            'actions_num'  : params.get('actions_num'),
            'rnn_hidden'   : params.get('rnn_hidden'),
            'z_dim'        : params.get('z_dim'),
        }
        # None인 값들은 제거
        build_kwargs = {k:v for k,v in build_kwargs.items() if v is not None}
        network = self.build(name, **build_kwargs)

        # Load weights if provided
        state_dict = net_cfg.get('state_dict', {})
        if state_dict:
            network.load_state_dict(state_dict)

        return network
