from isaacgymenvs.utils.actor_critic_dyros import DyrosActorCritic
'''
from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.algos_torch import torch_ext
import torch
import torch.nn as nn

class A2CDYROSBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()

            self.sigma_init = self.space_config['sigma_init']['val']
            self.sigma_last = self.space_config['sigma_last']['val']
            
            if self.has_cnn:
                input_shape = torch_ext.shape_whc_to_cwh(input_shape)
                cnn_args = {
                    'ctype' : self.cnn['type'], 
                    'input_shape' : input_shape, 
                    'convs' :self.cnn['convs'], 
                    'activation' : self.cnn['activation'], 
                    'norm_func_name' : self.normalization,
                }
                self.actor_cnn = self._build_conv(**cnn_args)

                if self.separate:
                    self.critic_cnn = self._build_conv( **cnn_args)

            mlp_input_shape = self._calc_input_size(input_shape, self.actor_cnn)

            in_mlp_shape = mlp_input_shape
            if len(self.units) == 0:
                out_size = mlp_input_shape
            else:
                out_size = self.units[-1]

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    rnn_in_size = out_size
                    out_size = self.rnn_units
                    if self.rnn_concat_input:
                        rnn_in_size += in_mlp_shape
                else:
                    rnn_in_size =  in_mlp_shape
                    in_mlp_shape = self.rnn_units

                if self.separate:
                    self.a_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    self.c_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.a_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                        self.c_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                else:
                    self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.layer_norm = torch.nn.LayerNorm(self.rnn_units)

            mlp_args = {
                'input_size' : in_mlp_shape, 
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

            self.value = torch.nn.Linear(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, actions_num)
            
                for multidiscrete actions num is a tuple
            
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList([torch.nn.Linear(out_size, num) for num in actions_num])
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.space_config['fixed_sigma']:
                    # self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.modules():         
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)    

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.space_config['fixed_sigma']:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)  

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            seq_length = obs_dict.get('seq_length', 1)
            if self.has_cnn:
                # for obs shape 4
                # input expected shape (B, W, H, C)
                # convert to (B, C, W, H)
                if len(obs.shape) == 4:
                    obs = obs.permute((0, 3, 1, 2))
                    
            if self.separate:
                a_out = c_out = obs
                a_out = self.actor_cnn(a_out)
                a_out = a_out.contiguous().view(a_out.size(0), -1)

                c_out = self.critic_cnn(c_out)
                c_out = c_out.contiguous().view(c_out.size(0), -1)                    

                if self.has_rnn:
                    if not self.is_rnn_before_mlp:
                        a_out_in = a_out
                        c_out_in = c_out
                        a_out = self.actor_mlp(a_out_in)
                        c_out = self.critic_mlp(c_out_in)

                        if self.rnn_concat_input:
                            a_out = torch.cat([a_out, a_out_in], dim=1)
                            c_out = torch.cat([c_out, c_out_in], dim=1)

                    batch_size = a_out.size()[0]
                    num_seqs = batch_size // seq_length
                    a_out = a_out.reshape(num_seqs, seq_length, -1)
                    c_out = c_out.reshape(num_seqs, seq_length, -1)

                    if self.rnn_name == 'sru':
                        a_out =a_out.transpose(0,1)
                        c_out =c_out.transpose(0,1)

                    if len(states) == 2:
                        a_states = states[0]
                        c_states = states[1]
                    else:
                        a_states = states[:2]
                        c_states = states[2:]                        
                    a_out, a_states = self.a_rnn(a_out, a_states)
                    c_out, c_states = self.c_rnn(c_out, c_states)
         
                    if self.rnn_name == 'sru':
                        a_out = a_out.transpose(0,1)
                        c_out = c_out.transpose(0,1)
                    else:
                        if self.rnn_ln:
                            a_out = self.a_layer_norm(a_out)
                            c_out = self.c_layer_norm(c_out)
                    a_out = a_out.contiguous().reshape(a_out.size()[0] * a_out.size()[1], -1)
                    c_out = c_out.contiguous().reshape(c_out.size()[0] * c_out.size()[1], -1)

                    if type(a_states) is not tuple:
                        a_states = (a_states,)
                        c_states = (c_states,)
                    states = a_states + c_states

                    if self.is_rnn_before_mlp:
                        a_out = self.actor_mlp(a_out)
                        c_out = self.critic_mlp(c_out)
                else:
                    a_out = self.actor_mlp(a_out)
                    c_out = self.critic_mlp(c_out)
                            
                value = self.value_act(self.value(c_out))

                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits, value, states

                if self.is_multi_discrete:
                    logits = [logit(a_out) for logit in self.logits]
                    return logits, value, states

                if self.is_continuous:
                    mu = self.mu_act(self.mu(a_out))
                    if self.space_config['fixed_sigma']:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))

                    return mu, sigma, value, states
            else:
                out = obs
                out = self.actor_cnn(out)
                out = out.flatten(1)                

                if self.has_rnn:
                    out_in = out
                    if not self.is_rnn_before_mlp:
                        out_in = out
                        out = self.actor_mlp(out)
                        if self.rnn_concat_input:
                            out = torch.cat([out, out_in], dim=1)

                    batch_size = out.size()[0]
                    num_seqs = batch_size // seq_length
                    out = out.reshape(num_seqs, seq_length, -1)

                    if len(states) == 1:
                        states = states[0]

                    if self.rnn_name == 'sru':
                        out = out.transpose(0,1)

                    out, states = self.rnn(out, states)
                    out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)

                    if self.rnn_name == 'sru':
                        out = out.transpose(0,1)
                    if self.rnn_ln:
                        out = self.layer_norm(out)
                    if self.is_rnn_before_mlp:
                        out = self.actor_mlp(out)
                    if type(states) is not tuple:
                        states = (states,)
                else:
                    out = self.actor_mlp(out)
                value = self.value_act(self.value(out))

                if self.central_value:
                    return value, states

                if self.is_discrete:
                    logits = self.logits(out)
                    return logits, value, states
                if self.is_multi_discrete:
                    logits = [logit(out) for logit in self.logits]
                    return logits, value, states
                if self.is_continuous:
                    mu = self.mu_act(self.mu(out))
                    if self.space_config['fixed_sigma']:
                        sigma = self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(out))
                    return mu, mu*0 + sigma, value, states
                    
        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            return self.has_rnn

        def get_default_rnn_state(self):
            if not self.has_rnn:
                return None
            num_layers = self.rnn_layers
            if self.rnn_name == 'identity':
                rnn_units = 1
            else:
                rnn_units = self.rnn_units
            if self.rnn_name == 'lstm':
                if self.separate:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)),
                            torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
                else:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
            else:
                if self.separate:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
                else:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)),)                


        def load(self, params):
            # 모델·네트워크 이름
            self.model_name   = params['model']['name']
            self.network_name = params['network']['name']
    
            # 1) 공통 차원 추출
            obs_dim  = params['env']['NumSingleStepObs']     # 이미 proprio+height=61
            act_dim  = params['env']['NumAction']
    
            # 2) 'dyros_actor_critic' 만 직접 생성하여 빌더를 우회
            if self.network_name == 'dyros_actor_critic':
                from isaacgymenvs.learning.rl_games_custom.actor_critic_dyros import DyrosActorCritic
    
                # 논문 원형 Actor–Critic–Decoder 네트워크
                net = DyrosActorCritic(
                    num_obs   = obs_dim,
                    priv_dim  = params['env']['priv_dim'],
                    num_actions = act_dim,
                    rnn_hidden  = params['model']['rnn_hidden'],
                    z_dim       = params['model']['latent_dim']
                )
                net.load(params['network'])
                network = net
            else:
                # 나머지 네트워크는 factory에 맡겨서 생성
                network = self.network_factory.create(
                    self.network_name,
                    input_shape = (obs_dim,),
                    actions_num = act_dim,
                    value_size  = value_sz,
                    num_seqs    = seq_len
                )
                network.load(params['network'])
    
            # 3) 마지막으로 모델 생성
            model = self.model_factory.create(self.model_name, network=network)
            return model

    def build(self, name, **kwargs):
        # 논문 구현을 쓰고자 하면 network.name='dyros_actor_critic' 으로 지정
        if name == 'dyros_actor_critic':
            # load() 시 저장한 self.params 에서 직접 차원 정보 읽어오기
            num_obs     = kwargs['input_shape'][0]
            priv_dim    = kwargs['priv_dim']
            num_actions = kwargs['actions_num']
            rnn_hidden  = kwargs['rnn_hidden']
            z_dim        = kwargs['z_dim']
            return DyrosActorCritic(
                num_obs     = num_obs,
                priv_dim    = priv_dim,
                num_actions = num_actions,
                rnn_hidden  = rnn_hidden,
                z_dim        = z_dim
            )

        # 그 외 기본 A2CBuilder.Network 에 위 kwargs 를 그대로 전달
        return A2CDYROSBuilder.Network(self.params, **kwargs)
'''
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
        """
        Construct network instances by name.

        If name=='dyros_actor_critic', builds the custom DyrosActorCritic network
        using parameters passed via kwargs.
        Otherwise, delegates to the base NetworkBuilder.
        """
        # 전달된 환경 및 모델 설정을 params로 저장
        self.params = kwargs

        if name == 'dyros_actor_critic':
            # Only build custom DyrosActorCritic when full env/model configs are provided
            if 'env' in kwargs and 'model' in kwargs:
                env_cfg   = self.params.get('env', {})
                model_cfg = self.params.get('model', {})
                return DyrosActorCritic(
                    num_obs     = env_cfg['NumSingleStepObs'],
                    priv_dim    = env_cfg['priv_dim'],
                    num_actions = env_cfg['NumAction'],
                    rnn_hidden  = model_cfg['rnn_hidden'],
                    z_dim        = model_cfg['latent_dim']
                )
                    # Fallback: if configs are missing, defer to base builder
            else:
                return super().build(name, *args, **kwargs)
        # Fallback for other network types
        return super().build(name, *args, **kwargs)

    def load(self, params):
        """
        Initialize builder with full params, build network, and load its weights.

        Expects params dict with structure:
            {
                'network': {
                    'name': <network name>,
                    'state_dict': <state_dict>
                },
                'env': { ... },
                'model': { ... }
            }

        Returns:
            network (nn.Module): built network with loaded weights
        """
        # 전체 설정 저장
        #self.params = params
        net_cfg = params.get('network', {})
        name = net_cfg.get('name')

        # Build network instance with env/model configs
        network = self.build(name, **params)

        # Load weights if provided
        state_dict = net_cfg.get('state_dict', {})
        if state_dict:
            network.load_state_dict(state_dict)

        return network
