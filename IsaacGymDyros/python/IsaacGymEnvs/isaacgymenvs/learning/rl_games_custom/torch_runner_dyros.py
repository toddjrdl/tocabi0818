import numpy as np
import copy
import torch
import yaml

from rl_games import envs
from rl_games.common import object_factory
from rl_games.common import env_configurations
from rl_games.common import experiment
from rl_games.common import tr_helpers

from isaacgymenvs.learning.rl_games_custom import network_builder_dyros
from isaacgymenvs.learning.rl_games_custom import a2c_continuous_seperate
from isaacgymenvs.learning.rl_games_custom import model_builder_dyros
from isaacgymenvs.learning.rl_games_custom.a2c_common_dyros import ContinuousA2CBase

from rl_games.algos_torch import a2c_discrete
from rl_games.algos_torch import players
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.algos_torch import sac_agent

class RunnerDyros:
    def __init__(self, algo_observer=None):
        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : ContinuousA2CBase(**kwargs))# 수정됨
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : a2c_discrete.DiscreteA2CAgent(**kwargs)) 
        self.algo_factory.register_builder('sac', lambda **kwargs: sac_agent.SACAgent(**kwargs))
        #self.algo_factory.register_builder('dqn', lambda **kwargs : dqnagent.DQNAgent(**kwargs))

        self.player_factory = object_factory.ObjectFactory()
        self.player_factory.register_builder('a2c_continuous', lambda **kwargs : players.PpoPlayerContinuous(**kwargs))
        self.player_factory.register_builder('a2c_discrete', lambda **kwargs : players.PpoPlayerDiscrete(**kwargs))
        self.player_factory.register_builder('sac', lambda **kwargs : players.SACPlayer(**kwargs))
        #self.player_factory.register_builder('dqn', lambda **kwargs : players.DQNPlayer(**kwargs))

        self.model_builder = model_builder_dyros.ModelBuilderDyros()
        self.network_builder = network_builder_dyros.NetworkBuilder()

        self.algo_observer = algo_observer

        torch.backends.cudnn.benchmark = True

    def reset(self):
        pass
    '''
    def load_config(self, params):
        if 'params' in params and isinstance(params['params'], dict):
            params = params['params']

        self.seed = params.get('seed', None)

        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.load_check_point = params['load_checkpoint']
        self.exp_config = None

        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)

        if self.load_check_point:
            print('Found checkpoint')
            print(params['load_path'])
            self.load_path = params['load_path']

        #self.model = self.model_builder.load(params)
        #self.config = copy.deepcopy(params['config'])
        
        #self.config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**self.config['reward_shaper'])
        self.config = copy.deepcopy(params)
      
        task_cfg = params.get('task', {})
        env_cfg  = task_cfg.get('env', {})
        # 2) 네트워크 빌더용 인자들 주입
        self.config['input_shape']  = (env_cfg['NumSingleStepObs'],)
        self.config['priv_dim']     = env_cfg.get('priv_dim', 181)
        self.config['actions_num']  = env_cfg['NumAction']

        mdl = self.config.get('model', {})
        self.config['rnn_hidden']   = mdl.get('rnn_hidden', 256)
        self.config['z_dim']        = mdl.get('latent_dim', 24)



        self.config['algo']    = params.get('algo', {})
        self.config['ppo']     = self.config['algo']
        self.config['model']   = params.get('model', {})
        self.config['network'] = params.get('network', {})

        self.config.setdefault('lr_schedule',        'linear')
        self.config.setdefault('learning_rate',      1e-4)
        self.config.setdefault('use_estimator_network', False)
        
        if 'env_name' not in self.config:
            # Hydra override로 넘어온 task key를 쓰거나, 기본값 지정
            self.config['env_name'] = params.get('task', 'DyrosDynamicWalk')
        self.config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(
            **self.config['reward_shaper']
        )
        self.config['num_actors'] = self.config.get('env', {}).get('num_envs', 4096)

        #self.model = self.model_builder.load(self.config)
        #if hasattr(self.model, 'build'):
        #    self.model.build(self.config)
        # 
        #self.config['network'] = self.model
        
        has_rnd_net = 'rnd_config' in self.config
        if has_rnd_net:
            print('Adding RND Network')
            rnd_net_cfg = self.config['rnd_config']['network']
            network = self.model_builder.network_factory.create(rnd_net_cfg['name'])
            network.load(rnd_net_cfg)
            self.config['rnd_config']['network'] = network
        
        has_central_value_net = 'central_value_config' in self.config
        if has_central_value_net:
            print('Adding Central Value Network')
            cv_cfg = self.config['central_value_config']['network']
            network = self.model_builder.network_factory.create(cv_cfg['name'])
            network.load(cv_cfg)
            self.config['central_value_config']['network'] = network
    '''
    def load_config(self, full_conf):
        """
        full_conf: Hydra가 로드한 전체 설정(dict)
        core      = full_conf.get('params', full_conf)
                    여기 안에 seed, algo, model, network, load_checkpoint, load_path 등이 들어갑니다.
        task_cfg  = full_conf.get('task', {})
                    여기 안에 env 블록(DyrosDynamicWalk.yaml의 env:)이 들어있습니다.
        """
        # ─── 0) core 언랩 ───────────────────────────────────
        core = full_conf.get('params', full_conf)

        # ─── 1) 시드, 체크포인트 로드 여부 ────────────────────
        self.seed             = core.get('seed', None)
        self.load_check_point = core.get('load_checkpoint', False)
        if self.load_check_point:
            print('Found checkpoint')
            print(core.get('load_path'))

        # ─── 2) 알고리즘/모델 정보 ───────────────────────────
        self.algo_params = core['algo']
        self.algo_name   = self.algo_params['name']
        mdl             = core.get('model', {})

        # ─── 3) 환경(env) 파라미터 ──────────────────────────
        task_cfg = full_conf.get('task', {})
        env_cfg  = task_cfg.get('env', {})
        # proprio(37) + heightCNN(24) = 61
        self.config['input_shape'] = (env_cfg['NumSingleStepObs'],)
        self.config['priv_dim']    = env_cfg.get('priv_dim', 181)
        self.config['actions_num'] = env_cfg['NumAction']

        # ─── 4) 네트워크 차원 ───────────────────────────────
        self.config['rnn_hidden'] = mdl.get('rnn_hidden', 256)
        self.config['z_dim']      = mdl.get('latent_dim', 24)

        # ─── 5) 나머지 블록 병합 ───────────────────────────
        self.config['algo']    = self.algo_params
        self.config['ppo']     = self.algo_params
        self.config['model']   = mdl
        self.config['network'] = core.get('network', {})

        # ─── 6) 기타 기본값 설정 ───────────────────────────
        self.config.setdefault('lr_schedule',         'linear')
        self.config.setdefault('learning_rate',       1e-4)
        self.config.setdefault('use_estimator_network', False)

        # ─── 7) reward_shaper, num_actors ──────────────────
        self.config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(
            **core['config']['reward_shaper']
        )
        self.config['num_actors'] = env_cfg.get('num_envs', 4096)

        # ─── 8) RND / Central Value 네트워크 로딩 ──────────
        if 'rnd_config' in core:
            rnd_cfg = core['rnd_config']['network']
            net = self.model_builder.network_factory.create(rnd_cfg['name'])
            net.load(rnd_cfg)
            self.config['rnd_config']['network'] = net

        if 'central_value_config' in core:
            cv_cfg = core['central_value_config']['network']
            net   = self.model_builder.network_factory.create(cv_cfg['name'])
            net.load(cv_cfg)
            self.config['central_value_config']['network'] = net

    def load(self, yaml_conf):
        self.default_config = yaml_conf
        self.load_config(copy.deepcopy(self.default_config))

        if 'experiment_config' in yaml_conf:
            self.exp_config = yaml_conf['experiment_config']

    def get_prebuilt_config(self):
        return self.config

    def run_train(self):
        print('Started to train')
        if self.algo_observer is None:
            self.algo_observer = DefaultAlgoObserver()

        if self.exp_config:
            self.experiment = experiment.Experiment(self.default_config, self.exp_config)
            exp_num = 0
            exp = self.experiment.get_next_config()
            while exp is not None:
                exp_num += 1
                print('Starting experiment number: ' + str(exp_num))
                self.reset()
                self.load_config(exp)
                if 'features' not in self.config:
                    self.config['features'] = {}
                self.config['features']['observer'] = self.algo_observer
                #if 'soft_augmentation' in self.config['features']:
                #    self.config['features']['soft_augmentation'] = SoftAugmentation(**self.config['features']['soft_augmentation'])
                agent = self.algo_factory.create(self.algo_name, base_name='run', config=self.config)  
                self.experiment.set_results(*agent.train())
                exp = self.experiment.get_next_config()
        else:
            self.reset()
            self.load_config(self.default_config)
            if 'features' not in self.config:
                self.config['features'] = {}
            self.config['features']['observer'] = self.algo_observer
            #if 'soft_augmentation' in self.config['features']:
            #    self.config['features']['soft_augmentation'] = SoftAugmentation(**self.config['features']['soft_augmentation'])
            agent = self.algo_factory.create(self.algo_name, base_name='run', config=self.config)  
            if self.load_check_point and (self.load_path is not None):
                agent.restore(self.load_path)
            agent.train()
            
    def create_player(self):
        return self.player_factory.create(self.algo_name, config=self.config)

    def create_agent(self, obs_space, action_space):
        return self.algo_factory.create(self.algo_name, base_name='run', observation_space=obs_space, action_space=action_space, config=self.config)

    def run(self, args):
        if 'checkpoint' in args and args['checkpoint'] is not None:
            if len(args['checkpoint']) > 0:
                self.load_path = args['checkpoint']

        if args['train']:
            self.run_train()
        elif args['play']:
            print('Started to play')
            player = self.create_player()
            player.restore(self.load_path)
            player.model.to("cpu")
            for name, param in player.model.state_dict().items():
                name= name.replace(".","_")
                weight_file_name = "./result/" + name + ".txt"
                np.savetxt(weight_file_name, param.data)
            player.model.to("cuda:0")
            player.run()
        else:
            self.run_train()