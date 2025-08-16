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
        self.config = {}
        self.exp_config = None
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
        self.network_builder = network_builder_dyros.A2CDYROSBuilder() #이름 맞추기

        self.algo_observer = algo_observer

        torch.backends.cudnn.benchmark = True

    def reset(self):
        pass

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

        # runner config에 있는 env_name을 우선 쓰고, 없으면 task 이름으로 기본값 지정
        # params.env_name (rlgpu) → config.env_name → task 순으로 우선순위 지정
        self.config['env_name'] = core.get(
            'env_name',                             # DyrosDynamicWalkPPO.yaml의 params.env_name
            full_conf.get('config', {}).get(        # (혹시 config 블록 안에 있다면)
                'env_name',
                full_conf.get('task', 'DyrosDynamicWalk')  # fallback: task 그룹 이름
            )
        )

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
        self.config['model'] = mdl

        # ─── 3) 환경(env) 파라미터 ──────────────────────────
        task_cfg = full_conf.get('task', {})
        env_cfg  = task_cfg.get('env', {})

        self.config['env'] = env_cfg
        self.config['env_name'] = self.config.get(
            'env_name', full_conf.get('config', {}).get('env_name', 'isaacgym')
        )
        self.config['env_config'] = env_cfg

        # === 병렬 환경 수(러너) → 반드시 env.NumEnvs와 동일 ===
        num_envs = int(env_cfg.get('numEnvs', env_cfg.get('num_envs', 4096)))
        self.config['num_actors'] = num_envs
        env_cfg['numEnvs'] = num_envs
        print(f"[DYROS][runner] num_actors = {self.config['num_actors']} (from env)")

        # === 관측/행동/priv 차원 확정 ===
        # 관측은 환경이 이미 height 임베딩을 합쳐 넣은 '최종' 차원을 그대로 사용
        obs_dim  = int(env_cfg.get('numObservations', env_cfg.get('NumSingleStepObs', 61)))
        act_dim  = int(env_cfg.get('NumAction', env_cfg.get('numActions', 13)))
        # priv_dim은 환경에서 계산한 최종 길이가 있으면 우선 사용
        priv_dim = int(env_cfg.get('numStates', env_cfg.get('priv_dim', 187)))

        self.config['input_shape'] = (obs_dim,)
        self.config['actions_num'] = act_dim
        self.config['priv_dim']    = priv_dim
        print(f"[DYROS][runner] obs_dim={obs_dim}, act_dim={act_dim}, priv_dim={priv_dim}")

        # ─── 4) 네트워크 차원 ───────────────────────────────
        self.config['rnn_hidden'] = int(mdl.get('rnn_hidden', 256))
        self.config['z_dim']      = int(mdl.get('latent_dim', 24))

        # ─── 5) 나머지 블록 병합 ───────────────────────────
        self.config['algo']    = self.algo_params
        self.config['ppo']     = self.algo_params

        # 모델/네트워크에도 확정 차원 “강제 주입”
        mdl.setdefault('priv_dim', priv_dim)
        mdl.setdefault('actions_num', act_dim)
        self.config['model']   = mdl

        net_cfg = core.get('network', {})
        net_params = net_cfg.setdefault('params', {})
        net_params['priv_dim']     = priv_dim
        net_params['actions_num']  = act_dim
        self.config['network'] = net_cfg

        # full_conf 최상위에 있는 학습률, 감가율 등 키들을 config에 복사합니다.
        general_cfg = core.get('config', {})
        for key, val in general_cfg.items():
            # hydra/runner/params/task/vecenv 블록은 제외
            if key in ['hydra', 'runner', 'params', 'task', 'vecenv']:
                continue
            # 이미 정의한 키(input_shape, env 등)도 덮어쓰지 않도록
            if key in self.config:
                continue
            self.config[key] = val

        # ─── 6) 기타 기본값 설정 ───────────────────────────
        self.config.setdefault('lr_schedule',         'linear')
        self.config.setdefault('learning_rate',       1e-5)
        self.config.setdefault('use_estimator_network', False)

        # ─── 7) reward_shaper, num_actors ──────────────────
        self.config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(
            **core['config']['reward_shaper']
        )
        #self.config['num_actors'] = env_cfg.get('num_envs', 4096)
        self.config['num_actors'] = int(env_cfg.get('numEnvs', env_cfg.get('num_envs', 4096)))
        

        # ← 여기서 최종 확정: config 우선, 없으면 model 값, 그래도 없으면 0.0
        self.config['reg_loss_coef'] = general_cfg.get(
            'reg_loss_coef', mdl.get('reg_loss_coef', self.config.get('reg_loss_coef', 0.0))
        )

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
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # agent.model은 ModelA2CContinuousLogStdDYROS.Network 인스턴스입니다.
                if hasattr(agent, 'model') and isinstance(agent.model, torch.nn.Module):
                    agent.model.to(device)
                # 학습 실행
                ####################################test if exact model is used
                # ==== DYROS DEBUG (CLEAN) ====
                import inspect, sys
                from importlib import import_module

                mods = {
                    "dyros_dynamic_walk"   : "isaacgymenvs.tasks.dyros_dynamic_walk",
                    "dyros_reward_v2"      : "isaacgymenvs.tasks.dyros_reward_v2",
                    "height_cnn"           : "isaacgymenvs.utils.height_cnn",
                    "actor_critic_dyros"   : "isaacgymenvs.utils.actor_critic_dyros",
                    "models_dyros"         : "isaacgymenvs.learning.rl_games_custom.models_dyros",
                    "network_builder_dyros": "isaacgymenvs.learning.rl_games_custom.network_builder_dyros",
                    "a2c_common_dyros"     : "isaacgymenvs.learning.rl_games_custom.a2c_common_dyros",
                    "rlgames_utils"        : "isaacgymenvs.utils.rlgames_utils",
                    "torch_runner_dyros"   : "isaacgymenvs.learning.rl_games_custom.torch_runner_dyros",
                }
                print("\n===== DYROS DEBUG =====")
                print("[MODULES]")
                for alias, modname in mods.items():
                    m = import_module(modname)
                    try:
                        path = inspect.getfile(m)
                    except Exception:
                        path = "<builtin/namespace>"
                    print(f"  - {alias:22s} : {path}")

                def _get(d, path, default=None):
                    cur = d
                    for k in path.split('.'):
                        try:
                            if isinstance(cur, dict):
                                cur = cur.get(k, default)
                            else:
                                cur = cur[k] if k in cur else getattr(cur, k)
                        except Exception:
                            return default
                    return cur

                cfg = self.config
                print("[CFG]")
                for label, key in [
                    ("network.name",               "network.name"),
                    ("model.latent_dim",           "model.latent_dim"),
                    ("reg_loss_coef",              "reg_loss_coef"),          # ← top-level
                    ("policy_coef",                "policy_coef"),
                    ("value_coef",                 "value_coef"),
                    ("entropy_coef",               "entropy_coef"),
                    ("env.NumSingleStepObs",       "env.NumSingleStepObs"),   # ← env.*
                    ("env.NumAction",              "env.NumAction"),
                    ("env.height_emb_dim",         "env.height_emb_dim"),
                    ("env.reward_version",         "env.reward_version"),
                ]:
                    print(f"  - {label:28s} : {_get(cfg, key)}")

                m = getattr(agent, 'model', None)
                print("[MODEL]")
                if m is None:
                    print("  - model: <none>")
                else:
                    print(f"  - class                : {m.__class__.__module__}.{m.__class__.__name__}")
                    try:
                        logstd = dict(m.named_parameters()).get("log_std", None)
                        if logstd is not None:
                            shp = tuple(logstd.shape)
                            smp = [round(x,4) for x in logstd.detach().flatten()[:4].cpu().tolist()]
                            print(f"  - log_std.shape        : {shp}  sample:{smp}")
                    except Exception:
                        pass
                    for attr in ("latent_dim","obs_shape","state_shape","action_space"):
                        if hasattr(m, attr):
                            print(f"  - {attr:20s} : {getattr(m, attr)}")

                print("[SYS]")
                print(f"  - sys.path[0]          : {sys.path[0]}")
                print("===== END =====\n")
                # ==== END DYROS DEBUG ====

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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if hasattr(agent, 'model') and isinstance(agent.model, torch.nn.Module):
                agent.model.to(device)
            if self.load_check_point and (self.load_path is not None):
                agent.restore(self.load_path)
            if self.load_check_point and (self.load_path is not None):
                agent.restore(self.load_path)


            # ==== DYROS DEBUG (CLEAN) ====
            import inspect, sys
            from importlib import import_module

            mods = {
                "dyros_dynamic_walk"   : "isaacgymenvs.tasks.dyros_dynamic_walk",
                "dyros_reward_v2"      : "isaacgymenvs.tasks.dyros_reward_v2",
                "height_cnn"           : "isaacgymenvs.utils.height_cnn",
                "actor_critic_dyros"   : "isaacgymenvs.utils.actor_critic_dyros",
                "models_dyros"         : "isaacgymenvs.learning.rl_games_custom.models_dyros",
                "network_builder_dyros": "isaacgymenvs.learning.rl_games_custom.network_builder_dyros",
                "a2c_common_dyros"     : "isaacgymenvs.learning.rl_games_custom.a2c_common_dyros",
                "rlgames_utils"        : "isaacgymenvs.utils.rlgames_utils",
                "torch_runner_dyros"   : "isaacgymenvs.learning.rl_games_custom.torch_runner_dyros",
            }
            print("\n===== DYROS DEBUG =====")
            print("[MODULES]")
            for alias, modname in mods.items():
                m = import_module(modname)
                try:
                    path = inspect.getfile(m)
                except Exception:
                    path = "<builtin/namespace>"
                print(f"  - {alias:22s} : {path}")

            def _get(d, path, default=None):
                cur = d
                for k in path.split('.'):
                    try:
                        if isinstance(cur, dict):
                            cur = cur.get(k, default)
                        else:
                            cur = cur[k] if k in cur else getattr(cur, k)
                    except Exception:
                        return default
                return cur

            cfg = self.config
            print("[CFG]")
            for label, key in [
                ("network.name",                 "network.name"),
                ("model.latent_dim",            "model.latent_dim"),
                ("reg_loss_coef",              "reg_loss_coef"),
                ("policy_coef",                 "policy_coef"),
                ("value_coef",                  "value_coef"),
                ("entropy_coef",                "entropy_coef"),
                ("env.NumSingleStepObs",       "env.NumSingleStepObs"),
                ("task.env.NumAction",          "env.NumAction"),
                ("task.env.height_emb_dim",     "env.height_emb_dim"),
                ("task.env.reward_version",     "env.reward_version"),
            ]:
                print(f"  - {label:28s} : {_get(cfg, key)}")

            m = getattr(agent, 'model', None)
            print("[MODEL]")
            if m is None:
                print("  - model: <none>")
            else:
                print(f"  - class                : {m.__class__.__module__}.{m.__class__.__name__}")
                try:
                    logstd = dict(m.named_parameters()).get("log_std", None)
                    if logstd is not None:
                        shp = tuple(logstd.shape)
                        smp = [round(x,4) for x in logstd.detach().flatten()[:4].cpu().tolist()]
                        print(f"  - log_std.shape        : {shp}  sample:{smp}")
                except Exception:
                    pass
                for attr in ("latent_dim","obs_shape","state_shape","action_space"):
                    if hasattr(m, attr):
                        print(f"  - {attr:20s} : {getattr(m, attr)}")

            print("[SYS]")
            print(f"  - sys.path[0]          : {sys.path[0]}")
            print("===== END =====\n")
            # ==== END DYROS DEBUG ====


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