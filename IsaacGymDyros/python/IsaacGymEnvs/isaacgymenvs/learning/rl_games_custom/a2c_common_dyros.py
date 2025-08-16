import os

from rl_games.common import tr_helpers
from rl_games.common import vecenv
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch.moving_mean_std import MovingMeanStd
from rl_games.algos_torch.self_play_manager import SelfPlayManager
from rl_games.algos_torch import torch_ext
from rl_games.common import schedulers
from rl_games.common.experience import ExperienceBuffer
from rl_games.common.interval_summary_writer import IntervalSummaryWriter
import numpy as np
import collections
import time
from collections import deque, OrderedDict
import gym
from isaacgymenvs.cfg.terrain.terrain_cfg import TerrainCfg

from datetime import datetime
from tensorboardX import SummaryWriter
import torch 
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss
import torch.optim as optim
 
from time import sleep
import wandb


def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


class A2CBase:
    def __init__(self, base_name, config):
        pbt_str = ''

        self.population_based_training = config.get('population_based_training', False)
        if self.population_based_training:
            # in PBT, make sure experiment name contains a unique id of the policy within a population
            pbt_str = f'_pbt_{config["pbt_idx"]:02d}'

        # This helps in PBT when we need to restart an experiment with the exact same name, rather than
        # generating a new name with the timestamp every time.
        full_experiment_name = config.get('full_experiment_name', None)
        if full_experiment_name:
            print(f'Exact experiment name requested from command line: {full_experiment_name}')
            self.experiment_name = full_experiment_name
        else:
            self.experiment_name = config['name'] + pbt_str + datetime.now().strftime("_%d-%H-%M-%S")

        self.config = config
        self.terrain_cfg = TerrainCfg()

        self.algo_observer = config['features']['observer']
        self.algo_observer.before_init(base_name, config, self.experiment_name)

        self.multi_gpu = config.get('multi_gpu', False)
        self.rank = 0
        self.rank_size = 1
        self.curr_frames = 0
        if self.multi_gpu:
            from rl_games.distributed.hvd_wrapper import HorovodWrapper
            self.hvd = HorovodWrapper()
            self.config = self.hvd.update_algo_config(config)
            self.rank = self.hvd.rank
            self.rank_size  = self.hvd.rank_size


        self.network_path = config.get('network_path', "./nn/")
        self.log_path = config.get('log_path', "runs/")
        self.env_config = config.get('env_config', {})
        self.num_actors = config['num_actors']
        self.env_name = config['env_name']

        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()

        self.ppo_device = config.get('device', 'cuda:0')
        print('Env info:')
        print(self.env_info)
        self.value_size = self.env_info.get('value_size',1)
        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)

        self.central_value_config = self.config.get('central_value_config', None)
        self.has_central_value = self.central_value_config is not None
        self.truncate_grads = self.config.get('truncate_grads', False)
        if self.has_central_value:
            ss = self.env_info.get('state_space', None)
            if ss is None:
                # state_space 정보가 없다면 priv_dim으로 fallback
                self.state_shape = (self.config.get('priv_dim'),)
            else:
                self.state_shape = ss.shape            
#            self.state_space = self.env_info.get('state_space', None)
#            if isinstance(self.state_space,gym.spaces.Dict):
#                self.state_shape = {}
#                for k,v in self.state_space.spaces.items():
#                    self.state_shape[k] = v.shape
#            else:
#                self.state_shape = self.state_space.shape

        self.self_play_config = self.config.get('self_play_config', None)
        self.has_self_play_config = self.self_play_config is not None

        self.self_play = config.get('self_play', False)
        self.save_freq = config.get('save_frequency', 0)
        self.save_best_after = config.get('save_best_after', 100)
        self.print_stats = config.get('print_stats', True)
        self.rnn_states = None
        self.name = base_name



        self.ppo = config['ppo']
        self.use_estimator_network = config['use_estimator_network'] # DH: for base velocity learning
        self.max_epochs = self.config.get('max_epochs', 1e6)

        self.is_adaptive_lr = config['lr_schedule'] == 'adaptive'
        self.linear_lr = config['lr_schedule'] == 'linear'
        self.schedule_type = config.get('schedule_type', 'legacy')
        if self.is_adaptive_lr:
            self.kl_threshold = config['kl_threshold']
            self.scheduler = schedulers.AdaptiveScheduler(self.kl_threshold)
        elif self.linear_lr:
            self.scheduler = schedulers.LinearScheduler(float(config['learning_rate']),
                min_lr=float(config['learning_rate_min']), 
                max_steps=self.max_epochs, 
                apply_to_entropy=config.get('schedule_entropy', False),
                start_entropy_coef=config.get('entropy_coef'))
        else:
            self.scheduler = schedulers.IdentityScheduler()

        self.e_clip = config['e_clip']
        self.clip_value = config['clip_value']
        self.network = config['network']
        self.rewards_shaper = config['reward_shaper']
        self.num_agents = self.env_info.get('agents', 1)
        self.horizon_length = config['horizon_length']
        self.seq_len = self.config.get('seq_length', 4)
        self.normalize_advantage = config['normalize_advantage']
        self.normalize_input = self.config['normalize_input']
        # set_eval() / set_train() 에서 사용
        self.running_mean_std = getattr(self, 'running_mean_std', None)
        self.value_mean_std   = getattr(self, 'value_mean_std', None)
        self.normalize_value = self.config.get('normalize_value', False)
        self.truncate_grads = self.config.get('truncate_grads', False)
        self.has_phasic_policy_gradients = False

        if isinstance(self.observation_space,gym.spaces.Dict):
            self.obs_shape = {}
            for k,v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
        
        self.critic_coef = config['critic_coef']
        self.grad_norm = config['grad_norm']
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']

        self.games_to_track = self.config.get('games_to_track', 100)
        self.game_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)

        #GENE : for wandb logging
        self.wandb_activate = config.get('wandb_activate',False)
        self.log_wandb_frequency = config.get('log_wandb_frequency',-1)
        self.init_wandb = False
        self.num_rewards = config.get('num_rewards',1) + config.get('num_extra_logging',1)
        if (self.terrain_cfg.curriculum and self.terrain_cfg.mesh_type in ["heightfield", "trimesh"]):
            self.num_rewards += self.terrain_cfg.num_cols
        self.reward_names = []
        self.current_specific_rewards = torch.zeros((self.num_actors*self.num_agents,self.num_rewards),dtype=torch.float32, device=self.ppo_device)

        #GENE : for adjusting model
        self.sep_op = config.get('separate_opt',False)
        
        self.obs = None
        self.games_num = self.config['minibatch_size'] // self.seq_len # it is used only for current rnn implementation
        self.batch_size = self.horizon_length * self.num_actors * self.num_agents
        self.batch_size_envs = self.horizon_length * self.num_actors
        self.minibatch_size = self.config['minibatch_size']
        self.mini_epochs_num = self.config['mini_epochs']
        self.num_minibatches = self.batch_size // self.minibatch_size
        assert(self.batch_size % self.minibatch_size == 0)

        self.mixed_precision = self.config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.last_lr = self.config['learning_rate']
        self.frame = 0
        self.update_time = 0
        self.mean_rewards = self.last_mean_rewards = -100500
        self.play_time = 0
        self.epoch_num = 0

        # allows us to specify a folder where all experiments will reside
        self.train_dir = config.get('train_dir', 'runs')

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.entropy_coef = self.config['entropy_coef']

        if self.rank == 0:
            writer = SummaryWriter(self.summaries_dir)

            if self.population_based_training:
                self.writer = IntervalSummaryWriter(writer, self.config)
            else:
                self.writer = writer

        else:
            self.writer = None

        self.value_bootstrap = self.config.get('value_bootstrap')

        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((1,)).to(self.ppo_device)

        self.is_tensor_obses = False

        self.last_rnn_indices = None
        self.last_state_indices = None

        #self_play
        if self.has_self_play_config:
            print('Initializing SelfPlay Manager')
            self.self_play_manager = SelfPlayManager(self.self_play_config, self.writer)

        # features
        self.algo_observer = config['features']['observer']

        self.soft_aug = config['features'].get('soft_augmentation', None)
        self.has_soft_aug = self.soft_aug is not None
        # soft augmentation not yet supported
        assert not self.has_soft_aug

    def write_stats(self, total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames):
        # do we need scaled time?

        self.writer.add_scalar('performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
        self.writer.add_scalar('performance/step_inference_fps', curr_frames / scaled_play_time, frame)
        self.writer.add_scalar('performance/step_fps', curr_frames / step_time, frame)
        self.writer.add_scalar('performance/rl_update_time', update_time, frame)
        self.writer.add_scalar('performance/step_inference_time', play_time, frame)
        self.writer.add_scalar('performance/step_time', step_time, frame)
        self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(a_losses).item(), frame)
        self.writer.add_scalar('losses/c_loss', torch_ext.mean_list(c_losses).item(), frame)
                
        self.writer.add_scalar('losses/entropy', torch_ext.mean_list(entropies).item(), frame)
        self.writer.add_scalar('info/last_lr', last_lr, frame)
        self.writer.add_scalar('info/lr_mul', 1.0, frame)
        self.writer.add_scalar('info/e_clip', self.e_clip, frame)
        self.writer.add_scalar('info/kl', torch_ext.mean_list(kls).item(), frame)
        self.writer.add_scalar('info/epochs', epoch_num, frame)
        self.algo_observer.after_print_stats(frame, epoch_num, total_time)

    def set_eval(self):
        self.model.eval()
        rms = getattr(self, 'running_mean_std', None)
        if rms is not None:
            rms.eval()
        vms = getattr(self, 'value_mean_std', None)
        if vms is not None:
            vms.eval()


    def set_train(self):
        self.model.train()
        rms = getattr(self, 'running_mean_std', None)
        if rms is not None:
            rms.train()
        vms = getattr(self, 'value_mean_std', None)
        if vms is not None:
            vms.train()

    def update_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr])
            self.hvd.broadcast_value(lr_tensor, 'learning_rate')
            lr = lr_tensor.item()
        if self.sep_op:
            for param_group in self.optimizer_actor.param_groups:
                param_group['lr'] = lr
        
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
        #if self.has_central_value:
        #    self.central_value_net.update_lr(lr)

    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])

        if 'states' in obs and obs['states'] is not None:
            input_states = obs['states']
        else:
            batch = processed_obs.size(0)
            priv_dim = self.config.get('priv_dim', self.config.get('env', {}).get('priv_dim'))
            input_states = torch.zeros(batch, priv_dim, device=self.ppo_device)

        self.set_eval()
        
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'states'     : input_states,
            'rnn_states' : self.rnn_states
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                    #'actions' : res_dict['action'],
                    #'rnn_states' : self.rnn_states
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        return res_dict

    def get_values(self, obs):
        """
        Critic(V) 계산. 반드시 privileged state('states')를 함께 넣는다.
        - obs: dict( {'obs': ..., 'states': ...} ) 또는 텐서
        - vec_env 래퍼의 깊이가 달라도 states_buf를 최대한 찾아서 사용
        """
        with torch.no_grad():
            # 0) 관측 전처리
            if isinstance(obs, dict):
                obs_tensor = obs.get('obs', obs)
            else:
                obs_tensor = obs
            processed_obs = self._preproc_obs(obs_tensor)

            # 1) privileged states 찾기
            def _find_priv_from_anywhere():
                # 1) obs dict에 이미 있는 경우
                if isinstance(obs, dict) and ('states' in obs) and (obs['states'] is not None):
                    return obs['states']

                # 2) vec_env 계층을 넓게 순회
                roots = []
                ve = getattr(self, 'vec_env', None)
                roots += [ve]
                for r in list(roots):
                    if r is None:
                        continue
                    roots += [getattr(r, k, None) for k in ('env', 'vec_env', 'venv', 'unwrapped')]

                for r in filter(lambda x: x is not None, roots):
                    # (a) task.states_buf 직접 노출 시 바로 사용
                    if hasattr(r, 'task') and hasattr(r.task, 'states_buf'):
                        return r.task.states_buf
                    if hasattr(r, 'states_buf'):
                        return r.states_buf

                    # (b) get_privileged_obs가 있더라도 .task가 없으면 호출하지 않음
                    if hasattr(r, 'get_privileged_obs') and hasattr(r, 'task'):
                        try:
                            p = r.get_privileged_obs()
                            if p is not None:
                                return p
                        except AttributeError:
                            # 일부 래퍼는 .task가 없어 내부에서 AttributeError 발생 → 무시하고 다음 후보 탐색
                            pass
                return None

            priv = _find_priv_from_anywhere()
            if priv is None:
                raise KeyError("get_values(): privileged 'states'가 없습니다. "
                            "env에서 states_buf를 갱신/노출하도록 확인하세요.")

            # 2) 차원/디바이스 정리
            if priv.dim() == 3:
                priv = priv[:, -1, :]           # [B,T,D] → 현재 시점만 [B,D]
            elif priv.dim() == 1:
                priv = priv.unsqueeze(0)        # [D] → [1,D]
            priv = priv.to(processed_obs.device)

            # 3) RNN state 형식 통일(텐서 → 튜플)
            rnn_states = self.rnn_states
            if isinstance(rnn_states, torch.Tensor):
                rnn_states = (rnn_states,)

            # 4) 모델 호출
            input_dict = {
                'is_train': False,
                'prev_actions': None,
                'obs': processed_obs,
                'states': priv,
                'rnn_states': rnn_states,
            }

            if self.has_central_value:
                self.central_value_net.eval()
                value = self.get_central_value(input_dict)
            else:
                self.set_eval()
                result = self.model(input_dict)
                value = result['values']

            if self.normalize_value:
                value = self.value_mean_std(value, True)
            return value

    @property
    def device(self):
        return self.ppo_device

    def reset_envs(self):
        self.obs = self.env_reset()

    def init_tensors(self):
        batch_size = self.num_agents * self.num_actors
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
            'use_action_masks' : self.use_action_masks
        }
        self.experience_buffer = ExperienceBuffer(self.env_info, algo_info, self.ppo_device)

        val_shape = (self.horizon_length, batch_size, self.value_size)
        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)


        if self.is_rnn:
            self.rnn_states = self.model.get_default_rnn_state()
            self.rnn_states = [s.to(self.ppo_device) for s in self.rnn_states]

            batch_size = self.num_agents * self.num_actors
            num_seqs = self.horizon_length * batch_size // self.seq_len
            assert((self.horizon_length * batch_size // self.num_minibatches) % self.seq_len == 0)
            self.mb_rnn_states = [torch.zeros((s.size()[0], num_seqs, s.size()[2]), dtype = torch.float32, device=self.ppo_device) for s in self.rnn_states]

    def init_rnn_from_model(self, model):
        self.is_rnn = self.model.is_rnn()

    def init_rnn_step(self, batch_size, mb_rnn_states):
        mb_rnn_states = self.mb_rnn_states
        mb_rnn_masks = torch.zeros(self.horizon_length*batch_size, dtype = torch.float32, device=self.ppo_device)
        steps_mask = torch.arange(0, batch_size * self.horizon_length, self.horizon_length, dtype=torch.long, device=self.ppo_device)
        play_mask = torch.arange(0, batch_size, 1, dtype=torch.long, device=self.ppo_device)
        steps_state = torch.arange(0, batch_size * self.horizon_length//self.seq_len, self.horizon_length//self.seq_len, dtype=torch.long, device=self.ppo_device)
        indices = torch.zeros((batch_size), dtype = torch.long, device=self.ppo_device)
        return mb_rnn_masks, indices, steps_mask, steps_state, play_mask, mb_rnn_states

    def process_rnn_indices(self, mb_rnn_masks, indices, steps_mask, steps_state, mb_rnn_states):
        # 종료 조건
        if indices.max().item() >= self.horizon_length:
            return None, True

        # 콜렉션 마스크 갱신
        if mb_rnn_masks.dtype != torch.float32:
            mb_rnn_masks = mb_rnn_masks.float()
        mb_rnn_masks[indices + steps_mask] = 1

        # 메타
        B = int(self.num_agents) * int(self.num_actors)  # 보통 env 수
        assert B == int(indices.numel()), f"batch 불일치: indices={indices.numel()} vs B={B}"
        assert (self.horizon_length % self.seq_len) == 0, \
            f"horizon({self.horizon_length}) % seq_len({self.seq_len}) != 0"

        S = self.horizon_length // self.seq_len         # 한 actor당 조각 수
        N = B * S                                       # 전체 조각 수
        device = indices.device

        # 이번 스텝에서 '시작 프레임'(t % seq_len == 0)인 actor 선별
        seq_indices   = torch.remainder(indices, self.seq_len)      # (B,)
        start_mask    = (seq_indices == 0)
        state_indices = start_mask.nonzero(as_tuple=False).squeeze(-1)  # (N_start,)

        # 시작하는 actor가 없으면 종료(대부분 step에서 정상)
        if state_indices.numel() == 0:
            self.last_rnn_indices   = None
            self.last_state_indices = None
            return seq_indices, False

        # 각 actor의 조각 번호와 최종 시퀀스 인덱스 계산
        seq_id_per_actor = torch.div(indices, self.seq_len, rounding_mode='floor')  # (B,) in [0..S-1]
        actor_ids        = torch.arange(B, device=device, dtype=torch.long)         # (B,)
        actor_offsets    = actor_ids * S
        rnn_indices      = actor_offsets[state_indices] + seq_id_per_actor[state_indices]  # (N_start,)

        # 범위 가드
        if not (0 <= int(rnn_indices.min()) and int(rnn_indices.max()) < N):
            raise RuntimeError(
                f"rnn_indices 범위 오류: [{int(rnn_indices.min())}, {int(rnn_indices.max())}] vs N={N}"
            )

        # RNN state 복사: [L, B?or1, H] → [L, N, H]
        for li, (layer_state, layer_mb) in enumerate(zip(self.rnn_states, mb_rnn_states)):
            # 모양 검증
            assert layer_state.dim() == 3 and layer_mb.dim() == 3, \
                f"shape 오류: state={tuple(layer_state.shape)}, mb={tuple(layer_mb.shape)}"
            assert layer_mb.size(1) == N, \
                f"mb_rnn_states[{li}].shape[1]={layer_mb.size(1)} != N({N})"

            # 배치축이 1이면 B로 확장 (초기 스텝 호환)
            if layer_state.size(1) == 1 and B > 1:
                # expand는 view이므로 그대로 사용 OK(grad 불필요한 hidden cache)
                s_expanded = layer_state.expand(layer_state.size(0), B, layer_state.size(2))
            else:
                s_expanded = layer_state
                # B 검사(디버그용): 배치축이 1이거나 B와 동일해야 함
                assert s_expanded.size(1) in (1, B), \
                    f"unexpected rnn state batch: {s_expanded.size(1)} (expected 1 or {B})"

            # 시작하는 actor들 슬롯에 기록
            layer_mb[:, rnn_indices, :] = s_expanded[:, state_indices, :]

        self.last_rnn_indices   = rnn_indices
        self.last_state_indices = state_indices
        return seq_indices, False

    def process_rnn_dones(self, all_done_indices, indices, seq_indices):
        if len(all_done_indices) > 0:
            shifts = self.seq_len - 1 - seq_indices[all_done_indices]
            indices[all_done_indices] += shifts
            for s in self.rnn_states:
                s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0
        indices += 1  

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert(self.observation_space.dtype != np.int8)
            if self.observation_space.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.ppo_device)
            else:
                obs = torch.FloatTensor(obs).to(self.ppo_device)
        return obs

    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or 'obs' not in obs:    
            upd_obs = {'obs' : upd_obs}
        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions)

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos

    def env_reset(self):
        obs = self.vec_env.reset()
        obs = self.obs_to_tensors(obs)
        return obs

    def discount_values(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam
        return mb_advs

    def discount_values_masks(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards, mb_masks):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)
        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)
            masks_t = mb_masks[t].unsqueeze(1)
            delta = (mb_rewards[t] + self.gamma * nextvalues * nextnonterminal  - mb_extrinsic_values[t])
            mb_advs[t] = lastgaelam = (delta + self.gamma * self.tau * nextnonterminal * lastgaelam) * masks_t
        return mb_advs

    def clear_stats(self):
        batch_size = self.num_agents * self.num_actors
        self.game_rewards.clear()
        self.game_lengths.clear()

        self.mean_rewards = self.last_mean_rewards = -100500
        self.algo_observer.after_clear_stats()

    def update_epoch(self):
        self.epoch_num = getattr(self, 'epoch_num', 0) + 1
        return self.epoch_num

    def train(self):       
        pass

    def prepare_dataset(self, batch_dict):
        pass

    def train_epoch(self):
        self.vec_env.set_train_info(self.frame)
    

#################################################################################
    def train_actor_critic(self, obs_dict, opt_step=True):
        # 1) 입력 분해
        obs_seq      = obs_dict['obs']              # (B,T,obs_dim)
        priv_seq     = obs_dict['states']           # (B,T,priv_dim)
        hidden_state = obs_dict.get('rnn_states')   # (num_layers, B, hidden)

        # 2) 모델 호출: dict 형태로 전달
        model_in = {
            'is_train': True,
            # 'prev_actions'가 별도로 없으면, dataset의 'actions'를 바로 사용
            'prev_actions': obs_dict.get('prev_actions', obs_dict.get('actions')),
            'obs':        obs_seq,
            'states':     priv_seq,
            'rnn_states': hidden_state,
        }
        res = self.model(model_in)

        # 3) 필요한 값들 꺼내기
        mu       = res['mus']
        sigma    = res['sigmas']
        value    = res['values']
        # rnn_states는 (new_hidden,) 형태로 오므로 첫 원소만
        new_hidden = res.get('rnn_states', None)
        if isinstance(new_hidden, (list, tuple)):
            new_hidden = new_hidden[0]
        # denoise용 재구성 출력(모델에서 제공하면 사용)
        recon = res.get('recon', None)

        # 4) PPO 손실들
        policy_loss, value_loss, entropy, kl, neglogp = self._compute_ppo_losses(res, obs_dict)

        # 5) Denoising loss (decoder 출력 vs. 마지막 privileged state)
        if recon is not None:
            denoise_loss = F.mse_loss(recon, obs_dict['states'][:, -1, :])
        else:
            # 모델이 아직 'recon'을 안 돌려주면 denoise는 0으로(학습 진행)
            denoise_loss = value.detach().sum() * 0.0

        # 6) 전체 손실: λᵣ·L_den + L_π + λᵥ·Lᵥ
        λr = self.config['reg_loss_coef']
        λπ = self.config.get('policy_coef', 5.0) # 논문 값
        λv = self.config.get('critic_coef',  5.0) # 논문 값
        total_loss = λr * denoise_loss + λπ * policy_loss + λv * value_loss - self.entropy_coef * entropy
        
        # 7) backward & step
        self.optimizer.zero_grad()
        total_loss.backward()
        if opt_step:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            self.optimizer.step()

        # 8) 로깅용 값 반환(기존 형식 유지)
        return (policy_loss.detach(), value_loss.detach(), entropy.detach(), kl.detach(),
                self.last_lr, self.last_lr_mu,
                mu.detach(), sigma.detach(),
                denoise_loss.detach(), self.curr_clip_frac)
    #################################################################################
    def _compute_ppo_losses(self, res, batch):
        import torch
        import numpy as np

        # --- 모델 출력 ---
        mu    = res['mus']          # [B,L,A]
        sigma = res['sigmas']       # [B,L,A]
        value = res['values']       # [B,L] or [B,L,1]
        if value.dim() == 3 and value.size(-1) == 1:
            value = value.squeeze(-1)   # -> [B,L]

        B, L, A = mu.shape
        N = B * L

        # --- actions 먼저 확보(배치 우선, 없으면 res) ---
        actions = batch.get('actions', None)
        if actions is None:
            actions = res.get('actions', None)
        if actions is None:
            raise RuntimeError("compute_ppo_losses: 'actions' not found in batch/res")

        if actions.dim() == 2 and actions.size(0) == N and actions.size(1) == A:
            actions = actions.view(B, L, A)
        elif actions.dim() != 3:
            actions = actions.reshape(B, L, A)   # 마지막 안전망

        # --- 한 번만 평탄화: 스칼라류 -> [N], 벡터류 -> [N,A] ---
        def flat1(x, name):
            if x is None: return None
            if x.dim() == 3 and x.size(-1) == 1:  # [B,L,1] -> [B,L]
                x = x.squeeze(-1)
            if x.dim() == 3 and x.size(-1) == A:  # [B,L,A] -> [B,L] (per-dim일 때)
                x = x.sum(dim=-1)
            x = x.reshape(-1)                      # -> [N]
            if x.numel() != N:
                raise RuntimeError(f"{name}: got {tuple(x.shape)}, expected {N}")
            return x

        # 벡터형(액션 차원 포함)은 [N,A], 스칼라형은 [N]
        mu      = mu.reshape(N, A)
        sigma   = sigma.reshape(N, A)
        actions = actions.reshape(N, A)
        value_f = flat1(value, 'value')

        returns    = flat1(batch['returns'], 'returns')
        advantages = flat1(batch['advantages'], 'advantages')
        old_values = flat1(batch.get('old_values', batch.get('values', value.detach())), 'old_values')

        old_neg = batch.get('neglogpacs', batch.get('old_neglogp', None))
        if old_neg is None:
            old_neg = torch.zeros(N, device=mu.device)
        else:
            old_neg = flat1(old_neg, 'old_neg')

        # --- new neglogp (모델이 주면 사용, 없으면 계산) ---
        if 'prev_neglogp' in res and res['prev_neglogp'] is not None:
            new_neg = flat1(res['prev_neglogp'], 'new_neg')
        else:
            log_std = torch.log(sigma)  # [N,A]
            new_neg = (
                0.5 * (((actions - mu) / sigma) ** 2).sum(dim=-1)
                + 0.5 * np.log(2.0 * np.pi) * A
                + log_std.sum(dim=-1)
            )  # [N]

        # --- 엔트로피 ---
        if 'entropy' in res:
            entropy = flat1(res['entropy'], 'entropy').mean()
        else:
            # Normal(mu,sigma) 엔트로피
            entropy = (torch.log(sigma).sum(dim=-1) + 0.5 * A * (1.0 + np.log(2.0 * np.pi))).mean()

        # --- 이점 정규화(옵션) ---
        adv = advantages
        if getattr(self, 'normalize_advantage', True):
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # --- PPO clipped objective ---
        # ratio = exp(logπ_new − logπ_old) = exp(old_neg − new_neg)  (neg = −logπ)
        ratio = torch.exp(old_neg - new_neg)          # [N]
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip) * adv
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        # 로깅용 clip frac
        self.curr_clip_frac = ((ratio > 1.0 + self.e_clip) | (ratio < 1.0 - self.e_clip)).float().mean()

        # --- value loss (clipped) ---
        if getattr(self, 'clip_value', True):
            v_clipped = old_values + (value_f - old_values).clamp(-self.e_clip, self.e_clip)
            v_loss1   = (value_f - returns) ** 2
            v_loss2   = (v_clipped - returns) ** 2
            value_loss = 0.5 * torch.mean(torch.max(v_loss1, v_loss2))
        else:
            value_loss = 0.5 * torch.mean((value_f - returns) ** 2)

        # 근사 KL: 0.5 * (Δlogπ)^2 = 0.5 * (new_neg − old_neg)^2
        approx_kl = 0.5 * torch.mean((new_neg - old_neg) ** 2)

        return policy_loss, value_loss, entropy, approx_kl, new_neg.mean()

    def calc_gradients(self):
        pass

    def get_central_value(self, obs_dict):
        return self.central_value_net.get_value(obs_dict)

    def train_central_value(self):
        return self.central_value_net.train_net()

    def get_full_state_weights(self):
        state = self.get_weights()
        state['epoch'] = self.epoch_num
        if self.sep_op:
            state['optimizer_actor'] = self.optimizer_actor.state_dict()
            state['optimizer_critic'] = self.optimizer_critic.state_dict()
        else:
            state['optimizer'] = self.optimizer.state_dict()
        if self.has_central_value:
            state['assymetric_vf_nets'] = self.central_value_net.state_dict()
        state['frame'] = self.frame

        # This is actually the best reward ever achieved. last_mean_rewards is perhaps not the best variable name
        # We save it to the checkpoint to prevent overriding the "best ever" checkpoint upon experiment restart
        state['last_mean_rewards'] = self.last_mean_rewards

        env_state = self.vec_env.get_env_state()
        state['env_state'] = env_state

        return state

    def set_full_state_weights(self, weights):
        self.set_weights(weights)
        self.epoch_num = weights['epoch']
        if self.has_central_value:
            self.central_value_net.load_state_dict(weights['assymetric_vf_nets'])
        if self.sep_op:
            self.optimizer_actor.load_state_dict(weights['optimizer_actor'])
            self.optimizer_critic.load_state_dict(weights['optimizer_critic'])        
        else:
            self.optimizer.load_state_dict(weights['optimizer'])
        self.frame = weights.get('frame', 0)
        self.last_mean_rewards = weights.get('last_mean_rewards', -100500)

        env_state = weights.get('env_state', None)
        self.vec_env.set_env_state(env_state)

    def get_weights(self):
        state = self.get_stats_weights()
        state['model'] = self.model.state_dict()
        return state

    def get_stats_weights(self):
        '''
        state = {}
        if self.normalize_input:
            state['running_mean_std'] = self.running_mean_std.state_dict()
        if self.normalize_value:
            state['reward_mean_std'] = self.value_mean_std.state_dict()
        if self.has_central_value:
            state['assymetric_vf_mean_std'] = self.central_value_net.get_stats_weights()
        if self.mixed_precision:
            state['scaler'] = self.scaler.state_dict()
        return state
        '''
        state = {}
        # 입력 정규화
        rms = getattr(self, 'running_mean_std', None)
        state['running_mean_std'] = rms.state_dict() if rms is not None else None
        # 가치/리워드 정규화
        vms = getattr(self, 'value_mean_std', None)
        state['reward_mean_std'] = vms.state_dict() if vms is not None else None
        return state

    def set_stats_weights(self, weights):
        '''
        if self.normalize_input:
            self.running_mean_std.load_state_dict(weights['running_mean_std'])
        if self.normalize_value:
            self.value_mean_std.load_state_dict(weights['reward_mean_std'])
        if self.has_central_value:
            self.central_value_net.set_stats_weights(weights['assymetric_vf_mean_std'])
        if self.mixed_precision and 'scaler' in weights:
            self.scaler.load_state_dict(weights['scaler'])
        '''
        # 입력 정규화
        rms_sd = weights.get('running_mean_std', None)
        if rms_sd is not None and getattr(self, 'running_mean_std', None) is not None:
            self.running_mean_std.load_state_dict(rms_sd)
        # 가치/리워드 정규화
        vms_sd = weights.get('reward_mean_std', None)
        if vms_sd is not None and getattr(self, 'value_mean_std', None) is not None:
            self.value_mean_std.load_state_dict(vms_sd)    

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        self.set_stats_weights(weights)
        
    def save(self, name):
        """체크포인트 저장: 모델 가중치 + 통계(RMS) 포함."""
        # 확장자 보정
        fname = str(name)
        if not fname.endswith('.pth'):
            fname += '.pth'
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        state = {
            'weights': self.get_weights(),   # {'model': state_dict, 'running_mean_std': ..., 'reward_mean_std': ...}
            'config': self.config,
            'epoch': getattr(self, 'epoch_num', None),
            'frame': getattr(self, 'frame', None),
        }
        import torch
        torch.save(state, fname)

    def restore(self, path):
        """체크포인트 로드(가능한 여러 포맷을 호환)."""
        import torch
        ckpt = torch.load(path, map_location=self.ppo_device)

        if isinstance(ckpt, dict) and 'weights' in ckpt:
            # 우리가 저장한 포맷
            self.set_weights(ckpt['weights'])
        elif isinstance(ckpt, dict) and 'model' in ckpt:
            # 최소 포맷(모델만)
            self.model.load_state_dict(ckpt['model'])
            stats = {}
            if 'running_mean_std' in ckpt: stats['running_mean_std'] = ckpt['running_mean_std']
            if 'reward_mean_std'  in ckpt: stats['reward_mean_std']  = ckpt['reward_mean_std']
            if stats: self.set_stats_weights(stats)
        else:
            # 순수 state_dict만 있는 경우
            self.model.load_state_dict(ckpt)

        if hasattr(self.model, 'to'):
            self.model.to(self.ppo_device)

    def _preproc_obs(self, obs_batch):
        '''
        if type(obs_batch) is dict:
            for k,v in obs_batch.items():
                obs_batch[k] = self._preproc_obs(v)
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        if self.normalize_input:
            obs_batch = self.running_mean_std(obs_batch)
        return obs_batch
        '''
            # normalize_input이 꺼져 있으면 running_mean_std는 None일 수 있음
        if getattr(self, 'running_mean_std', None) is not None:
            return self.running_mean_std(obs_batch)
        return obs_batch

    def play_steps(self):
        epinfos = []
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_asked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            if rewards.dim()==3 and rewards.size(1)==1:
                rewards = rewards.squeeze(1)      # → [4096,2]
            
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)

            if self.value_bootstrap and 'time_outs' in infos:
                vals  = res_dict['values'].squeeze(-1)               # [4096, 1]
                masks = self.cast_obs(infos['time_outs']).float()    # [4096]
                shaped_rewards[:,0] += self.gamma * vals * masks     # [4096] vs [4096] 브로드캐스트
                #shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            #self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards[:, 0:1])

            self.current_rewards += rewards[:, 0:1]
            self.current_lengths += 1

            #GENE: updating specific rewards for wandb
            if 'reward_names' in infos and 'stacked_rewards' in infos:
                self.reward_names = infos['reward_names']
                self.current_specific_rewards = infos['stacked_rewards']
            #######################################

            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            #GENE: Update specific rewards

            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            #GENE : resetting current specific rewards
            #print(self.current_specific_rewards)
            #self.current_specific_rewards = self.current_specific_rewards * not_dones.unsqueeze(1)

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        return batch_dict

    def play_steps_rnn(self):
        mb_rnn_states = []
        epinfos = []

        # 버퍼 초기화
        self.experience_buffer.tensor_dict['values'].fill_(0)
        self.experience_buffer.tensor_dict['rewards'].fill_(0)
        self.experience_buffer.tensor_dict['dones'].fill_(1)

        step_time = 0.0
        update_list = self.update_list

        batch_size = self.num_agents * self.num_actors

        # RNN 인덱스/마스크 초기화
        mb_rnn_masks, indices, steps_mask, steps_state, play_mask, mb_rnn_states = \
            self.init_rnn_step(batch_size, mb_rnn_states)

        # -------------------------------------------------
        # 수집 루프
        # -------------------------------------------------
        for n in range(self.horizon_length):
            seq_indices, full_tensor = self.process_rnn_indices(
                mb_rnn_masks, indices, steps_mask, steps_state, mb_rnn_states
            )
            if full_tensor:
                break

            if self.has_central_value:
                self.central_value_net.pre_step_rnn(self.last_rnn_indices, self.last_state_indices)

            # 액션/가치 계산
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            # 최신 RNN hidden 유지
            self.rnn_states = res_dict['rnn_states']

            # 관측/던/값 등 버퍼 기록
            self.experience_buffer.update_data_rnn('obses',  indices, play_mask, self.obs['obs'])
            self.experience_buffer.update_data_rnn('dones',  indices, play_mask, self.dones.byte())
            for k in update_list:
                # 보통 ['values','mus','sigmas','prev_neglogp','entropy','actions'] 등이 들어있음
                self.experience_buffer.update_data_rnn(k, indices, play_mask, res_dict[k])

            # 중앙가치망을 쓰는 경우 스텝별 privileged state도 저장(가능할 때만)
            if self.has_central_value:
                priv_now = None
                if isinstance(self.obs, dict) and ('states' in self.obs):
                    priv_now = self.obs['states']
                elif hasattr(self.vec_env, 'get_privileged_obs'):
                    priv_now = self.vec_env.get_privileged_obs()
                elif hasattr(self.vec_env, 'task') and hasattr(self.vec_env.task, 'states_buf'):
                    priv_now = self.vec_env.task.states_buf
                if priv_now is not None:
                    # 다중 에이전트 호환(rl-games의 중앙가치 버퍼 규약)
                    self.experience_buffer.update_data_rnn(
                        'states',
                        indices[::self.num_agents],
                        play_mask[::self.num_agents] // self.num_agents,
                        priv_now
                    )

            # 환경 스텝
            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()
            step_time += (step_time_end - step_time_start)
            ##########################################################################
            # v2 리워드 구성요소 수집
            if isinstance(infos, dict):
                if 'reward_names' in infos and 'stacked_rewards' in infos:
                    self.reward_names = infos['reward_names']
                    self.current_specific_rewards = infos['stacked_rewards']
                elif 'extras' in infos and isinstance(infos['extras'], dict):
                    ex = infos['extras']
                    if 'reward_names' in ex and 'stacked_rewards' in ex:
                        self.reward_names = ex['reward_names']
                        self.current_specific_rewards = ex['stacked_rewards']
            # ---------------------------
            # 보상 정규화 + timeout 부트스트랩 (항상 [B,1] 유지)
            # ---------------------------
            shaped_rewards = self.rewards_shaper(rewards)
            if shaped_rewards.dim() == 1:
                shaped_rewards = shaped_rewards.unsqueeze(1)           # [B] -> [B,1]
            elif shaped_rewards.dim() == 2 and shaped_rewards.size(1) != 1:
                shaped_rewards = shaped_rewards[:, :1]                 # [B,K] -> [B,1]

            if self.value_bootstrap and ('time_outs' in infos):
                to_mask = self.cast_obs(infos['time_outs'])
                if to_mask.dim() == 1:
                    to_mask = to_mask.unsqueeze(1)                     # [B] -> [B,1]
                elif to_mask.dim() == 2 and to_mask.size(1) != 1:
                    to_mask = to_mask[:, :1]                           # [B,2] -> [B,1] (단일 에이전트)
                else:
                    to_mask = to_mask.reshape(-1, 1)                   # 기타 -> [B,1]
                shaped_rewards = shaped_rewards + self.gamma * res_dict['values'] * to_mask.float()

            # ✅ 항상 보상을 기록(이 줄이 if 블록 밖에 있어야 함)
            self.experience_buffer.update_data_rnn('rewards', indices, play_mask, shaped_rewards)

            # 에피소드 통계/던 처리
            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.process_rnn_dones(all_done_indices, indices, seq_indices)
            if self.has_central_value:
                self.central_value_net.post_step_rnn(all_done_indices)

            self.algo_observer.process_infos(infos, done_indices)

            fdones = self.dones.float()
            not_dones = 1.0 - fdones

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        # -------------------------------------------------
        # 루프 종료 후 마지막 V(s) 및 어드밴티지 계산
        # -------------------------------------------------
        last_values = self.get_values(self.obs)                     # [B,1]

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']

        non_finished = (indices != self.horizon_length).nonzero(as_tuple=False)
        ind_to_fill = indices[non_finished]
        mb_fdones[ind_to_fill, non_finished] = fdones[non_finished]
        mb_values[ind_to_fill, non_finished] = last_values[non_finished]
        fdones[non_finished] = 1.0
        last_values[non_finished] = 0

        mb_advs = self.discount_values_masks(
            fdones, last_values, mb_fdones, mb_values, mb_rewards,
            mb_rnn_masks.view(-1, self.horizon_length).transpose(0, 1)
        )
        mb_returns = mb_advs + mb_values

        # 최신 privileged state 캐시 (prepare_dataset에서 못 찾을 경우 대비)
        try:
            if hasattr(self.vec_env, 'get_privileged_obs'):
                self.last_priv_states = self.vec_env.get_privileged_obs().detach().clone()
            elif hasattr(self.vec_env, 'task') and hasattr(self.vec_env.task, 'states_buf'):
                self.last_priv_states = self.vec_env.task.states_buf.detach().clone()
            elif isinstance(self.obs, dict) and ('states' in self.obs):
                self.last_priv_states = self.obs['states'].detach().clone()
            else:
                self.last_priv_states = None
        except Exception:
            self.last_priv_states = None

        # 배치 딕셔너리 구성
        batch_dict = self.experience_buffer.get_transformed_list(
            swap_and_flatten01, self.tensor_list
        )
        batch_dict['returns']     = swap_and_flatten01(mb_returns)
        batch_dict['rnn_states']  = mb_rnn_states
        batch_dict['rnn_masks']   = mb_rnn_masks
        batch_dict['played_frames'] = (n + 1) * self.num_actors * self.num_agents
        batch_dict['step_time']   = step_time

        # (선택) privileged를 함께 넘겨두면 prepare_dataset이 더 안전
        if self.last_priv_states is not None:
            batch_dict['states'] = self.last_priv_states

        return batch_dict


class ContinuousA2CBase(A2CBase):
    class _OneBatchDataset:
        def __init__(self, device, horizon_len, seq_len,
                    obses, states, actions, returns, old_values, advantages,
                    neglogpacs, mus, sigmas, rnn_states=None, rnn_masks=None):
            self.device = device
            self.horizon_len = horizon_len   # = T
            self.seq_len = seq_len           # = L
            self._B = None                   # B(=num_envs) 추론용

            def _reshape_to_seq(x, *, is_states=False):
                if x is None:
                    return None
                # [N] -> [N,1]
                if x.dim() == 1:
                    x = x.unsqueeze(-1)

                if x.dim() == 2:  # [N, D]
                    N, D = x.shape
                    # B 추론 (obs가 먼저 들어오므로 보통 여기서 B가 잡힘)
                    if self._B is None:
                        assert N % self.horizon_len == 0, \
                            f"N({N}) % horizon_len({self.horizon_len}) != 0"
                        self._B = N // self.horizon_len

                    # states가 [B, priv_dim]이면 T번 반복
                    if is_states and N == self._B:
                        x = x.unsqueeze(1).repeat(1, self.horizon_len, 1) \
                            .reshape(self._B * self.horizon_len, D)

                    # [B, T, D] -> [B*S, L, D]
                    B = self._B
                    T = self.horizon_len
                    assert x.shape[0] == B * T
                    S = T // self.seq_len
                    x = x.view(B, T, D).view(B * S, self.seq_len, D)
                    return x.to(self.device)

                if x.dim() == 3:  # [T,B,D] or [B,T,D]
                    if x.size(0) == self.horizon_len:  # [T,B,D] -> [B,T,D]
                        x = x.transpose(0, 1)
                    B, T, D = x.shape
                    assert (T % self.seq_len) == 0
                    S = T // self.seq_len
                    x = x.view(B, S, self.seq_len, D).reshape(B * S, self.seq_len, D)
                    return x.to(self.device)

                raise RuntimeError(f"Unexpected tensor rank {x.dim()}")

            self.batch = {
                'obs':        _reshape_to_seq(obses, is_states=False),
                'states':     _reshape_to_seq(states, is_states=True),
                'actions':    _reshape_to_seq(actions),
                'returns':    _reshape_to_seq(returns),
                'old_values': _reshape_to_seq(old_values),
                'advantages': _reshape_to_seq(advantages),
                'neglogpacs': _reshape_to_seq(neglogpacs) if neglogpacs is not None else None,
                'mu':         _reshape_to_seq(mus),
                'sigma':      _reshape_to_seq(sigmas),
                'rnn_states': rnn_states,
                'rnn_masks':  rnn_masks,
            }


        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return self.batch

        def update_values_dict(self, d):
            """외부에서 계산된 returns/advantages/old_values/old_neg 등을 배치에 반영."""
            if not d:
                return

            # returns
            x = d.get('returns', None)
            if x is not None:
                self.batch['returns'] = x.to(self.device)

            # advantages
            x = d.get('advantages', None)
            if x is not None:
                self.batch['advantages'] = x.to(self.device)

            # old_values (없으면 values를 대체로 사용)
            x = d.get('old_values', d.get('values', None))
            if x is not None:
                self.batch['old_values'] = x.to(self.device)

            # old neglogp (neglogpacs 우선, 없으면 old_neglogp)
            x = d.get('neglogpacs', d.get('old_neglogp', None))
            if x is not None:
                # 이름을 맞춰 저장(아래 학습 코드가 'neglogpacs' / 'old_neglogp' 둘 다 처리)
                self.batch['neglogpacs'] = x.to(self.device)

    def __init__(self, base_name, config):
        if 'name' not in config:
            config['name'] = 'ContinuousA2CBase'
        #A2CBase.__init__(self, base_name, config)
        #self.model = config['network']
        # 1) 먼저 부모 A2CBase 초기화
        super().__init__(base_name, config)
        self.epoch_num = 0

        # 2) 논문 구조의 Actor–Critic 모델(HeightCNN→GRU→Decoder)을 생성·래핑
        from .network_builder_dyros import A2CDYROSBuilder
        from .models_dyros import ModelA2CContinuousLogStdDYROS
        builder = A2CDYROSBuilder()
        net = builder.load(self.config)                       # DyrosActorCritic 생성
        wrapper = ModelA2CContinuousLogStdDYROS(net)         # RL-Games 래퍼
        self.model = wrapper.build(self.config)               # 최종 nn.Module
        self.model.to(self.ppo_device)
        # ① inner DyrosActorCritic 뿐 아니라, 이 Network 래퍼 전체를 GPU 로 이동
        self.model.to(self.ppo_device)
        # ② GRU 파라미터 캐시 초기화
        if hasattr(self.model, 'gru'):
            self.model.gru.flatten_parameters()
        if hasattr(self.model, 'gru'):
            self.model.gru.flatten_parameters()
        # 3) RNN 사용 여부 플래그
        self.is_rnn = getattr(self.model, 'is_rnn', False)
        self.is_discrete = False
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]
        self.bounds_loss_coef = config.get('bounds_loss_coef', None)
        #self.is_rnn = config.get('model', {}).get('rnn_hidden', 0) > 0
        self.clip_actions = config.get('clip_actions', True)

        # todo introduce device instead of cuda()
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)
        # GENE: for wandb
        self.init_wandb = False

        # --- Optimizer 생성 (공용 1개) ---
        lr = self.config['learning_rate']
        wd = getattr(self, 'weight_decay', 0.0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd, eps=1e-5)

        # --- separate_opt 호환: actor/critic 옵티마이저가 필요하면 같은 객체를 참조 ---
        if getattr(self, 'sep_op', False):
            self.optimizer_actor  = self.optimizer
            self.optimizer_critic = self.optimizer
        
        self.last_lr       = float(getattr(self, 'last_lr', lr))          # 전체/critic용으로 쓰던 값
        self.last_lr_mu    = float(getattr(self, 'last_lr_mu', lr))       # actor(μ) lr
        self.last_lr_sigma = float(getattr(self, 'last_lr_sigma', lr))    # actor(σ) lr (쓰지 않더라도 존재시켜 둠)
        self.lr_mul        = float(getattr(self, 'lr_mul', 1.0))          # lr multiplier 로깅용
        self.curr_clip_frac = float(getattr(self, 'curr_clip_frac', 0.0)) # PPO clip frac 로깅용

   
    def preprocess_actions(self, actions):
        if self.clip_actions:
            clamped_actions = torch.clamp(actions, -1.0, 1.0)
            rescaled_actions = rescale_actions(self.actions_low, self.actions_high, clamped_actions)
        else:
            rescaled_actions = actions

        if not self.is_tensor_obses:
            rescaled_actions = rescaled_actions.cpu().numpy()

        return rescaled_actions

    def init_tensors(self):
        A2CBase.init_tensors(self)
        super().init_tensors()
        self.update_list = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']
        self.tensor_list = self.update_list + ['obses', 'states', 'dones']
        if self.is_rnn:
            self.rnn_states = self.model.get_default_rnn_state()
            self.rnn_states = [s.to(self.ppo_device) for s in self.rnn_states]

    def train_epoch(self):
        super().train_epoch()
        self.epoch_num += 1

        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []
        clip_fracs = []

        if self.is_rnn:
            frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())
            print(frames_mask_ratio)

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                if self.ppo:
                    a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss, clip_frac = self.train_actor_critic(self.dataset[i])
                    lr_mul = 1.0 # PPO에서는 lr_mul이 항상 1.0
                    clip_fracs.append(clip_frac)
                else:
                    a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)
                #little update
                if hasattr(self.dataset, "update_mu_sigma"):
                    self.dataset.update_mu_sigma(cmu, csigma) 

                if self.schedule_type == 'legacy':  
                    if self.multi_gpu:
                        kl = self.hvd.average_value(kl, 'ep_kls')
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,kl.item())
                    self.update_lr(self.last_lr)

            av_kls = torch_ext.mean_list(ep_kls)

            if self.schedule_type == 'standard':
                if self.multi_gpu:
                    av_kls = self.hvd.average_value(av_kls, 'ep_kls')
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,av_kls.item())
                self.update_lr(self.last_lr)
            kls.append(av_kls)

        if self.schedule_type == 'standard_epoch':
            if self.multi_gpu:
                av_kls = self.hvd.average_value(torch_ext.mean_list(kls), 'ep_kls')
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,av_kls.item())
            self.update_lr(self.last_lr)

        if self.has_phasic_policy_gradients:
            self.ppg_aux_loss.train_net(self)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start
        if self.ppo:
            return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul, clip_fracs
        else:
            return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul

    def prepare_dataset(self, batch_dict):
        """
        play_steps_rnn()가 반환한 batch_dict로 학습용 데이터셋(self.dataset)을 생성/갱신.
        - privileged 'states'는 batch_dict → vec_env → last_priv_states 순서로 확보
        - values/returns 정규화(옵션), advantages 계산
        - Dataset 없으면 생성, 있으면 update_values_dict로 갱신
        """
        # ---------- 1) 텐서 꺼내기 ----------
        obses       = batch_dict['obses']          # [N,obs] 또는 [N,1,obs]
        returns     = batch_dict['returns']        # [N,1]  (play_steps_rnn에서 보장)
        values      = batch_dict['values']         # [N,1]
        actions     = batch_dict['actions']        # [N,act]
        mus         = batch_dict['mus']            # [N,act]
        sigmas      = batch_dict['sigmas']         # [N,act]
        neglogpacs  = batch_dict.get('neglogpacs', None)
        rnn_states  = batch_dict.get('rnn_states', None)
        rnn_masks   = batch_dict.get('rnn_masks', None)

        # ---------- 2) privileged states 확보(반드시 존재) ----------
        def _find_priv_from_anywhere_for_dataset():
            # 0) batch_dict에 이미 담겨 있으면 그대로 사용
            s = batch_dict.get('states', None)
            if s is not None:
                return s

            # 1) vec_env 트리 전체를 너비우선으로 훑는다
            roots = []
            ve = getattr(self, 'vec_env', None)
            roots.append(ve)
            i = 0
            while i < len(roots):
                r = roots[i]; i += 1
                if r is None: 
                    continue
                # 가장 확실한 경로: task.states_buf
                if hasattr(r, 'task') and hasattr(r.task, 'states_buf'):
                    return r.task.states_buf
                if hasattr(r, 'states_buf'):
                    return r.states_buf
                # get_privileged_obs가 있더라도 내부에서 AttributeError를 낼 수 있으므로 try로 감싼다
                if hasattr(r, 'get_privileged_obs'):
                    try:
                        p = r.get_privileged_obs()
                        if p is not None:
                            return p
                    except Exception:
                        pass
                # 더 깊은 래퍼 후보들을 큐에 추가
                for k in ('env', 'vec_env', 'venv', 'unwrapped'):
                    roots.append(getattr(r, k, None))

            # 2) 수집 루프에서 캐시해둔 값이 있으면 사용
            return getattr(self, 'last_priv_states', None)

        states = _find_priv_from_anywhere_for_dataset()
        if states is None:
            raise KeyError("prepare_dataset(): privileged 'states'를 가져올 수 없습니다.")

        # ---------- 3) advantages / 값 정규화 ----------
        advantages = returns - values
        if self.normalize_value:
            values_n  = self.value_mean_std(values)
            returns_n = self.value_mean_std(returns)
        else:
            values_n, returns_n = values, returns

        # ---------- 4) Dataset 생성/갱신 ----------
        if not hasattr(self, 'dataset') or (self.dataset is None):
            # 폴백 원배치 데이터셋 생성 (_OneBatchDataset는 ContinuousA2CBase 내부 클래스)
            self.dataset = self._OneBatchDataset(
                device=self.ppo_device,
                horizon_len=self.horizon_length,
                seq_len=self.seq_len,
                obses=obses, states=states, actions=actions,
                returns=returns_n, old_values=values_n, advantages=advantages,
                neglogpacs=neglogpacs, mus=mus, sigmas=sigmas,
                rnn_states=rnn_states, rnn_masks=rnn_masks
            )
        else:
            # 값들만 갱신
            self.dataset.update_values_dict({
                'old_values': values_n,
                'advantages': advantages,
                'returns':    returns_n,
                'actions':    actions,
                'obs':        obses,
                'rnn_states': rnn_states,
                'rnn_masks':  rnn_masks,
                'mu':         mus,
                'sigma':      sigmas,
            })

        # ---------- 5) 중앙가치망용 dataset 갱신(옵션) ----------
        if self.has_central_value:
            cv_dict = {
                'old_values': values_n,
                'advantages': advantages,
                'returns':    returns_n,
                'actions':    actions,
                'obs':        states,     # critic에는 privileged 입력
                'rnn_masks':  rnn_masks,
            }
            self.central_value_net.update_dataset(cv_dict)

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            self.hvd.setup_algo(self)

        while True:
            epoch_num = self.update_epoch()
            #self.model.update_action_noise((self.max_epochs-epoch_num) / self.max_epochs)
            if self.ppo:
                step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul, clip_fracs = self.train_epoch()
            else:
                step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time
            frame = self.frame

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            if self.multi_gpu:
                self.hvd.sync_stats(self)
            should_exit = False
            if self.rank == 0:
                # do we need scaled_time?
                scaled_time = sum_time #self.num_agents * sum_time
                scaled_play_time = play_time #self.num_agents * play_time
                curr_frames = self.curr_frames
                self.frame += curr_frames
                if self.print_stats:
                    fps_step = curr_frames / step_time
                    fps_step_inference = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    print(f'fps step: {fps_step:.1f} fps step and policy inference: {fps_step_inference:.1f}  fps total: {fps_total:.1f}')

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames)
                if len(b_losses) > 0:
                    self.writer.add_scalar('losses/denoise_loss', torch_ext.mean_list(b_losses).item(), frame)

                if self.has_soft_aug:
                    self.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)

                
                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    #get specific mean rewards
                    mean_specific_rewards = torch.mean(self.current_specific_rewards, axis=0)

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)
                    for nm, val in zip(self.reward_names, mean_specific_rewards):
                        self.writer.add_scalar(f'specific_rewards/{nm}/step', val, frame)
                        self.writer.add_scalar(f'specific_rewards/{nm}/iter', val, epoch_num)
                        self.writer.add_scalar(f'specific_rewards/{nm}/time', val, total_time)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards[0] <= self.last_mean_rewards):
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))
                    
                    #GENE : Adding line for logging wandb
                    if self.wandb_activate and self.ppo:
                        self.log_wandb(
                        self.frame,
                        epoch_num,
                        mean_rewards,
                        mean_lengths,
                        a_losses, 
                        b_losses,
                        c_losses,
                        entropies,
                        kls,
                        clip_fracs,
                        mean_specific_rewards,
                        self.reward_names
                        )

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))
                        if self.last_mean_rewards > self.config['score_to_win']:
                            print('Network won!')
                            self.save(os.path.join(self.nn_dir, checkpoint_name))
                            should_exit = True
                            

                if epoch_num > self.max_epochs:
                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + 'ep' + str(epoch_num) + 'rew' + str(mean_rewards)))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                update_time = 0
            if self.multi_gpu:
                    should_exit_t = torch.tensor(should_exit).float()
                    self.hvd.broadcast_value(should_exit_t, 'should_exit')
                    should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

    def log_wandb(
        self,
        frame,
        epoch_num,
        mean_rewards,
        mean_lengths,
        a_losses,
        b_losses,
        c_losses,
        entropies,
        kls,
        clip_fracs,
        specific_mean_rewards,
        reward_names
    ):
        # ── WandB 초기화 (최초 1회) ──
        if self.init_wandb is False:
            os.environ['WANDB_API_KEY'] = ''
            wandb.init(project=self.config['name'], tensorboard=False)

            # 주요 파일 저장
            if self.config['name'] == 'DyrosTocabiWalk':
                wandb.save(os.path.join(os.getcwd(), 'cfg/task/DyrosTocabiWalk.yaml'), policy="now")
                wandb.save(os.path.join(os.getcwd(), 'cfg/train/DyrosTocabiWalkPPO.yaml'), policy="now")
                wandb.save(os.path.join(os.getcwd(), 'tasks/dyros_tocabi_walk.py'), policy="now")
            if self.config['name'] == 'DyrosTocabiSquat':
                wandb.save(os.path.join(os.getcwd(), 'cfg/task/DyrosTocabiSquat.yaml'), policy="now")
                wandb.save(os.path.join(os.getcwd(), 'cfg/train/DyrosTocabiSquatPPO.yaml'), policy="now")
                wandb.save(os.path.join(os.getcwd(), 'tasks/dyros_tocabi_squat.py'), policy="now")
            if self.config['name'] == 'TocabiNewWalk':
                wandb.save(os.path.join(os.getcwd(), 'cfg/task/TocabiNewWalk.yaml'), policy="now")
                wandb.save(os.path.join(os.getcwd(), 'cfg/train/TocabiNewWalkPPO.yaml'), policy="now")
                wandb.save(os.path.join(os.getcwd(), 'tasks/tocabi_new_walk.py'), policy="now")
            if self.config['name'] == 'DyrosDynamicWalk':
                wandb.save(os.path.join(os.getcwd(), 'cfg/task/DyrosDynamicWalk.yaml'), policy="now")
                wandb.save(os.path.join(os.getcwd(), 'cfg/train/DyrosDynamicWalkPPO.yaml'), policy="now")
                wandb.save(os.path.join(os.getcwd(), 'tasks/dyros_dynamic_walk.py'), policy="now")

            wandb.save(os.path.join(os.getcwd(), '../assets/mjcf/dyros_tocabi/xml/dyros_tocabi.xml'), policy="now")
            wandb.save(os.path.join(os.getcwd(), 'tasks/base/vec_task.py'), policy="now")

            self.init_wandb = True
            self.init_wandb_time = time.time()

        # ── 주기적 로깅 ──
        if epoch_num % (self.log_wandb_frequency * self.mini_epochs_num) == 0:
            wandb_time = time.time()
            wandb_dict = {}

            # 기본 메트릭
            wandb_dict["serial_timesteps"] = epoch_num * self.horizon_length
            wandb_dict["n_updates"]       = epoch_num
            wandb_dict["total_timesteps"]  = frame
            wandb_dict["fps"]              = int(frame / (wandb_time - self.init_wandb_time))
            wandb_dict["ep_reward_mean"]   = mean_rewards.item()
            wandb_dict["ep_len_mean"]      = mean_lengths.item()

            # rollout 보상 컴포넌트
            for name, val in zip(reward_names, specific_mean_rewards):
                wandb_dict[f"rollout/{name}"] = val.item()

            # 학습 손실
            wandb_dict["time_elapsed"]               = int(wandb_time - self.init_wandb_time)
            wandb_dict["train/entropy_loss"]         = torch_ext.mean_list(entropies).item()
            wandb_dict["train/policy_gradient_loss"] = torch_ext.mean_list(a_losses).item()
            wandb_dict["train/value_loss"]           = torch_ext.mean_list(c_losses).item()
            wandb_dict["train/kl"]                   = torch_ext.mean_list(kls).item()
            wandb_dict["train/clip_fracs"]           = torch_ext.mean_list(clip_fracs).item()

            # 전체 loss 재계산
            losses = []
            for i in range(len(a_losses)):
                losses.append(
                    a_losses[i]
                    + 0.5 * c_losses[i] * self.critic_coef
                    - entropies[i] * self.entropy_coef
                    + b_losses[i]*self.config['reg_loss_coef']
                )
            wandb_dict["train/loss"] = torch_ext.mean_list(losses).item()

            # Actor–Critic 네트워크 외보상(denoise 등)도 포함한 컴포넌트별 보상
            for name, val in zip(reward_names, specific_mean_rewards):
                wandb_dict[f"reward/{name}"] = val.item()

            # 글로벌 스텝 정보
            wandb_dict["global_step"] = frame

            # 최종 로깅
            wandb.log(wandb_dict)