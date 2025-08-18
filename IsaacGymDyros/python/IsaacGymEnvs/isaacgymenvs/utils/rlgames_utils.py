# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch import torch_ext
import torch
import numpy as np
from typing import Callable

from tasks import isaacgym_task_map


def get_rlgames_env_creator(
        # used to create the vec task
        task_config: dict,
        task_name: str,
        sim_device: str,
        rl_device: str,
        graphics_device_id: int,
        headless: bool,
        # Used to handle multi-gpu case
        multi_gpu: bool = False,
        post_create_hook: Callable = None,
):
    """Parses the configuration parameters for the environment task and creates a VecTask

    Args:
        task_config: environment configuration.
        task_name: Name of the task, used to evaluate based on the imported name (eg 'Trifinger')
        sim_device: The type of env device, eg 'cuda:0'
        rl_device: Device that RL will be done on, eg 'cuda:0'
        graphics_device_id: Graphics device ID.
        headless: Whether to run in headless mode.
        multi_gpu: Whether to use multi gpu
        post_create_hook: Hooks to be called after environment creation.
            [Needed to setup WandB only for one of the RL Games instances when doing multiple GPUs]
    Returns:
        A VecTaskPython object.
    """
    def create_rlgpu_env(_sim_device=sim_device, _rl_device=rl_device, **kwargs):
        """
        Creates the task from configurations and wraps it using RL-games wrappers if required.
        """

        if multi_gpu:
            import horovod.torch as hvd

            rank = hvd.rank()
            print("Horovod rank: ", rank)

            _sim_device = f'cuda:{rank}'
            _rl_device = f'cuda:{rank}'

            task_config['rank'] = rank
            task_config['rl_device'] = 'cuda:' + str(rank)
        else:
            _sim_device = sim_device
            _rl_device = rl_device

        # create native task and pass custom config
        env = isaacgym_task_map[task_name](
            cfg=task_config,
            sim_device=_sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless
        )

        if post_create_hook is not None:
            post_create_hook()

        return env
    return create_rlgpu_env


class RLGPUAlgoObserver(AlgoObserver):
    """Allows us to log stats from the env along with the algorithm running stats."""

    def __init__(self):
        self.algo = None
        self.writer = None
        self.ep_infos = []       # ← 핵심: 리스트로 미리 생성
        self.direct_info = {}
        self.mean_scores = None  # after_init에서 실제 객체로 바꿈

    def after_init(self, algo):
        self.algo = algo
        self.mean_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer

        if not isinstance(self.ep_infos, list):
            self.ep_infos = []
        self.direct_info = {}

    def process_infos(self, infos, done_indices):
        assert isinstance(infos, dict), "RLGPUAlgoObserver expects dict info"

        if 'episode' in infos and isinstance(infos['episode'], dict):
            self.ep_infos.append(infos['episode'])

        self.direct_info = {}
        for k, v in infos.items():
            if isinstance(v, (float, int)):
                self.direct_info[k] = float(v)
            elif isinstance(v, torch.Tensor) and v.dim() == 0:
                self.direct_info[k] = float(v.item())

    def after_clear_stats(self):
        if self.mean_scores is not None:
            self.mean_scores.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.ep_infos:
            keys = list(self.ep_infos[0].keys())
            for key in keys:
                vals = []
                for ep in self.ep_infos:
                    x = ep.get(key)
                    if isinstance(x, torch.Tensor):
                        x = x if x.dim() == 0 else x.mean()
                        vals.append(x.to(self.algo.device).float())
                    elif isinstance(x, (float, int)):
                        vals.append(torch.tensor(float(x), device=self.algo.device))
                if len(vals) > 0:
                    value = torch.stack(vals).mean()
                    self.writer.add_scalar(f'Episode/{key}', value.item(), epoch_num)
            self.ep_infos.clear()

        for k, v in self.direct_info.items():
            self.writer.add_scalar(f'{k}/frame', v, frame)
            self.writer.add_scalar(f'{k}/iter', v, epoch_num)
            self.writer.add_scalar(f'{k}/time', v, total_time)

        ms = getattr(self, 'mean_scores', None)
        if ms is not None and getattr(ms, 'current_size', 0) > 0:
            m = ms.get_mean()
            self.writer.add_scalar('scores/mean', m, frame)
            self.writer.add_scalar('scores/iter', m, epoch_num)
            self.writer.add_scalar('scores/time', m, total_time)

class MultiObserver(AlgoObserver):
    """Meta-observer that allows the user to add several observers."""

    def __init__(self, observers_):
        super().__init__()
        self.observers = observers_

    def _call_multi(self, method, *args_, **kwargs_):
        for o in self.observers:
            getattr(o, method)(*args_, **kwargs_)

    def before_init(self, base_name, config, experiment_name):
        self._call_multi('before_init', base_name, config, experiment_name)

    def after_init(self, algo):
        self._call_multi('after_init', algo)

    def process_infos(self, infos, done_indices):
        self._call_multi('process_infos', infos, done_indices)

    def after_steps(self):
        self._call_multi('after_steps')

    def after_clear_stats(self):
        self._call_multi('after_clear_stats')

    def after_print_stats(self, frame, epoch_num, total_time):
        self._call_multi('after_print_stats', frame, epoch_num, total_time)


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)

    def step(self, action):
        return  self.env.step(action)

    def reset(self):
        return self.env.reset()
    
    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        if hasattr(self.env, "amp_observation_space"):
            info['amp_observation_space'] = self.env.amp_observation_space

        if self.env.num_states > 0:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info

    def get_privileged_obs(self):
        """
        래퍼 깊이에 상관없이 task.states_buf를 최대한 찾아서 반환.
        없으면 None.
        """
        # 직접 보유
        if hasattr(self, 'task') and hasattr(self.task, 'states_buf'):
            return self.task.states_buf
        if hasattr(self, 'states_buf'):
            return self.states_buf

        # 한 단계 아래 래퍼
        e = getattr(self, 'env', None)
        if e is not None:
            if hasattr(e, 'task') and hasattr(e.task, 'states_buf'):
                return e.task.states_buf
            if hasattr(e, 'states_buf'):
                return e.states_buf
            if hasattr(e, 'get_privileged_obs'):
                try:
                    return e.get_privileged_obs()
                except Exception:
                    pass
        return None