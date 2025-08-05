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
            prev_actions = input_dict.get('prev_actions', None)
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma)

            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                return {
                    'prev_neglogp': torch.squeeze(prev_neglogp),
                    'values': value,
                    'entropy': entropy,
                    'rnn_states': states,
                    'mus': mu,
                    'sigmas': sigma
                }
            else:
                action = distr.sample()
                neglogp = self.neglogp(action, mu, sigma, logstd)
                return {
                    'neglogpacs': torch.squeeze(neglogp),
                    'values': value,
                    'actions': action,
                    'rnn_states': states,
                    'mus': mu,
                    'sigmas': sigma
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

'''
    class Network(nn.Module):
        def __init__(self, a2c_network):
            nn.Module.__init__(self)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()
            
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }                
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : value,
                    'actions' : selected_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result

        def neglogp(self, x, mean, std, logstd):
            return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                + logstd.sum(dim=-1)

        def update_action_noise(self, progress_remaining):
            if (progress_remaining > 0.5):
                progress_remaining_biased = 2*progress_remaining - 1
            else:
                progress_remaining_biased = 0.0

            self.a2c_network.sigma[:] = self.a2c_network.sigma_init * progress_remaining_biased + self.a2c_network.sigma_last * (1-progress_remaining_biased)
'''