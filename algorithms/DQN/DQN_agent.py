import torch, os
import numpy as np, random
import math

from algorithms.DQN.network import Value_net
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(object):

    def __init__(self, h, w, n_actions,args):
        self.h = h
        self.w = w
        self.config = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps_done = 0
        self.n_action = n_actions
        self.memory = ReplayMemory(10000)
        # Networks
        self.value_network = Value_net(self.h, self.w, self.n_action).to(self.device)
        self.value_network_target = Value_net( self.h, self.w, self.n_action).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.value_network.parameters())
        self.value_network_target.eval()

    def load_model(self):
        model_path = "./trained_model/" + str(self.config.model_episode) + ".pth"
        if os.path.exists(model_path) :
            print("load model!")
            value_network = torch.load(model_path)
            value_network_target = torch.load(model_path)
            self.value_network.load_state_dict(value_network)
            self.value_network_target.load_state_dict(value_network)

    def save_model(self, episode):
        if not os.path.exists("./trained_model/" ):
            os.mkdir("./trained_model/" )
        torch.save(self.value_network.state_dict(),
                   "./trained_model/" + str(episode) + ".pth"),

    def select_action(self,state):
        sample = random.random()
        eps_threshold = self.config.EPS_END + (self.config.EPS_START - self.config.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.config.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.value_network(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_action)]], device=self.device, dtype=torch.long)

    def update(self,i_episode):
        if len(self.memory) < self.config.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.config.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.value_network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.config.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.value_network_target(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.config.GAMMA) + reward_batch

        # Compute Huber loss
        loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.value_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
