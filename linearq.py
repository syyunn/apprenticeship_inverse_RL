# -*- coding: utf-8 -*-
import gym
import math
import torch
import random
import argparse
import matplotlib
import debug as db
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from itertools import count
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from collections import namedtuple

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

HIDDEN_LAYER = 32  # NN hidden layer size
class DQN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, HIDDEN_LAYER)
        self.l1_1 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.l1_2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.l1_3 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l1_1(x))
        x = F.relu(self.l1_2(x))
        x = F.relu(self.l1_3(x))
        x = self.l2(x)
        return x

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQN_Trainer(object):

    env = gym.make('CartPole-v0').unwrapped
    #
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.95
    num_episodes = 500

    EPS_END = 0.05
    EPS_DECAY = 500
    TARGET_UPDATE = 10
    resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])


    def __init__(self, args):
        # Get screen size so that we can initialize layers correctly based on shape
        # returned from AI gym. Typical dimensions at this point are close to 3x40x90
        # which is the result of a clamped and down-scaled render buffer in get_screen()
        self.env.reset()

        # policy_net = DQN(screen_height, screen_width).to(device)
        # target_net = DQN(screen_height, screen_width).to(device)
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayMemory(100000)

        self.NUM_UPDATE = 1
        self.steps_done = 0
        self.episode_durations = []
        self.plot = args.plot
        if self.plot:
            plt.ion()
            plt.figure()
            self.init_screen = self.get_screen()
            plt.imshow(self.get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
                    interpolation='none')
            plt.title('Example extracted screen')
            plt.show()

    def get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen).unsqueeze(0).to(self.device)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # print(self.policy_net(state).max(1))
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        for i in range(self.NUM_UPDATE):
            transitions = self.memory.sample(self.BATCH_SIZE)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                                        if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

    def train(self):
        for i_episode in range(self.num_episodes):
            # Initialize the environment and state
            state = torch.from_numpy(self.env.reset()).unsqueeze(0).to(self.device, dtype=torch.float)
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action.item())
                if self.plot and i_episode % 100 == 0:
                    self.get_screen()
                next_state = torch.from_numpy(next_state).unsqueeze(0).to(self.device, dtype=torch.float)
                reward = torch.tensor([reward], device=self.device)

                # Observe new state
                last_screen = state
                current_screen = next_state
                if done:
                    next_state = None

                # Store the transition in self.memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done:
                    self.episode_durations.append(t + 1)
                    self.showProgress(i_episode)
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        #
        # Done training.
        print('Complete')
        self.env.render()
        self.env.close()
        plt.ioff()
        plt.show()

    def showProgress(self, e_num):
        means = 0
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if len(self.episode_durations) >= 100:
            means = durations_t[-100:-1].mean().item()
        db.printInfo('Episode %d/%d Duration: %d AVG: %d'%(e_num, self.num_episodes, durations_t[-1], means))
        if self.plot:
            plt.figure(2)
            plt.clf()
            plt.title('Training...')
            plt.xlabel('Episode')
            plt.ylabel('Duration')
            plt.plot(durations_t.numpy())
            # Take 100 episode averages and plot them too
            if len(durations_t) >= 100:
                means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(99), means))
                plt.plot(means.numpy())

            plt.pause(0.001)  # pause a bit so that plots are updated
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('General tool to train a NN based on passed configuration.')
    # parser.add_argument('--config', dest='configStr', default='DefaultConfig', type=str, help='Name of the config file to import.')
    parser.add_argument('--plot', dest='plot', default=False, action='store_true', help='Whether to plot the training progress.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = getInputArgs()
    dqnTrainer = DQN_Trainer(args)
    dqnTrainer.train()