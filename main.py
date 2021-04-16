import argparse, datetime
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from algorithms.DQN.DQN_agent import DQN

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def main(args):
    env = gym.make(args.scenario).unwrapped
    env.reset()
    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. Typical dimensions at this point are close to 3x40x90
    # which is the result of a clamped and down-scaled render buffer in get_screen()
    init_screen = get_screen(env)
    _, _, screen_height, screen_width = init_screen.shape

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    model = DQN(screen_height, screen_width, n_actions,args)

    episode_durations = []

    for i_episode in range(args.num_episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = model.select_action(state)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            model.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            model.update(i_episode)
            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % args.TARGET_UPDATE == 0:
            model.value_network_target.load_state_dict(model.value_network.state_dict())

    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()


def get_cart_location(env,screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env,screen_width)
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
    return resize(screen).unsqueeze(0).to(device)

def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
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
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="CartPole-v0", type=str)
    parser.add_argument('--num_episodes', default=50, type=int)
    parser.add_argument('--BATCH_SIZE', default=128, type=int)
    parser.add_argument('--GAMMA', default=0.999, type=float)
    parser.add_argument('--EPS_START', default=0.9, type=float)
    parser.add_argument('--EPS_END', default=0.05, type=float)
    parser.add_argument('--EPS_DECAY', default=200, type=int)
    parser.add_argument('--TARGET_UPDATE', default=10, type=int)

    args = parser.parse_args()
    main(args)