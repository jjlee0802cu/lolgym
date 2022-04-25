# MIT License
# 
# Copyright (c) 2020 MiscellaneousStuff
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Full game environment implementing PPO for a 1v1 game"""

import uuid
import random
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Activation, Lambda, LSTM

tf.compat.v1.disable_eager_execution()

import gym
import lolgym.envs
from pylol.lib import actions, features, point
from pylol.lib import point
from absl import flags
FLAGS = flags.FLAGS

_NO_OP = [actions.FUNCTIONS.no_op.id]
_MOVE = [actions.FUNCTIONS.move.id]
_SPELL = [actions.FUNCTIONS.spell.id]

import gym
from gym.spaces import Box, Tuple, Discrete, Dict, MultiDiscrete
import matplotlib.pyplot as plt
from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_string("config_path", "/mnt/c/Users/win8t/Desktop/pylol/config.txt", "Path to file containing GameServer and LoL Client directories")
flags.DEFINE_string("host", "192.168.0.16", "Host IP for GameServer, LoL Client and Redis")
flags.DEFINE_integer("epochs", 50, "Number of episodes to run the experiment for")
flags.DEFINE_float("step_multiplier", 1.0, "Run game server x times faster than real-time")
flags.DEFINE_bool("run_client", False, "Controls whether the game client is run or not")

# Number of possible actions that PPO Agent can take
act_space_size = 8

# Number of steps to do NO_OPs in because of lag due to fitting
fit_lag_offset = 200

class Controller(object):
    def __init__(self,
                 units=1,
                 gamma=0.99,
                 observation_space=None,
                 action_space=None,
                 name='agent'):

        self.units = units
        self.gamma = gamma

        self.observation_space = observation_space
        self.action_space = action_space

        self.init_policy_function(units=units)
        self.init_value_function(units=units)

        self.X = []
        self.Y = []
        self.V = []
        self.P = []

        self.n_agents = 0
        self.d_agents = 0
        self.cur_updating = True

        self.name = name

    def init_value_function(self, units):
        observation_space = self.observation_space

        # value function
        x = in1 = Input(observation_space.shape)
        x = Dense(units+31, activation='elu')(x)
        x = Dense(units+31, activation='elu')(x)
        x = Dense(1)(x)
        v = Model(in1, x)

        v.compile(Adam(1e-3), 'mse')
        v.summary()

        vf = K.function(v.layers[0].input, v.layers[-1].output)

        self.vf = vf
        self.v = v
    
    def init_policy_function(self, units):
        observation_space = self.observation_space
        action_space = self.action_space

        # policy function
        x = in_state = Input(observation_space.shape)
        x = Dense(units+31, activation='elu')(x)
        x = Dense(units+31, activation='elu')(x)
        x = Dense(units, activation='elu')(x)
        x = Dense(action_space.n)(x)
        action_dist = Lambda(lambda x: tf.nn.log_softmax(x, axis=-1))(x)
        p = Model(in_state, action_dist)
        
        in_advantage = Input((1,))
        in_old_prediction = Input((action_space.n,))

        def loss(y_true, y_pred):
            advantage = tf.reshape(in_advantage, (-1,))
        
            # y_pred is the log probs of the actions
            # y_true is the action mask
            prob = tf.reduce_sum(y_true * y_pred, axis=-1)
            old_prob = tf.reduce_sum(y_true * in_old_prediction, axis=-1)
            ratio = tf.exp(prob - old_prob)
            
            # this is the VPG objective
            #ll = -(prob * advantage)
            
            # this is PPO objective
            ll = -K.minimum(ratio*advantage, K.clip(ratio, 0.8, 1.2)*advantage)
            return ll

        popt = Model([in_state, in_advantage, in_old_prediction], action_dist)
        popt.compile(Adam(5e-4), loss)
        popt.summary()

        pf = K.function(p.layers[0].input,
                        [p.layers[-1].output,
                        tf.random.categorical(p.layers[-1].output, 1)[0]])
                        
        self.pf = pf
        self.popt = popt
        self.p = p

    def fit(self, batch_size=5, epochs=10, shuffle=True, verbose=0):
        X = self.X
        Y = self.Y
        V = self.V
        P = self.P

        self.d_agents += 1 

        if self.d_agents < self.n_agents:
            return None, None

        print("[FIT] TRAINING ON DATA FOR", self.name)
        X, Y, V, P = [np.array(x) for x in [X, Y, V, P]]

        # Subtract value baseline to get advantage
        A = V - self.vf(X)[:, 0]

        loss = self.popt.fit([X, A, P], Y, batch_size=5, epochs=10, shuffle=True, verbose=0)
        loss = loss.history["loss"][-1]
        vloss = self.v.fit(X, V, batch_size=5, epochs=10, shuffle=True, verbose=0)
        vloss = vloss.history["loss"][-1]

        self.X = []
        self.Y = []
        self.V = []
        self.P = []
        
        self.d_agents = 0

        return loss, vloss

    def get_pred_act(self, obs):
        pred, act = [x[0] for x in self.pf(obs[None])]
        return pred, act

    def register_agent(self):
        self.n_agents += 1
        return self.n_agents

class PPOAgent(object):
    """Basic PPO implementation for LoLGym environment."""
    def __init__(self, controller=None, run_client=False):
        if not controller:
            raise ValueError("PPOAgent needs to be provided an external controller")
        
        self.controller = controller
        self.agent_id = controller.register_agent()

        print("PPOAgent:", self.agent_id, "Controller:", self.controller)

        env = gym.make("LoLGame-v0")
        env.settings["map_name"] = "Old Summoners Rift" # Map to play on. Howling Abyss doesn't spawn minions
        env.settings["human_observer"] = run_client # Set to True to run league client
        env.settings["host"] = FLAGS.host # Set this using "hostname -i" ip on Linux
        env.settings["players"] = "Ezreal.BLUE,Ezreal.PURPLE" # The champions each player will be
        env.settings["config_path"] = FLAGS.config_path
        env.settings["step_multiplier"] = FLAGS.step_multiplier
        self.env = env

        # initialize vars used for computing the reward at each step
        self.old_me_kills = 0
        self.old_enemy_kills = 0
        self.old_me_hp_ratio = 1
        self.old_enemy_hp_ratio = 1

    def save_pair(self, obs, act):
        action_space = self.controller.action_space
        self.controller.X.append(np.copy(obs))
        act_mask = np.zeros((action_space.n))
        act_mask[act] = 1.0
        self.controller.Y.append(act_mask)

    def close(self):
        self.env.close()

def convert_action_singular(raw_obs, act, which_unit):
    """Converts a given action index for the specified unit into an action used by the env"""

    me_unit_str = 'me_unit' if which_unit == 'me_unit' else 'enemy_unit'
    enemy_unit_str = 'enemy_unit' if which_unit == 'me_unit' else 'me_unit'
    
    me_pos = point.Point(raw_obs[0].observation[me_unit_str].position_x,
                                raw_obs[0].observation[me_unit_str].position_y)
    enemy_pos = point.Point(raw_obs[0].observation[enemy_unit_str].position_x,
                                raw_obs[0].observation[enemy_unit_str].position_y)
    
    if act == 0: # Movement (direction 1)
        act = _MOVE + [point.Point(8,4)]
    elif act == 1: # Movement (direction 2)
        act = _MOVE + [point.Point(0,4)]
    elif act == 2: # Movement (direction 3)
        act = _MOVE + [point.Point(4,8)]
    elif act == 3: # Movement (direction 4)
        act = _MOVE + [point.Point(4,0)]
    elif act == 4: # Spell (Q)
        act = _SPELL + [[0], enemy_pos]
    elif act == 5: # Spell (W)
        act = _SPELL + [[1], enemy_pos]
    elif act == 6: # Spell (R)
        act = _SPELL + [[3], enemy_pos]
    elif act == 7: # no action
        act = _NO_OP
    elif act == 8: # E (direction 1)
        act = _SPELL + [[2], point.Point(me_pos.x + 0, me_pos.y + 400)]
    elif act == 9: # E (direction 2)
        act = _SPELL + [[2], point.Point(me_pos.x + 400, me_pos.y + 800)]
    elif act == 10: # E (direction 3)
        act = _SPELL + [[2], point.Point(me_pos.x + 400, me_pos.y + 0)]
    else: # E (direction 4)
        act = _SPELL + [[2], point.Point(me_pos.x + 800, me_pos.y + 400)]

    return act

def convert_action(raw_obs, act, enemy_act):
    """Converts a given action index within the action space for each unit into a list of actions for each unit used by the env"""
    return [convert_action_singular(raw_obs, act, "me_unit"), convert_action_singular(raw_obs, enemy_act, "enemy_unit")]

def create_obs_vectors_singular(raw_obs, which_unit):
        """Creates a singular observation vector from the persepctive of which_unit
        
        See the link below for a list of available features and corresponding indices
        https://github.com/jjlee0802cu/pylol/blob/main/pylol/lib/features.py
        """
        me_unit_str = 'me_unit' if which_unit == 'me_unit' else 'enemy_unit'
        enemy_unit_str = 'enemy_unit' if which_unit == 'me_unit' else 'me_unit'

        arr = []

        me_unit = raw_obs[0].observation[me_unit_str]
        indices_to_omit = set([0,3,7,8,9,13,16,18,19,21,22,23,24,25])
        for i in range(len(me_unit)):
            if i not in indices_to_omit:
                arr.append(me_unit[i])

        enemy_unit = raw_obs[0].observation[enemy_unit_str]
        indices_to_omit = set([0,1,2,3,7,8,9,13,16,18,19,21,22])
        for i in range(len(enemy_unit)):
            if i not in indices_to_omit:
                arr.append(enemy_unit[i])

        arr = np.array(arr)[None].reshape(-1) / 100

        return arr

def create_obs_vectors(raw_obs):
    """Creates a two observation vectors: one from me_unit's perspective and one from enemy_unit's perspective"""
    return create_obs_vectors_singular(raw_obs, 'me_unit'), create_obs_vectors_singular(raw_obs, 'enemy_unit')

def compute_rewards(agent1, agent2, raw_obs):
    """Compute reward using custom reward function"""
    # hp delta difference
    agent1_hp_ratio = raw_obs[0].observation["me_unit"].current_hp / raw_obs[0].observation["me_unit"].max_hp
    agent2_hp_ratio = raw_obs[0].observation["enemy_unit"].current_hp / raw_obs[0].observation["enemy_unit"].max_hp
    
    agent1_delta_ratio = agent1_hp_ratio - agent1.old_me_hp_ratio
    agent2_delta_ratio = agent2_hp_ratio - agent2.old_me_hp_ratio

    agent1_rew = agent1_delta_ratio - agent2_delta_ratio
    agent2_rew = agent2_delta_ratio - agent1_delta_ratio

    agent1.old_me_hp_ratio = agent1_hp_ratio
    agent1.old_enemyh_hp_ratio = agent2_hp_ratio
    agent2.old_me_hp_ratio = agent2_hp_ratio
    agent2.old_enemyh_hp_ratio = agent1_hp_ratio

    # kills (+20 for a kill, -20 for a death)
    agent1_kills = raw_obs[0].observation["me_unit"].kill_count
    agent2_kills = raw_obs[0].observation["enemy_unit"].kill_count

    if agent1_kills > agent1.old_me_kills:
        agent1_rew = 20
    if agent2_kills > agent1.old_enemy_kills:
        agent1_rew = -20

    if agent2_kills > agent2.old_me_kills:
        agent2_rew = 20
    if agent1_kills > agent2.old_enemy_kills:
        agent2_rew = -20

    agent1.old_me_kills = agent1_kills
    agent1.old_enemy_kills = agent2_kills
    agent2.old_me_kills = agent2_kills
    agent2.old_enemy_kills = agent1_kills

    return agent1_rew, agent2_rew

def train_both(agent1, agent2, epochs, batch_steps, episode_steps):
    """Trains two PPO agents with specified number of epochs, batch_stes, and episode_steps"""
    final_out = ""
    lll1 = []
    lll2 = []

    # use agent1's controller as the main environment controller
    controller = agent1.controller
    env = agent1.env 

    for epoch in range(epochs):
        print(epoch)
        st = time.perf_counter()
        ll1 = []
        ll2 = []

        while len(controller.X) < batch_steps:
            # reset the environment and location of players
            obs = env.reset()
            env.teleport(1, point.Point(7100.0, 7000.0))
            env.teleport(2, point.Point(7500.0, 7000.0))

            # Get raw observation and create new observation vector
            raw_obs = obs
            agent1_obs, agent2_obs = create_obs_vectors(raw_obs)

            rews1 = []
            rews2 = []
            steps = 0
            while True:
                print()
                steps += 1

                # Prediction, action, save prediction
                agent1_pred, agent1_act = [x[0] for x in agent1.controller.pf(agent1_obs[None])]
                agent2_pred, agent2_act = [x[0] for x in agent2.controller.pf(agent2_obs[None])]
                agent1.controller.P.append(agent1_pred)
                agent2.controller.P.append(agent2_pred)

                # Add no-ops  for the first few steps in order to wait for training lag
                if steps < fit_lag_offset:
                    agent1_act = 7
                    agent2_act = 7
                else: # Add a decaying randomness to the chosen action
                    probability = 1 - epoch/epochs
                    probability = 0 if probability < 0 else probability
                    if np.random.random_sample() < probability:
                        agent1_act = np.random.choice(act_space_size)
                    if np.random.random_sample() < probability:
                        agent2_act = np.random.choice(act_space_size)

                # Save this state action pair
                agent1.save_pair(agent1_obs, agent1_act)
                agent2.save_pair(agent2_obs, agent2_act)

                # Get action
                act = convert_action(raw_obs, agent1_act, agent2_act)
                print("\t", act)

                # Take the action and save the reward
                obs, _, done, _ = env.step(act)
                raw_obs = obs
                agent1_obs, agent2_obs = create_obs_vectors(raw_obs)

                # Compute rewards for each agent and save
                agent1_rew, agent2_rew = compute_rewards(agent1, agent2, raw_obs)
                rews1.append(agent1_rew)
                rews2.append(agent2_rew)
                #print("rewards:")
                #print("\t Agent 1:", agent1_rew)
                #print("\t Agent 2:", agent2_rew)

                done = done[0]
                if steps == episode_steps or done:
                    ll1.append(np.sum(rews1))
                    ll2.append(np.sum(rews2))

                    for i in range(len(rews1)-2, -1, -1):
                        rews1[i] += rews1[i+1] * agent1.controller.gamma
                    agent1.controller.V.extend(rews1)

                    for i in range(len(rews2)-2, -1, -1):
                        rews2[i] += rews2[i+1] * agent2.controller.gamma
                    agent2.controller.V.extend(rews2)

                    break
        
        loss1, vloss1 = agent1.controller.fit()
        loss2, vloss2 = agent2.controller.fit()

        if loss1 != None and vloss1 != None:
            lll1.append((epoch, np.mean(ll1), loss1, vloss1, len(agent1.controller.X), len(ll1), time.perf_counter() - st))
            print("%3d  ep_rew:%9.2f  loss:%7.2f   vloss:%9.2f  counts: %5d/%3d tm: %.2f s" % lll1[-1])
            env.broadcast_msg("Episode No: %3d  Episode Reward: %9.2f" % (lll1[-1][0], lll1[-1][1]))
            sign = "+" if lll1[-1][1] >= 0 else ""
            final_out += sign + str(lll1[-1][1])

        if loss2 != None and vloss2 != None:
            lll2.append((epoch, np.mean(ll2), loss2, vloss2, len(agent2.controller.X), len(ll2), time.perf_counter() - st))
            print("%3d  ep_rew:%9.2f  loss:%7.2f   vloss:%9.2f  counts: %5d/%3d tm: %.2f s\n" % lll2[-1])
            env.broadcast_msg("Episode No: %3d  Episode Reward: %9.2f" % (lll2[-1][0], lll2[-1][1]))
            sign = "+" if lll2[-1][1] >= 0 else ""
            final_out += sign + str(lll2[-1][1])

def save_model(agent, i):
    print()
    print('saving popt')
    tf.keras.models.save_model(agent.controller.popt, f'./lolgym/examples/saved_models/adversarial_popt{i}')
    print('saving p')
    tf.keras.models.save_model(agent.controller.p, f'./lolgym/examples/saved_models/adversarial_p{i}')
    print('saving v')
    tf.keras.models.save_model(agent.controller.v, f'./lolgym/examples/saved_models/adversarial_v{i}')

def main(unused_argv):
    units = 1
    gamma = 0.99
    epochs = FLAGS.epochs
    batch_steps = 200 + fit_lag_offset
    episode_steps = batch_steps
    run_client = FLAGS.run_client

    # Declare observation space, action space and model controller
    observation_space = Box(low=0, high=50000, shape=(45,), dtype=np.float32)
    action_space = Discrete(act_space_size)
    controller1 = Controller(units, gamma, observation_space, action_space, 'CONTROLLER 1')
    controller2 = Controller(units, gamma, observation_space, action_space, 'CONTROLLER 2')

    # Declare, train and run agent
    agent1 = PPOAgent(controller=controller1, run_client=run_client)
    agent2 = PPOAgent(controller=controller2, run_client=run_client)
    train_both(agent1, agent2, epochs=epochs, batch_steps=batch_steps, episode_steps=episode_steps)
    save_model(agent1, "1")
    save_model(agent2, "2")
    agent1.close()
    agent2.close()

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)
