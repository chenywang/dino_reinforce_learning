import json
import os
import random
import time
from collections import deque

import numpy as np
from IPython.core.display import clear_output
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from game import loss_file_path, loss_df, q_values_df, scores_df, actions_df, q_value_file_path
from util import load_obj, save_obj

# game parameters
ACTIONS = 2  # possible actions: jump, do nothing
GAMMA = 0.99  # decay rate of past observations original 0.99
OBSERVATION = 100.  # timesteps to observe before training
EXPLORE = 100000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 16  # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
img_rows, img_cols = 80, 80
img_channels = 4  # We stack 4 frames


class DinoAgent:
    def __init__(self, game):  # takes game as input for taking actions
        self._game = game
        self.jump()  # to start the game, we need to jump once

    def is_running(self):
        return self._game.get_playing()

    def is_crashed(self):
        return self._game.get_crashed()

    def jump(self):
        self._game.press_up()

    def duck(self):
        self._game.press_down()


def build_model():
    print("Now we build the model")
    model = Sequential()
    model.add(
        Conv2D(32, (8, 8), padding='same', strides=(4, 4), input_shape=(img_cols, img_rows, img_channels)))  # 80*80*4
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)

    # create model file if not present
    if not os.path.isfile(loss_file_path):
        model.save_weights('model.h5')
    print("We finish building the model")
    return model


''' 
main training module
Parameters:
* model => Keras Model to be trained
* game_state => Game State module with access to game environment and dino
* observe => flag to indicate wherther the model is to be trained(weight updates), else just play
'''


def train_network(model, game_state, observe=False):
    last_time = time.time()
    # store the previous observations in replay memory
    D = load_obj("D")  # load from file system
    # get the first state by doing nothing
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1  # 0 => do nothing,
    # 1=> jump

    x_t, r_0, terminal = game_state.get_state(do_nothing)  # get next step after performing the action

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)  # stack 4 images to create placeholder input

    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*20*40*4

    initial_state = s_t

    if observe:
        OBSERVE = 999999999  # We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        print ("Weight load successfully")
    else:  # We go to training mode
        OBSERVE = OBSERVATION
        epsilon = load_obj("epsilon")
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)

    t = load_obj("time")  # resume from the previous time step stored in file system
    while True:  # endless running

        loss = 0
        Q_sa = 0
        action_index = 0
        a_t = np.zeros([ACTIONS])  # action at t

        # choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:  # parameter to skip frames for actions
            if random.random() <= epsilon:  # randomly explore an action
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[0] = 1
            else:  # predict the output
                q = model.predict(s_t)  # input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)  # choose index with maximum q value
                action_index = max_Q
                a_t[action_index] = 1  # 0=> do nothing, 1=> jump

        # We reduced the epsilon (exploration parameter) gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            # run the selected action and observed next state and reward
        x_t1, r_t, terminal = game_state.get_state(a_t)
        print('fps: {0}'.format(1 / (time.time() - last_time)))  # helpful for measuring frame rate
        last_time = time.time()
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x20x40x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)  # append the new image to input stack and remove the first one

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:

            # sample a mini_batch to train on
            mini_batch = random.sample(D, BATCH)
            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # 32, 20, 40, 4
            targets = np.zeros((inputs.shape[0], ACTIONS))  # 32, 2

            # Now we do the experience replay
            for i in range(0, len(mini_batch)):
                state_t = mini_batch[i][0]  # 4D stack of images
                action_t = mini_batch[i][1]  # This is action index
                reward_t = mini_batch[i][2]  # reward at state_t due to action_t
                state_t1 = mini_batch[i][3]  # next state
                terminal = mini_batch[i][4]  # wheather the agent died or survided due the action

                inputs[i:i + 1] = state_t

                targets[i] = model.predict(state_t)  # predicted q values
                Q_sa = model.predict(state_t1)  # predict q values for next step

                if terminal:
                    targets[i, action_t] = reward_t  # if terminated, only equals reward
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            loss += model.train_on_batch(inputs, targets)
            loss_df.loc[len(loss_df)] = loss
            q_values_df.loc[len(q_values_df)] = np.max(Q_sa)

        s_t = initial_state if terminal else s_t1  # reset game to initial frame if terminate
        t = t + 1

        # save progress every 1000 iterations
        if t % 1000 == 0:
            print("Now we save model")
            game_state._game.pause()  # pause game while saving to filesystem
            model.save_weights("model.h5", overwrite=True)
            save_obj(D, "D")  # saving episodes
            save_obj(t, "time")  # caching time steps
            save_obj(epsilon, "epsilon")  # cache epsilon to avoid repeated randomness in actions
            loss_df.to_csv("./objects/loss_df.csv", index=False)
            scores_df.to_csv("./objects/scores_df.csv", index=False)
            actions_df.to_csv("./objects/actions_df.csv", index=False)
            q_values_df.to_csv(q_value_file_path, index=False)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)
            clear_output()
            game_state._game.resume()
        # print info
        if t <= OBSERVE:
            state = "observe"
        elif t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print "TIMESTAMP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION"\
            , action_index, "/ REWARD", r_t, "/ Q_MAX ", np.max(Q_sa), "/ Loss ", loss


print "Episode finished!"
print "************************"


def init_cache():
    """initial variable caching, done only once"""
    save_obj(INITIAL_EPSILON, "epsilon")
    t = 0
    save_obj(t, "time")
    D = deque()
    save_obj(D, "D")
