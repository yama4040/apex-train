import random

import numpy as np
import tensorflow as tf
import keras.layers as kl
from actions import Actions


class QNetwork(tf.keras.Model):
    def __init__(self, num_states):
        super(QNetwork, self).__init__()
        self.num_states = num_states
        self.input_layer = kl.Dense(512, activation="relu", kernel_initializer="he_normal")
        self.hidden_layer_1 = kl.Dense(256, activation="relu", kernel_initializer="he_normal")
        self.hidden_layer_2 = kl.Dense(128, activation="relu", kernel_initializer="he_normal")
        self.hidden_layer_3 = kl.Dense(64, activation="relu", kernel_initializer="he_normal")
        self.hidden_layer_4 = kl.Dense(32, activation="relu", kernel_initializer="he_normal")
        self.output_layer = kl.Dense(len(Actions), kernel_initializer="he_normal")
        self.action_list=np.array([i for i in range(len(Actions))])

    def call(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.hidden_layer_3(x)
        x = self.hidden_layer_4(x)
        x = self.output_layer(x)
        return x

    def sample_action(self, state, epsilon,forbidden_action):
        state = np.atleast_2d(state)
        qvalues = self(state)
        qvalues=np.array(qvalues)[0]
        if random.random() > epsilon:
            qvalues[forbidden_action]=-1.0 * (10**12)
            action = np.argmax(qvalues)
        else:
            action = np.random.choice(self.action_list[~forbidden_action])

        return action,qvalues
