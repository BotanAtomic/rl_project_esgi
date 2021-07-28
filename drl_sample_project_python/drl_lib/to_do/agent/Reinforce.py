import numpy as np
import tensorflow as tf


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape=(9,), activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(9, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.categorical_crossentropy)
    model.summary()
    return model


class PGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.states = []
        self.rewards = []
        self.actions = []
        self.model = build_model()
        self.model.summary()

    def memorize(self, state, action, reward):
        # print("Memorize ", state)
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)

    def get_next_action(self, state, available_actions):
        if len(available_actions) == 1:
            return available_actions[0]

        model_probabilities = self.model.predict(state, batch_size=1).flatten()
        for index, probability in enumerate(model_probabilities):
            if index not in available_actions:
                model_probabilities[index] = 0

        if np.count_nonzero(model_probabilities) < 1:
            model_probabilities[np.random.choice(available_actions)] = 1.0

        prob = model_probabilities / np.sum(model_probabilities)
        return np.random.choice(self.action_size, 1, p=prob)[0]

    def update_network(self):
        reward_sum = 0
        discounted_rewards = []

        for reward in self.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + self.gamma * reward_sum
            discounted_rewards.append(reward_sum)

        discounted_rewards.reverse()
        discounted_rewards = np.array(discounted_rewards)
        # standardise the rewards
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        states = np.vstack(self.states)

        Y = []

        for i in reversed(range(len(discounted_rewards))):
            o_h = np.zeros(states.shape[1])
            o_h[self.actions[i]] = discounted_rewards[i]
            Y.insert(0, o_h)

        Y = np.vstack(Y)

        loss = self.model.train_on_batch(states, Y)

        self.states = []
        self.rewards = []
        self.actions = []
        return loss
