import tensorflow as tf
import numpy as np


class REINFORCENet():
    def __init__(self, state_size, action_size, learning_rate=0.01):
        self.writer = tf.summary.FileWriter("/tensorboard/pg/1")
        self.state_size = state_size
        self.action_size = action_size
        self.input = tf.placeholder(
            tf.float32, shape=[None, state_size], name="input_")
        self.action = tf.placeholder(
            tf.float32, shape=[None, action_size], name="action_")
        self.discount_ep_reward = tf.placeholder(
            tf.float32, shape=[None, ], name="discount_ep_reward")
        # tracking use
        self.mean_rewards = tf.placeholder(
            tf.float32, name="mean_rewards")

        with tf.name_scope("fc1") as v:
            fc1 =  tf.contrib.layers.fully_connected(inputs=self.input, num_outputs=10, activation_fn=tf.nn.relu,
                                  weights_initializer=tf.contrib.layers.xavier_initializer())

        with tf.name_scope("fc2") as v:
            fc2 =  tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=action_size, activation_fn=tf.nn.relu,
                                  weights_initializer=tf.contrib.layers.xavier_initializer())

        with tf.name_scope("fc3") as v:
            fc3 =  tf.contrib.layers.fully_connected(inputs=fc2, num_outputs=action_size, activation_fn=None,
                                  weights_initializer=tf.contrib.layers.xavier_initializer())

        with tf.name_scope("softmax"):
            self.action_distribution = tf.nn.softmax(fc3)

        with tf.name_scope("loss"):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=fc3, labels=self.action)
            self.loss = tf.reduce_mean(neg_log_prob * self.discount_ep_reward)

        with tf.name_scope("train"):
            self.train_opt = tf.train.AdamOptimizer(
                learning_rate).minimize(self.loss)
        tf.summary.scalar("Loss", self.loss)
        tf.summary.scalar("Reward_mean", self.mean_rewards)
        tf.write_op = tf.summary.merge_all()

    def inferenceActionDistribution(self, sess, current_state):
        return sess.run(self.action_distribution, feed_dict={self.input: current_state.reshape([1, self.state_size])})

    def generateAction(self, sess, current_state):
        distrabution = self.inferenceActionDistribution(sess, current_state)
        return np.random.choice(range(self.action_size), p=distrabution.ravel())

    def step(self, sess, episode_states, episode_actions, discounted_ep_rewards, mean_rewards, episode ,gemma):
        discounted_ep_rewards = self.normalize_rewards(discounted_ep_rewards, gemma)
        # print(mean_rewards)
        loss, _, summary = sess.run([self.loss, self.train_opt, tf.write_op], feed_dict={self.input:np.vstack(np.array(episode_states)),
         self.action:np.vstack(np.array(episode_actions)),
          self.discount_ep_reward: discounted_ep_rewards,
           self.mean_rewards: mean_rewards})
        self.writer.add_summary(summary, episode)
        self.writer.flush()

    def normalize_rewards(self,episode_rewards, gemma):
        discounted_episode_rewards_normalized = np.zeros_like(episode_rewards)
        cumlative = 0.0
        for i in reversed (range( len(discounted_episode_rewards_normalized))):
            cumlative = cumlative * gemma + episode_rewards[i]
            discounted_episode_rewards_normalized[i] = cumlative
        mean = np.mean(discounted_episode_rewards_normalized)
        std = np.std(discounted_episode_rewards_normalized)
        # print((discounted_episode_rewards_normalized-mean))
        discounted_episode_rewards_normalized = (discounted_episode_rewards_normalized-mean)/ ( std)
        # print(discounted_episode_rewards_normalized)
        return discounted_episode_rewards_normalized
