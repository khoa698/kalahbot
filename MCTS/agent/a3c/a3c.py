from __future__ import print_function
from MCTS.environment.kalah import KalahEnvironment
from MCTS.environment.side import Side
from MCTS.environment.move import Move
import random
from collections import namedtuple
import numpy as np
import tensorflow as tf
from MCTS.agent.a3c.model import ACNetwork
from MCTS.agent.a3c.agent import Agent
import scipy.signal
import distutils.version
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def generate_training_batch(rollout, gamma: float, bootstrap_value=0.0):
    """Computes the advantages and prepares the batch for training
       The bootstrap value is used only for incomplete episodes which is not the case in our case where we always
       play a full Mancala Game. We can think of the environment's MDP final state as a loop state
       which always produces a reward of 0. This justifies the default value of 0 which we always use.
    """
    states = np.asarray(rollout.states)
    actions = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    values = np.asarray(rollout.values)
    masks = np.asarray(rollout.masks)

    # The advantage function is "Generalized Advantage Estimation"
    # For more details: https://arxiv.org/abs/1506.02438
    rewards_plus = np.concatenate((rewards.tolist(), [bootstrap_value]))
    discounted_rewards = discount(rewards_plus[:-1], gamma)
    value_plus = np.concatenate((values.tolist(), [bootstrap_value]))
    advantages = rewards + gamma * value_plus[1:] - value_plus[:-1]
    advantages = discount(advantages, gamma)

    return Batch(states, actions, advantages, discounted_rewards, masks)


Batch = namedtuple("Batch", ["states", "actions", "advantages", "discounted_rewards", "masks"])


class Rollout(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.masks = []
        self.win = 0

    def add(self, state: np.array, action: int, reward: int, value: int, mask: [float]):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.masks.append(mask)

    def update_last_reward(self, reward):
        assert len(self.rewards) > 0
        self.rewards[-1] = reward

    def add_win(self):
        self.win = 1


class RunnerThread(object):
    """
One of the key distinctions between a normal environment and a universe environment
is that a universe environment is _real time_.  This means that there should be a thread
that would constantly interact with the environment and tell it what to do.  This thread is here.
"""

    def __init__(self, env: KalahEnvironment, ac_net: ACNetwork):
        self.env = env
        self.ac_net = ac_net
        self.sess = None
        self.opp_agent = None
        self.trainer_side = None

    def run(self, sess: tf.Session, opp_agent: Agent) -> Rollout:
        self.sess = sess
        self.opp_agent = opp_agent
        with self.sess.as_default():
            return self._run()

    def _run(self):
        # Choose randomly the side to play
        self.trainer_side = Side.SOUTH if random.randint(0, 1) == 0 else Side.NORTH
        # Reset the environment so everything is in a clean state.
        self.env.reset()

        rollout = env_runner(self.env, self.trainer_side, self.ac_net, self.opp_agent)

        return rollout



def env_runner(env, trainer_side, ac_net, opp_agent):
    """
The logic of the thread runner.  In brief, it constantly keeps on running
the policy, and as long as the rollout exceeds a certain length, the thread
runner appends the policy to the queue.
"""
    rollout = Rollout()

    while not env.has_game_ended():
        # There is no choice if only one action is left. Taking that action automatically must be seen as
        # a characteristic behaviour of the environment. This helped the learning of the agent
        # to be more numerically stable (this is an empirical observation).
        if len(env.get_valid_moves()) == 1:
            action_left_to_perform = env.get_valid_moves()[0]
            env.do_move(action_left_to_perform)
            continue

        if env.side_to_play == trainer_side:
            # If the agent is playing as NORTH, it's input would be a flipped board
            flip_board = env.side_to_play == Side.NORTH
            state = env.board.get_board_image(flipped=flip_board)
            mask = env.get_mask()

            action, value = ac_net.sample(state, mask)
            # Because the pie move with index 0 is ignored, the action indexes must be shifted by one
            reward = env.do_move(Move(trainer_side, action + 1))
            rollout.add(state, action, reward, value, mask)
        else:
            assert env.side_to_play == Side.opposite(trainer_side)
            action = opp_agent.produce_action(env.board.get_board_image(),
                                              env.get_mask(),
                                              env.side_to_play)
            env.do_move(Move(env.side_to_play, action + 1))

        # We replace the partial reward of the last move with the final reward of the game
    final_reward = env.calculate_score_diff(trainer_side)
    rollout.update_last_reward(final_reward)

    if env.get_winner() == trainer_side:
        rollout.add_win()
    return rollout


class A3C(object):
    def __init__(self, env, task):
        """
An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
should be computed.
"""

        self.env = env
        self.task = task

        # Performance statistics
        self.episodes_reward = []
        self.episodes_length = []
        self.episodes_mean_value = []
        self.wins = 0
        self.games = 0

        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                # self.network = LSTMPolicy(env.observation_space.shape, env.action_space.n) replace Open AI policy
                self.network = ACNetwork(state_shape=[2, 8, 1], num_act=7)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                # self.local_network = pi = LSTMPolicy(env.observation_space.shape, env.action_space.n) replace Open AI policy
                self.local_network = pi = self.network
                pi.global_step = self.global_step

            self.action = tf.placeholder(shape=[None], dtype=tf.int32)
            self.action_one_hot = tf.one_hot(self.action, 7, dtype=tf.float32)
            self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
            self.advantage = tf.placeholder(shape=[None], dtype=tf.float32)

            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)

            act_log_prob = tf.reduce_sum(log_prob_tf * self.action_one_hot, [1])

            # loss of value function
            self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(pi.value, [-1])))
            self.entropy = -tf.reduce_sum(prob_tf * log_prob_tf)
            self.policy_loss = -tf.reduce_sum(act_log_prob * self.advantage)

            self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

            # Get gradients from local network using local losses and clip them to avoid exploding gradients
            self.gradients = tf.gradients(self.loss, pi.vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 100.0)

            # Define operation for downloading the weights from the parameter server (ps)
            # on the local model of the worker
            self.down_sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.vars, self.network.vars)])

            # Define the training operation which applies the gradients on the parameter server network (up sync)
            optimiser = tf.train.RMSPropOptimizer(learning_rate=0.0007)
            grads_and_global_vars = list(zip(grads, self.network.vars))
            inc_step = self.global_step.assign_add(tf.shape(self.action)[0])
            self.train_op = tf.group(*[optimiser.apply_gradients(grads_and_global_vars), inc_step])

            # 20 represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            # The larger local steps is, the lower is the variance in our policy gradients estimate
            # on the one hand;  but on the other hand, we get less frequent parameter updates, which
            # slows down learning.  In this code, we found that making local steps be much
            # smaller than 20 makes the algorithm more difficult to tune and to get to work.
            self.env_runner = RunnerThread(KalahEnvironment(), pi)

            episode_size = tf.to_float(tf.shape(pi.value)[0])

            if use_tf12_api:
                tf.summary.scalar("model/policy_loss", self.policy_loss / episode_size)
                tf.summary.scalar("model/value_loss", self.value_loss / episode_size)
                tf.summary.scalar("model/entropy", self.entropy / episode_size)
                tf.summary.scalar("model/grad_global_norm", self.grad_norms)
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.vars))
                self.summary_op = tf.summary.merge_all()

            else:
                tf.scalar_summary("model/policy_loss", self.policy_loss / episode_size)
                tf.scalar_summary("model/value_loss", self.value_loss / episode_size)
                tf.scalar_summary("model/entropy", self.entropy / episode_size)
                tf.scalar_summary("model/grad_global_norm", self.grad_norms)
                tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.vars))
                self.summary_op = tf.merge_all_summaries()

            self.summary_writer = None
            self.local_steps = 0

    def play(self, sess: tf.Session, opp_agent: Agent, summary_writer: tf.summary.FileWriter):
        self.summary_writer = summary_writer
        sess.run(self.down_sync)
        rollout = self.env_runner.run(sess, opp_agent)
        self.train(sess, rollout)

    def train(self, sess: tf.Session, rollout: Rollout, sum_period=100):
        """
train_result_a3c grabs a rollout that's been produced by the thread runner,
and updates the parameters.
"""

        # Record the statistics of this new rollout
        self.episodes_reward.append(np.sum(rollout.rewards))
        self.episodes_length.append(len(rollout.states))
        self.episodes_mean_value.append(np.mean(rollout.values))
        self.wins += rollout.win
        self.games += 1

        batch = generate_training_batch(rollout, gamma=0.99)

        feed_dict = {
            self.local_network.state: batch.states,
            self.action: batch.actions,
            self.advantage: batch.advantages,
            self.target_v: batch.discounted_rewards,
            self.local_network.mask: batch.masks,
        }

        should_compute_summary = self.task == 0 and self.local_steps % sum_period == 0

        if should_compute_summary:
            fetches = [self.train_op, self.global_step, self.summary_op]
        else:
            fetches = [self.train_op, self.global_step]

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            # Keep only the last sum_period entries
            self.episodes_reward = self.episodes_reward[-sum_period:]
            self.episodes_length = self.episodes_length[-sum_period:]
            self.episodes_mean_value = self.episodes_mean_value[-sum_period:]

            # Add stats to tensorboard
            summary = tf.Summary()
            mean_reward = np.mean(self.episodes_reward[-sum_period:])
            mean_length = np.mean(self.episodes_length[-sum_period:])
            mean_value = np.mean(self.episodes_mean_value[-sum_period:])
            summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
            summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
            summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
            summary.value.add(tag='Perf/WinRate', simple_value=float(self.wins / self.games))

            self.summary_writer.add_summary(summary, fetched[1])
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[2]), fetched[1])
            self.summary_writer.flush()

            # Restart the win rate statistics
            self.wins = self.games = 0
            self.summary_writer = None
        self.local_steps += 1