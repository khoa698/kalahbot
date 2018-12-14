import tensorflow as tf
from MCTS.agent.a3c.utils import FastSaver
from MCTS.environment.side import Side
from MCTS.environment.kalah import KalahEnvironment
from MCTS.agent.a3c.model import ACNetwork
import traceback


class A3Client(object):
    def __init__(self, sess):
        self.sess = sess
        self.network = ACNetwork(state_shape=[2, 8, 1], num_act=7)
        self._restore_model()

    def _restore_model(self):
        saver = FastSaver()
        try:
            checkpoint_path = tf.train.get_checkpoint_state(checkpoint_dir="MCTS/train")
            saver.restore(sess=self.sess, save_path=checkpoint_path.model_checkpoint_path)
        except Exception as e:
            print(traceback.print_exc())

    def sample(self, env: KalahEnvironment) -> (int, float):
        flip_board = env.side_to_play == Side.NORTH
        state = env.board.get_board_image(flipped=flip_board)
        mask = env.get_mask()
        return self.network.sample(state=state, mask=mask)

    def evaluate(self, env: KalahEnvironment) -> (float, float):
        flip_board = env.side_to_play == Side.NORTH
        state = env.board.get_board_image(flipped=flip_board)
        mask = env.get_mask()
        dist, _, value = self.network.evaluate_move(state=state, mask=mask)

        return dist, float(value)

