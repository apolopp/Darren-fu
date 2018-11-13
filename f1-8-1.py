import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

lr = 0.06
BATCH_START = 0
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUTS_SIZE = 1
CELL_SIZE = 10
TIME_STEPS = 20


def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    # returned seq, res and xs: shape (batch, step, input)
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


class LSTMRNN(object):
    def __init__(self, n_steps, input_size, outputs_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.outputs_size = outputs_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, outputs_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('iout_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op=tf.train.AdamOptimizer(lr).minimize(self.cost)

        def add_input_layer(self,):
            l_in_x = tf.reshape(self.xs, [-1, input_size], name='2_2D')
            Ws_in = self._weight_variable([self.input_size, self.ell_size])
            bs_in = self._biase_variable([self.cell_size])
            with tf.name_scope('Wx_plus_b'):
                l_in_y = tf.matmul(l_in_x, Ws_in)+bs_in
            self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

        def add_cell(self):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
            with tf.name_scope('init_state'):
                init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            self.cell_outputs, self.final_state = tf.nn.dynamic_rnn(lstm_cell, self.l_in_y, initial_state=self.init_state, time_major=False)

        def add_output_layer(self,):
            l_out_x = tf.reshape(self.cell_outpus, [-1, self.cell_size], name='2_2D')
            Ws_out = self._weight_variable([self.ell_size, self.output_size])
            bs_out = self._biase_variable([self.output_size])
            with tf.name_scope('Wx_plus_b'):
                self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

        def compute_cost(self):

            cost = tf.reduce_mean()
            self.train_op = tf.train.AdamOptimizer(lr).minimize()

            pass

if __name__ == '__main__':
    init = tf.global_variables_initializer()
    sess = tf.SSession()
    sess.run(init)

    for step in range(1000):
        sess.run()

