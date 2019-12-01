import tensorflow as tf
import copy
import numpy as np
import sys


xavier = tf.contrib.layers.xavier_initializer()
bias_const = tf.constant_initializer(0.01)
rand_unif = tf.keras.initializers.RandomUniform(minval=-3e-3,maxval=3e-3)



class Policy_net:
    def __init__(self, name, sess, env, sdim, adim, temp=0.1):

        self.sess = sess 
        self.lstm_n_units = 256
        self.n_hidden1 = 512 
        self.n_hidden2 = 256 
        self.n_hidden3 = 256
        self.sdim = sdim
        self.adim = adim  

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None, self.sdim], name='obs')

            layer_1 = tf.layers.dense(inputs=self.obs, units=self.n_hidden1, activation=tf.tanh, kernel_initializer=xavier, bias_initializer=bias_const)
            layer_2 = tf.layers.dense(inputs=layer_1, units=self.n_hidden2, activation=tf.tanh, kernel_initializer=xavier, bias_initializer=bias_const)
 
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_n_units, state_is_tuple=True)

            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]

            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)

            rnn_in = tf.expand_dims(layer_2, [0])
            step_size = tf.shape(self.obs)[:1]
            state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in) 

            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size, time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, self.lstm_n_units])

            # policy   
            layer_3 = tf.layers.dense(inputs=rnn_out, units=self.n_hidden3, activation=tf.tanh, kernel_initializer=xavier, bias_initializer=bias_const)
            self.act_probs = tf.layers.dense(inputs=tf.divide(layer_3,temp), units=self.adim, activation=tf.nn.softmax, kernel_initializer=rand_unif, bias_initializer=None)

            # value
            layer_4 = tf.layers.dense(inputs=rnn_out, units=self.n_hidden3, activation=tf.tanh, kernel_initializer=xavier, bias_initializer=bias_const)
            self.v_preds = tf.layers.dense(inputs=layer_4, units=1, activation=None, kernel_initializer=rand_unif, bias_initializer=None)


            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, rnn_state, stochastic=True):
        if stochastic:
            return self.sess.run([self.act_stochastic, self.v_preds, self.state_out], feed_dict={self.obs: obs, self.state_in[0]: rnn_state[0], self.state_in[1]: rnn_state[1]})
        else:
            return self.sess.run([self.act_deterministic, self.v_preds, self.state_out], feed_dict={self.obs: obs, self.state_in[0]: rnn_state[0], self.state_in[1]: rnn_state[1]})


    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def num_train_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables(self.scope):
            shape = variable.get_shape()
            variable_parameters = 1
            print(shape)
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters

#------------------------------------------------------------------------------------------------------------------

class PPOTrain:
    def __init__(self, Policy, Old_Policy, sess, gamma=0.95, c_1=0.5): 

        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.sess = sess
        self.gamma = gamma

        pi_trainable = self.Policy.get_trainable_variables()
        old_pi_trainable = self.Old_Policy.get_trainable_variables()

        self.clip_value = tf.placeholder(dtype=tf.float32, shape=(), name='eps_ppo')
        self.c_2 = tf.placeholder(dtype=tf.float32, shape=(), name='entropy_ppo')

        # assign_operations for policy parameter values to old policy parameters
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assign_ops.append(tf.assign(v_old, v))

        # inputs for train_op
        with tf.variable_scope('train_inp'):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')
             

        act_probs = self.Policy.act_probs
        act_probs_old = self.Old_Policy.act_probs

        # probabilities of actions which agent took with policy
        act_probs = act_probs * tf.one_hot(indices=self.actions, depth=act_probs.shape[1])
        act_probs = tf.reduce_sum(act_probs, axis=1)

        # probabilities of actions which agent took with old policy
        act_probs_old = act_probs_old * tf.one_hot(indices=self.actions, depth=act_probs_old.shape[1])
        act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

        with tf.variable_scope('loss/clip'):
            ratios = tf.exp(tf.log(tf.maximum(act_probs,1.0e-8)) - tf.log(tf.maximum(act_probs_old,1.0e-8)))
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.clip_value, clip_value_max=1 + self.clip_value)
            loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
            loss_clip = tf.reduce_mean(loss_clip)

        # construct computation graph for loss of value function
        with tf.variable_scope('loss/vf'):
            v_preds = self.Policy.v_preds
            loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
            loss_vf = tf.reduce_mean(loss_vf)

        # construct computation graph for loss of entropy bonus
        with tf.variable_scope('loss/entropy'):
            entropy_form = 'shannon'
 
            if (entropy_form == 'shannon'):
               # shannon entropy
               entropy = -tf.reduce_sum(self.Policy.act_probs *
                                     tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), axis=1)
            elif (entropy_form == 'renyi'):
               # Renyi entropy
               renyi_alpha = tf.constant(0.5) 
               entropy = 1.0/(1.0-renyi_alpha)*tf.reduce_sum(tf.pow(self.Policy.act_probs,renyi_alpha), axis=1) 
               entropy = tf.reduce_mean(entropy, axis=0) 
            elif (entropy_form == 'sharma_taneja'): 
               # sharma-Taneja entropy
               alpha_st = 1.0 
               beta_st = 0.5
               factor = 1.0/(2.0**(1.0-alpha_st) - 2.0**(1.0-beta_st)) 
               sum1 = tf.reduce_sum(tf.pow(self.Policy.act_probs,alpha_st), axis=1) 
               sum2 = tf.reduce_sum(tf.pow(self.Policy.act_probs,beta_st), axis=1)   
               entropy = factor*(sum1 - sum2)  
               entropy = tf.reduce_mean(entropy, axis=0) 

        with tf.variable_scope('loss'):
            loss = loss_clip - c_1 * loss_vf + self.c_2 * entropy
            loss = -loss  

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
        self.train_op = optimizer.minimize(loss, var_list=pi_trainable)

    def train(self, obs, actions, rewards, v_preds_next, gaes, epsilon_ppo, c_2):
        c, h = self.Policy.state_init
        fd = {self.Policy.obs: obs, self.Policy.state_in[0]: c, self.Policy.state_in[1]: h, self.Old_Policy.obs: obs, self.Old_Policy.state_in[0]: c, self.Old_Policy.state_in[1]: h, self.actions: actions, self.rewards: rewards, self.v_preds_next: v_preds_next, self.gaes: gaes, self.clip_value: epsilon_ppo, self.c_2: c_2}
        _ = self.sess.run([self.train_op], feed_dict=fd)

    def assign_policy_parameters(self):
        return self.sess.run(self.assign_ops)

    def get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes

