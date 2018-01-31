# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from Stack import *
class LSTMConfig(object):
    embedding_size=256
    vocab_size=1000
    num_layers=2
    num_classes=104
    num_steps=400
    hidden_size=256
    dropout_keep_prob=1.0
    learning_rate=1e-4
    batch_size=32 #无法并行,因此只能为1
    num_epochs=50
    print_per_batch=200
    save_per_batch=200
    max_grad_norm=5
    rnn='lstm'

class StackLSTM(object):
    def __init__(self,config,start_mark,end_mark):
        self.config=config
        self.input=tf.placeholder(tf.int64,[config.batch_size,self.config.num_steps],name='input')
        self.label=tf.placeholder(tf.int64,[config.batch_size,self.config.num_classes],name='label')
        self.keep_prob=tf.placeholder(tf.float32,name='keep_prob')
        self.start=start_mark
        self.end=end_mark
        self.rnn()
    def rnn(self):
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size,state_is_tuple=True)
        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.config.hidden_size)
        with tf.device('/cpu:0'):
            embedding=tf.get_variable('embedding',[self.config.vocab_size,self.config.embedding_size])
            embedding_inputs=tf.nn.embedding_lookup(embedding,self.input)
            embedding_inputs=tf.nn.dropout(embedding_inputs,0.8)
        with tf.name_scope('rnn'):
            outputs,_=self._build_rnn_graph_lstm(embedding_inputs,self.config)
            # print(outputs)
            last=outputs
            # print(last)
            # output = tf.reshape(tf.concat(outputs, 1), [-1, self.config.hidden_size])
            with tf.name_scope('score'):
                #全连接层
                fc=tf.layers.dense(last,self.config.hidden_size,name='fc1')
                fc=tf.contrib.layers.dropout(fc,self.keep_prob)
                fc=tf.nn.relu(fc)
                #输出层
                self.logits=tf.layers.dense(fc,self.config.num_classes,name='fc2')
                self.predict_class=tf.argmax(tf.nn.softmax(self.logits),1)

        with tf.name_scope('optimize'):
            cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.label)
            self.loss=tf.reduce_mean(cross_entropy)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                              self.config.max_grad_norm)
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.train.get_or_create_global_step())

        with tf.name_scope('accuracy'):
            correct_pre=tf.equal(tf.argmax(self.label,1),self.predict_class)
            self.acc=tf.reduce_mean(tf.cast(correct_pre,tf.float32))


    def _build_rnn_graph_lstm(self, inputs, config):
        """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
        def make_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.dropout_keep_prob)
            return cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(config.batch_size, tf.float32)
        state = self._initial_state
        # Simplified version of tensorflow_models/tutorials/rnn/rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
        #                            initial_state=self._initial_state)
        outputs = []

        def func_push(state_i,i):
            self.state_stack[i].push(state_i)
            return state_i


        #-----------------特殊情况需要保留名称------------------
        # def updateState(state,time_step):
        #     # state = ((state[0][0], state[0][1]), (state[1][0],state[1][1]))
        #
        #     (out, newstate) = cell(inputs[:, time_step, :], state)
        #     # print('------------------------------hhhhhh----------------------------')
        #     tf.get_variable_scope().reuse_variables()
        #     return newstate
        # nameSet=[word_to_id['Import'],word_to_id['ClassDef'],word_to_id['FunctionDef'],word_to_id['Assign'],word_to_id['AsyncFunctionDef'],word_to_id['Attribute']]
        # def f_default(state):
        #     return state,state
        #
        # def func_push(state, time_step):
        #     #add特殊情况需要保留名称
        #     state,newState = tf.cond(tf.logical_or(
        #         tf.logical_or(tf.equal(self._input_data[0][time_step-1], nameSet[0]), tf.equal(self._input_data[0][time_step-1], nameSet[1])),
        #         tf.logical_or(tf.equal(self._input_data[0][time_step-1], nameSet[2]), tf.equal(self._input_data[0][time_step-1], nameSet[3])),
        #     ),lambda: updateState(state, time_step), lambda: f_default(state))
        #
        #     self.state_stack.push(newState)
        #     return state[0][0], state[0][1], state[1][0], state[1][1]
        # #-------------------------------------------------------------

        def func_pop(i):
            # (cell_output,state)=cell(inputs[:,time_step,:],state)
            state_i=self.state_stack[i].pop()
            # (cell_output,state)=cell(cell_output,state)
            return state_i
        #def func_pop(state):
        #    #------增加一层---------------#
        #    w = tf.get_variable(
        #        "state_w", [2*config.hidden_size, config.hidden_size], dtype=tf.float32)
        #    # b = tf.get_variable("state_b", [1,config.hidden_size], dtype=data_type())
        #    old_state=self.state_stack.pop()
        #    concat_state=[[],[]]
        #    new_state=[[],[]]
        #    # concat_state=tf.concat([old_state,state],1)
        #    # concat_state=tf.reshape(concat_state,[1,-1])
        #    for i in range(2):
        #        for j in range(2):
        #            concat_state[i].append(tf.concat([old_state[i][j],state[i][j]],1))
        #            new_state[i].append(tf.matmul(concat_state[i][j], w))
        #    return new_state[0][0], new_state[0][1], new_state[1][0], new_state[1][1]

        def func_default(state_i):
            return state_i

        with tf.variable_scope("RNN"):
            self.state_stack=[Stack() for _ in range(self.config.batch_size)]
        #     for time_step in range(self.config.num_steps):
        #         if time_step > 0: tf.get_variable_scope().reuse_variables()
        #         # print(self.input)
        #         new_state=tf.cond(tf.equal(self.input[0][time_step],self.start),
        #                           lambda:func_push(state),lambda:func_default(state))
        #         new_state=tf.cond(tf.equal(self.input[0][time_step],self.end),
        #                           lambda:func_pop(time_step,state),lambda:func_default(state))
        #         state=((new_state[0],new_state[1]),(new_state[2],new_state[3]))
        #
        #         (outputs, state) = cell(inputs[:, time_step, :], state)
        #         #outputs.append(cell_output)
        # # output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])

        #----------------------------------------------------------------------
            for time_step in range(self.config.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                STATES=[]
                NEW_STATES=[]
                for i in range(config.batch_size):
                    # print(state)
                    state_i=[]
                    for m in range(len(state)):#layers
                        for n in range(len(state[m])):# c & h
                            # print(new_state[m][n][i])
                            state_i.append(tf.reshape(state[m][n][i],[1,-1]))
                    STATES.append(state_i)
                # print(STATES)

                for i in range(config.batch_size):
                    new_state_i=tf.cond(tf.equal(self.input[i][time_step],self.start),
                                             lambda:func_push(STATES[i],i),lambda:func_default(STATES[i]))
                    new_state_i=tf.cond(tf.equal(self.input[i][time_step],self.end),
                                      lambda:func_pop(i),lambda:func_default(STATES[i]))
                    state_i=new_state_i
                    # state_i=((new_state_i[0],new_state_i[1]),(new_state_i[2],new_state_i[3]))
                    NEW_STATES.append(state_i)
                # print(NEW_STATES)
                NEW_STATES=tf.concat(NEW_STATES,1)
                # print(tf.concat(NEW_STATES,1)[0])
                state=((NEW_STATES[0],NEW_STATES[1]),(NEW_STATES[2],NEW_STATES[3]))
                print(state)
                #只需要最后一个step的output
                # print(state)
                (outputs, state) = cell(inputs[:, time_step, :], state)
            #-----------------------------------------------------------------------------
        return outputs, state
