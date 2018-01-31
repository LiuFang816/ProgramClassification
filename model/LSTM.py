# -*- coding: utf-8 -*-
import tensorflow as tf
class LSTMConfig(object):
    embedding_size=256
    vocab_size=5000
    num_layers=2
    num_steps=400
    num_classes=104
    hidden_size=256
    dropout_keep_prob=1.0
    learning_rate=1e-3
    batch_size=64
    num_epochs=50
    print_per_batch=100
    save_per_batch=50
    max_grad_norm=5
    rnn='lstm'

class LSTM(object):
    def __init__(self,config):
        self.config=config
        self.input=tf.placeholder(tf.int64,[config.batch_size,self.config.num_steps],name='input')
        self.label=tf.placeholder(tf.int64,[config.batch_size,self.config.num_classes],name='label')
        self.keep_prob=tf.placeholder(tf.float32,name='keep_prob')
        self.rnn()
    def rnn(self):
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size,state_is_tuple=True)
        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.config.hidden_size)
        def drop_out():
            if self.config.rnn=='lstm':
                cell=lstm_cell()
            else:
                cell=gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.config.dropout_keep_prob)
        with tf.device('/cpu:0'):
            embedding=tf.get_variable('embedding',[self.config.vocab_size,self.config.embedding_size])
            embedding_inputs=tf.nn.embedding_lookup(embedding,self.input)
            embedding_inputs=tf.nn.dropout(embedding_inputs,0.8)
        with tf.name_scope('rnn'):
            cells=[drop_out() for _ in range(self.config.num_layers)]
            rnn_cell=tf.contrib.rnn.MultiRNNCell(cells,state_is_tuple=True)
            outputs,_=tf.nn.dynamic_rnn(rnn_cell,embedding_inputs,dtype=tf.float32)
            # output = tf.reshape(tf.concat(outputs, 1), [-1, self.config.hidden_size])
            last=outputs[:,-1,:]#取最后一个timestep的输出作为结果

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