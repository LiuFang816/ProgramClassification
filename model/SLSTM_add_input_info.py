# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import numpy as np
import tensorflow as tf
from lstm.tools import Stack

from lstm.tools import data_reader
from lstm.tools.generate_test_data import is_nonterminal, is_terminal, stop_words

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
  "model", "small",
  "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", '../data/',
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", '../data/test/',
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool("decode", False,
                  "Set to True for interactive decoding.")
flags.DEFINE_bool("generate", False, "Set to True for interactive generating.")
flags.DEFINE_bool("test", False, "Set to True for interactive generating.")

FLAGS = flags.FLAGS

word_to_id = data_reader.get_word_to_id(FLAGS.data_path)
nameSet = [word_to_id['ClassDef'], word_to_id['FunctionDef'], word_to_id['Assign'], word_to_id['AsyncFunctionDef']]


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class NoInitInput(object):
  """The input data."""

  def __init__(self, config, data, name=None, isDecode=False):
    if isDecode:
      self.num_steps = len(data) + 1
      X = [[0]]
      X[0] = data
      data = tf.convert_to_tensor(X, name="data", dtype=tf.int32)
      self.input_data = data
      self.targets = data  # 这个没有用
      self.batch_size = 1
      self.epoch_size = 1
    else:
      self.batch_size = batch_size = config.batch_size
      self.num_steps = num_steps = config.num_steps
      self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
      self.input_data, self.targets = data_reader.data_producer(
        data, batch_size, num_steps, name=name)


class NoInitModel(object):
  """The NoInit model."""

  def __init__(self, is_training, config, input_, START_MARK, END_MARK, PAD_MARK):
    self._input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps - 1
    size = config.hidden_size
    vocab_size = config.vocab_size

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
        lstm_cell, output_keep_prob=config.keep_prob)
    # cell=lstm_cell
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())
    self.state_stack = Stack.Stack()

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
        "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)
    self._input_data = input_.input_data

    test = []
    self._test = test
    # self._inputs=inputs
    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
    outputs = []
    state = self._initial_state  # [1,hidden_size]

    # def func_push(state,batch):
    #     self.state_stack.push(state)
    #     return state[0][0][batch], state[0][1][batch], state[1][0][batch], state[1][1][batch]
    #
    # def func_pop(batch):
    #     state = self.state_stack.pop()
    #     return state[0][0][batch], state[0][1][batch], state[1][0][batch], state[1][1][batch]
    #
    # def func_default(state,batch):
    #     return state[0][0][batch], state[0][1][batch], state[1][0][batch], state[1][1][batch]

    def updateState(state, time_step):
      state = ((state[0][0], state[0][1]), (state[1][0], state[1][1]))

      (out, newstate) = cell(inputs[:, time_step, :], state)

      # print('------------------------------hhhhhh----------------------------')
      tf.get_variable_scope().reuse_variables()
      return newstate

    def f_default(state):
      return state

    def func_proceed_name(state, time_step):
      (cell_output, newstate) = cell(inputs[:, time_step, :], state)
      tf.get_variable_scope().reuse_variables()
      (cell_output, newstate) = cell(inputs[:, time_step, :], newstate)
      tf.get_variable_scope().reuse_variables()

      return newstate

    def func_not_proceed_name(state):
      return state

    def func_push(state):
      """
      传入一个LSTMTuple
      1. push
      2. 返回4个state
      :param state:
      :param time_step:
      :return:
      """

      self.state_stack.push(state)
      return state[0][0], state[0][1], state[1][0], state[1][1]

    def func_pop(time_step,state):
      (cell_output,state)=cell(inputs[:,time_step,:],state)
      state=self.state_stack.pop()
      (cell_output,state)=cell(cell_output,state)
      return state[0][0],state[0][1],state[1][0],state[1][1]

    def func_default(state):
      return state[0][0], state[0][1], state[1][0], state[1][1]

    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        # if time_step<num_steps/2:
        #   new_state=func_push(state,time_step)
        # else:
        #   new_state=func_default(state)
        #
        # if time_step>=num_steps/2:
        #   new_state=func_pop()
        # else:
        #   new_state=func_default(state)

        maybeUpdateState = tf.cond(tf.logical_or(
          tf.logical_or(tf.equal(self._input_data[0][time_step-1], nameSet[0]), tf.equal(self._input_data[0][time_step-1], nameSet[1])),
          tf.logical_or(tf.equal(self._input_data[0][time_step-1], nameSet[2]), tf.equal(self._input_data[0][time_step-1], nameSet[3])),
        ),lambda: func_proceed_name(state, time_step),lambda: func_not_proceed_name(state))

        new_state = tf.cond(tf.equal(self._input_data[0][time_step], START_MARK),
                            lambda: func_push(maybeUpdateState), lambda: func_default(maybeUpdateState))
        #state经过push更新之后需要回到初始时的状态，然后进行后续的cell计算
        new_state=state[0][0], state[0][1], state[1][0], state[1][1]
        # # fixme : func_default(state) 是不是应该是func_default(new_state)?
        new_state = tf.cond(tf.equal(self._input_data[0][time_step], END_MARK), lambda: func_pop(),
                            lambda: func_default(state))

        state = ((new_state[0], new_state[1]), (new_state[2], new_state[3]))

        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(outputs, 1), [-1, size])
    softmax_w = tf.get_variable(
      "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
      [logits],
      [tf.reshape(input_.targets, [-1])],
      [tf.ones([batch_size * num_steps], dtype=data_type())])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state
    self._targets = input_.targets
    self._logits = logits

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
      zip(grads, tvars),
      global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
      tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def step(self, session):
    output = session.run(self._logits)
    return output

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 0.1
  max_grad_norm = 5
  num_layers = 2
  max_data_row = None
  num_steps = 60
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 1
  vocab_size = 5000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 1024
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 200
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1024
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 200
  vocab_size = 10000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


def run_epoch(session, model, eval_op=None, verbose=False, id_to_word=None, end_id=None, isDecode=False):
  """Runs the model on the given data."""
  if isDecode:
    output = model.step(session)
    return output
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  SUM = 0
  correct_tok = 0

  fetches = {
    "cost": model.cost,
    "final_state": model.final_state,
    "input_data": model._input_data,
    "targets": model._targets,
    "pred_output": model._logits
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    # todo add 计算accuracy
    midInputData = vals["input_data"]
    midTargets = vals["targets"]
    pred_output = vals["pred_output"]
    try:
      for i in range(model.input.batch_size):
        for j in range(model.input.num_steps - 1):
          SUM += 1
          trueOutput = id_to_word[midTargets[i][j]]
          tmp = list(pred_output[i * (model.input.num_steps - 1) + j])
          # todo topN这里注释
          predOutput = id_to_word[tmp.index(max(tmp))]
          if midInputData[i][j] == end_id:
            SUM -= 1
            break
          if trueOutput == predOutput:
            correct_tok += 1
            # todo topN使用这里
            # predOutput=[]
            # for m in range(N):
            #     index=tmp.index(max(tmp))
            #     predOutput.append(id_to_word[index])
            #     tmp[index]=-100
            # if trueOutput in predOutput:
            #     correct_tok+=1
    except:
      print("--------ERROR 计算acc----")
      pass

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))

      if True:
        midInputData = vals["input_data"]
        midTargets = vals["targets"]
        pred_output = vals["pred_output"]
        global num_steps
        try:
          for i in range(1):
            inputStr = ''
            trueOutput = ''
            predOutput = ''
            for j in range(num_steps):
              inputStr += id_to_word[midInputData[i][j]] + ' '

              trueOutput += id_to_word[midTargets[i][j]] + ' '

              tmp = list(pred_output[i * num_steps + j])
              predOutput += id_to_word[tmp.index(max(tmp))] + ' '
            print('Input: %s \n True Output: %s \n Pred Output: %s \n' % (inputStr, trueOutput, predOutput))
        except:
          pass

  acc = correct_tok * 1.0 / SUM
  print("Accuracy : %.3f" % acc)
  return np.exp(costs / iters)


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


num_steps = 0


def train():
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")
  config = get_config()

  word_to_id = data_reader.get_word_to_id(FLAGS.data_path)
  # todo raw_data还应包含weights
  raw_data = data_reader.raw_data(max_data_row=config.max_data_row, data_path=FLAGS.data_path, word_to_id=word_to_id,
                                  max_length=config.num_steps)
  train_data, test_data, voc_size, end_id, _, START_MARK, END_MARK, PAD_MARK = raw_data
  id_to_word = data_reader.reverseDic(word_to_id)

  config = get_config()
  global num_steps
  num_steps = config.num_steps - 1

  eval_config = get_config()
  # eval_config.batch_size = 1
  # eval_config.num_steps = 1

  # 使用动态vocab_size
  # config.vocab_size=voc_size
  # eval_config.voc_size=voc_size


  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = NoInitInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = NoInitModel(is_training=True, config=config, input_=train_input,
                        START_MARK=START_MARK, END_MARK=END_MARK, PAD_MARK=PAD_MARK)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    # with tf.name_scope("Valid"):
    #     valid_input = NoInitInput(config=config, data=valid_data, name="ValidInput")
    #     with tf.variable_scope("Model", reuse=True, initializer=initializer):
    #         mvalid = NoInitModel(is_training=False, config=config, input_=valid_input)
    #     tf.scalar_summary("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = NoInitInput(config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = NoInitModel(is_training=False, config=eval_config, input_=test_input,
                            START_MARK=START_MARK, END_MARK=END_MARK, PAD_MARK=PAD_MARK)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True, id_to_word=id_to_word,
                                     end_id=end_id)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        # valid_perplexity = run_epoch(session, mvalid, id_to_word=id_to_word)
        # print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      test_perplexity = run_epoch(session, mtest, id_to_word=id_to_word, end_id=end_id)
      print("Test Perplexity: %.3f" % test_perplexity)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


def decode():
  choice = ['1', '2', '3', '4', '5', 'q']
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")
  config = get_config()

  word_to_id = data_reader.get_word_to_id(FLAGS.data_path)
  # todo raw_data还应包含weights
  raw_data = data_reader.raw_data(max_data_row=config.max_data_row, data_path=FLAGS.data_path, word_to_id=word_to_id,
                                  max_length=config.num_steps)
  train_data, test_data, voc_size, end_id, _, START_MARK, END_MARK, PAD_MARK = raw_data
  id_to_word = data_reader.reverseDic(word_to_id)

  config = get_config()
  global num_steps
  num_steps = config.num_steps - 1

  sys.stdout.write("> ")
  sys.stdout.flush()
  token = sys.stdin.readline().strip('\n').split(' ')
  for i in range(len(token)):
    if token[i] not in word_to_id:
      token[i] = word_to_id['UNK']
    else:
      token[i] = word_to_id[token[i]]

  while True:
    with tf.Graph().as_default():
      initializer = tf.random_uniform_initializer(-config.init_scale,
                                                  config.init_scale)

      with tf.name_scope("Train"):
        decode_input = NoInitInput(config=config, data=token, name="TrainInput", isDecode=True)
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
          decode_model = NoInitModel(is_training=False, config=config, input_=decode_input,
                                     START_MARK=START_MARK, END_MARK=END_MARK, PAD_MARK=PAD_MARK)

          # ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
          # if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
          #     print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
          #     # model.saver.restore(session, ckpt.model_checkpoint_path)
          # else:
          #     print("Created model with fresh parameters.")
      Config = tf.ConfigProto()
      Config.gpu_options.allow_growth = True
      sv = tf.train.Supervisor(logdir=FLAGS.save_path)
      with sv.managed_session(config=Config) as session:
        output = run_epoch(session, decode_model, id_to_word, end_id=end_id, isDecode=True)
        # output = decode_model.step(session)
        # print(output)
        tmp = list(output[-1])

        # output=id_to_word[tmp.index(max(tmp))]
        # print('next token --> %s'%output)
        # ------------

        # todo 输出top5
        predOutput = []
        count = 0
        while count < 5:
          index = tmp.index(max(tmp))
          if index == PAD_MARK:
            tmp[index] = -100
            continue
          predOutput.append(id_to_word[index])
          count += 1
          tmp[index] = -100

        print('next token --> ')
        for i in range(len(predOutput)):
          print('%d: %s' % (i + 1, predOutput[i]))
        print('You Choose: ')
        x = sys.stdin.readline().strip('\n')
        if x != 'q' and x in choice:
          token.append(word_to_id[predOutput[int(x) - 1]])
        elif x not in choice:
          if x not in word_to_id:
            token.append(word_to_id['UNK'])
          else:
            token.append(word_to_id[x])
        else:
          break


def test(type, filename):
  # wfname='data/'+type+'.txt'
  # wf=open(wfname,'w')
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")
  config = get_config()

  word_to_id = data_reader.get_word_to_id(FLAGS.data_path)
  # todo raw_data还应包含weights
  raw_data = data_reader.raw_data(FLAGS.data_path, word_to_id, config.num_steps)
  train_data, test_data, voc_size, end_id, _, START_MARK, END_MARK, PAD_MARK = raw_data
  id_to_word = data_reader.reverseDic(word_to_id)

  config = get_config()
  global num_steps
  num_steps = config.num_steps - 1
  SUM = 0
  correct_tok = 0

  f = open(filename)
  data = f.readlines()
  for i in range(len(data)):
    try:
      SUM += 1
      code = data[i].strip('\n').split(' ')
      # print(data)
      testInput = code[0:len(code) - 1]
      testTarget = code[-1]
      if testTarget not in word_to_id:
        testTarget = 'UNK'
      for j in range(len(testInput)):
        if testInput[j] not in word_to_id:
          testInput[j] = word_to_id['UNK']
        else:
          testInput[j] = word_to_id[testInput[j]]

      with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.name_scope("Train"):
          decode_input = NoInitInput(config=config, data=testInput, name="TrainInput", isDecode=True)
          with tf.variable_scope("Model", reuse=None, initializer=initializer):
            decode_model = NoInitModel(is_training=True, config=config, input_=decode_input,
                                       START_MARK=START_MARK, END_MARK=END_MARK, PAD_MARK=PAD_MARK)

            # ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
            # if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            #     print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            #     # model.saver.restore(session, ckpt.model_checkpoint_path)
            # else:
            #     print("Created model with fresh parameters.")

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
          output = run_epoch(session, decode_model, id_to_word, end_id=end_id, isDecode=True)
          tmp = list(output[-1])

          # top10
          predOutput = []
          count = 0
          while count < 10:
            index = tmp.index(max(tmp))
            # todo fix me
            if index == PAD_MARK or id_to_word[index] in stop_words:
              tmp[index] = -100
              continue
            predOutput.append(id_to_word[index])
            count += 1
            tmp[index] = -100
            # sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

        if type == 'T':
          for i in range(len(predOutput)):
            if is_terminal(predOutput[i]):
              print("%s,%s" % (predOutput[i], testTarget))
              if (predOutput[i] == testTarget):
                correct_tok += 1
              break
        else:
          for i in range(len(predOutput)):
            if is_nonterminal(predOutput[i]):
              print("%s,%s" % (predOutput[i], testTarget))
              if (predOutput[i] == testTarget):
                correct_tok += 1
              break
      print(' %d %d' % (correct_tok, SUM))
      acc = correct_tok * 1.0 / SUM
      print("Accuracy : %.3f" % acc)
    except:
      pass
  acc = correct_tok * 1.0 / SUM
  print("Final Accuracy : %.3f" % acc)


def main(_):
  if FLAGS.decode:
    decode()
  if FLAGS.test:
    test('NT', r'../data/train_nonterminal-60-60.txt')
  else:
    train()


if __name__ == "__main__":
  tf.app.run()
