# -*- coding: utf-8 -*-
# 这里决定生成哪个模型的测试 ：  from 模型名字 import *
from newmodel.basic_lstm import *
from tools import data_reader
import json

def new_run_epoch(session, model, eval_op=None, verbose=False, id_to_word=None,end_id=None,N=10,save_path=None):
    """Runs the model on the given data."""

    f = open(save_path,'w')
    # start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

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
        # for step in range(1):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps-1

        midTargets=vals["targets"]
        pred_output=vals["pred_output"]
        for i in range(model.input.batch_size):
            target=[]
            prediction=[]
            for j in range(model.input.num_steps-1):
                prediction.append([])
                target.append(int(midTargets[i][j]))
                tmp = list(pred_output[i * (model.input.num_steps-1) + j])
                for m in range(N):
                    index=tmp.index(max(tmp))
                    prediction[j].append(index)
                    tmp[index]=-100
            # print(target)
            # print(prediction)
            # print('\n',end='')
            target=json.dumps(target)
            # print(target)
            prediction=json.dumps(prediction)
            # print(prediction)
            f.write(target+'\n')
            f.write(prediction+'\n')
    f.close()
    return np.exp(costs / iters)

'''
f = open('data/testres.txt')
line=f.readlines()
lineNum=len(line)
for i in range(int(lineNum/2)):
    targets=json.loads(line[2*i])
    prediction=json.loads(line[2*i+1])
    print(targets)
    print(prediction)
'''

def SaveTestRes():
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")
    config = get_config()


    word_to_id = data_reader.get_word_to_id(FLAGS.data_path)

    raw_data = data_reader.raw_data(max_data_row=config.max_data_row,data_path=FLAGS.data_path, word_to_id=word_to_id, max_length=config.num_steps)
    train_data, test_data,voc_size, end_id, _, START_MARK, END_MARK, PAD_MARK = raw_data
    id_to_word = data_reader.reverseDic(word_to_id)

    config = get_config()
    global num_steps
    num_steps = config.num_steps-1

    eval_config = get_config()

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Test"):
            test_input = LSTMInput(config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                mtest = LSTMModel(is_training=False, config=eval_config, input_=test_input,
                                 START_MARK=START_MARK, END_MARK=END_MARK, PAD_MARK=PAD_MARK)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            test_perplexity = new_run_epoch(session, mtest, id_to_word=id_to_word,end_id=end_id,save_path='../data/twolayer/testres/100with1000row/ptb.txt')
            print("Test Perplexity: %.3f" % test_perplexity)

SaveTestRes()
