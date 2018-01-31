# -*- coding: utf-8 -*-
import tensorflow as tf
import os
from model.SLSTM_b import *
from tools import data_reader
import time
from datetime import timedelta
MODE='train'
flags=tf.flags
flags.DEFINE_string('data_path','../data','data path')
flags.DEFINE_string('save_path','../data/checkpoints/stacklstm/best_validation','best val save path')
flags.DEFINE_string('save_dir','../data/checkpoints/stacklstm','save dir')
flags.DEFINE_string('tensorboard_dir','../data/tensorboard/stacklstm','tensorboard path')
flags.DEFINE_string('MODE','train','mode')
flags=flags.FLAGS

def get_time_dif(start_time):
    end_time=time.time()
    time_dif=end_time-start_time
    return timedelta(seconds=int(round(time_dif)))

def feed_data(x_batch,y_batch,keep_prob,total_loss=None):
    feed_dict={
        model.input:x_batch,
        model.label:y_batch,
        model.keep_prob:keep_prob,
        model.total_loss:total_loss
    }
    return feed_dict

def evaluate(sess,x_,y_):
    data_len=len(x_)
    eval_batch=data_reader.batch_iter(x_,y_,config.batch_size)
    total_loss=0.0
    total_acc=0.0
    for x_batch,y_batch in eval_batch:
        batch_len=len(x_batch)
        feed_dict=feed_data(x_batch,y_batch,1.0)
        loss,acc=sess.run([model.loss,model.acc],feed_dict=feed_dict)
        total_loss+=loss*batch_len
        total_acc+=acc*batch_len
    return total_loss/data_len,total_acc/data_len

def train(id_to_word):
    tensorboard_dir=flags.tensorboard_dir
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tf.summary.scalar('loss',model.loss)
    tf.summary.scalar('accuracy',model.acc)
    merged_summary=tf.summary.merge_all()
    writer=tf.summary.FileWriter(tensorboard_dir)

    saver=tf.train.Saver()
    if not os.path.exists(flags.save_dir):
        os.makedirs(flags.save_dir)
    Config = tf.ConfigProto()
    Config.gpu_options.allow_growth = True
    with tf.Session(config=Config) as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        print('training and evaluating...')
        start_time=time.time()
        total_batch=0
        best_eval_acc=0.0
        last_improved=0
        required_improvement=1000 #超过1000轮未提升则提前结束训练
        flag=False
        for epoch in range(config.num_epochs):
            print('Epoch:',epoch+1)
            total_loss=0.0
            train_batch=data_reader.batch_iter(train_inputs,train_labels,config.batch_size)
            for x_batch,y_batch in train_batch:
                print(total_loss)
                feed_dict=feed_data(x_batch,y_batch,config.dropout_keep_prob,total_loss)
                total_loss=sess.run(model.loss,feed_dict=feed_dict)

                if total_batch%config.save_per_batch==0:
                    s=sess.run(merged_summary,feed_dict=feed_dict)
                    writer.add_summary(s,total_batch)
                if total_batch%config.print_per_batch==0:
                    loss,acc,prediction=sess.run([model.loss,model.acc,model.predict_class],feed_dict=feed_dict)
                    loss_val,acc_val=evaluate(sess,val_inputs,val_labels)
                    if acc_val>best_eval_acc:
                        best_eval_acc=acc_val
                        last_improved=total_batch
                        saver.save(sess,flags.save_path)
                        improved_str='*'
                    else:
                        improved_str=''
                    time_dif=get_time_dif(start_time)
                    msg='Iter:{0:>6}, Train Loss:{1:>6.2}, Train Acc:{2:>7.2%},' \
                        'Val Loss:{3:>6.2}, Val Acc:{4:>7.2%}, Time:{5} {6}'
                    print(msg.format(total_batch,loss,acc,loss_val,acc_val,time_dif,improved_str))


                #---------------
                if total_batch%64==0:
                    sess.run(model._train_op,feed_dict=feed_dict)
                    total_loss=0.0
                #----------------

                # sess.run(model._train_op,feed_dict=feed_dict)
                # sess.run(model.total_loss,)

                total_batch+=1

            #     if total_batch-last_improved>required_improvement:
            #         print('No optimization dor a long time, stop training...')
            #         flag=True
            #         break
            loss_val,acc_val=evaluate(sess,val_inputs,val_labels)
            if acc_val>best_eval_acc:
                best_eval_acc=acc_val
                last_improved=total_batch
                saver.save(sess,flags.save_path)
                improved_str='*'
            else:
                improved_str=''
            time_dif=get_time_dif(start_time)
            msg='Val Loss:{3:>6.2}, Val Acc:{4:>7.2%}, Time:{5} {6}'
            print(msg.format(loss_val,acc_val,time_dif,improved_str))
            # for i in range(config.batch_size):
            #     input_=''
            #     target_=''
            #     predic_=''
            #     for j in range(config.num_steps-1):
            #         input_+=id_to_word[x_batch[i][j]]+' '
            #         target_+=id_to_word[y_batch[i][j]]+' '
            #         predic_+=id_to_word[prediction[i][j]]+' '
            #     print('input:   %s\n\nTarget:   %s\n\nPrediction: %s \n\n'%(input_,target_,predic_))

            # if flag:
            #     break
def test():
    pass

if __name__ == '__main__':
    config=LSTMConfig()
    word_to_id=data_reader._build_vocab(os.path.join(flags.data_path,"pro_cla/train.txt"),config.vocab_size,config.num_steps)

    train_inputs,train_labels,val_inputs,val_labels,test_inputs,test_labels,left_id, right_id, PAD_ID=data_reader.raw_data(data_path=flags.data_path,word_to_id=word_to_id,num_steps=config.num_steps,num_classes=config.num_classes)

    model=StackLSTM(config,left_id,right_id)
    id_to_word=data_reader.reverseDic(word_to_id)
    if MODE=='train':
        train(id_to_word)
    else:
        test()
