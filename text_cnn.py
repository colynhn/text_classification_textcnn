import sys
import tensorflow as tf
import gensim
from gensim.models import word2vec
from tqdm import tqdm
import numpy as np
import os
import jieba
import json
from collections import Counter
import time
import datetime
#from zhon.hanzi import punctuation
import re

data_dir = './data/'
model_dir = './save_model/'
class TrainModel(object):
    
    learning_rate = 0.001
    training_epoches = 10
    evaluate_every = 100

class ModelConfig(object):
    
    #vec_size = 10  
    dropout_keep_prob = 0.5
    filter_sizes = [2, 3, 4, 5]
    embedding_size = 100
    num_filters = 128
    l2_lamda = 0.0

class Config(object):
    num_class = 14
    sentence_mean_length = 13
    batch_size = 128
    if sys.argv[1] == '0':
        data_file = data_dir + 'labeled_small_toutiao.txt'
        write_file = data_dir + 'small_new_data.txt'
    elif sys.argv[1] == '1':
        data_file = data_dir + 'labeled_toutiao_cat_data.txt'
        write_file = data_dir + 'new_data.txt'
    stop_words_file = data_dir + 'hgd_stop_words.txt'
    model_file = model_dir + 'word2vec.model'
    #train_dev_rate = 0.8
    
    training = TrainModel()
    model = ModelConfig()


class DataSet(object):
    def __init__(self,config):
        self.config = config
        self._data_file = config.data_file
        self._write_file = config.write_file
        self._stop_words_file = config.stop_words_file
        self._model_file = config.model_file
        #self._train_dev_rate = config.train_dev_rate
        self._batch_size = config.batch_size
        self._sentence_mean_length = config.sentence_mean_length
        self._embedding_size = config.model.embedding_size
        #self._dropout_keep_prob = config.model.dropout_keep_prob
        self._label = ['故事','文化','娱乐','体育','财经','房产','汽车','教育','科技','军事','旅游','国际','股票','游戏']
        self._all_labels = []
        #格式：['我 爱 中国','我 是 中国 人']
        self._content = []   
        #格式：['我','爱','中国','是']
        self._all_words_content = []
        self._stop_words_voc = []
        self._word_embedding = []
        self._train_data = [] 
        self._train_labels = []
        self._dev_data = []
        self._dev_labels = []
    def _read_file(self,file):
        c = []
        for i in range(15):
            c.append(0)
        #content = []
        #all_words_content = []
        with tf.device('/gpu:0'):
            with open(file,'r') as f:
                for line in tqdm(f.readlines()):
                    line = line.split('_!_')
                    #print(line)
                    #exit()
                    if line[2] == 'news_story':
                        line[2] = '故事'
                        c[0] += 1
                        if c[0] > 200000:
                            continue
                    elif line[2] == 'news_culture':
                        line[2] = '文化'
                        c[1] += 1
                        if c[1] > 200000:
                            continue
                    elif line[2] == 'news_sports':
                        line[2] = '体育'
                        c[2] += 1
                        if c[2] > 200000:
                            continue
                    elif line[2] == 'news_finance':
                        line[2] = '财经'
                        c[3] += 1
                        if c[3] > 200000:
                            continue
                    elif line[2] == 'news_house':
                        line[2] = '房产'
                        c[4] += 1
                        if c[4] > 200000:
                            continue
                    elif line[2] == 'news_car':
                        line[2] = '汽车'
                        c[5] += 1
                        if c[5] > 200000:
                            continue
                    elif line[2] == 'news_edu':
                        line[2] = '教育'
                        c[6] += 1
                        if c[6] > 200000:
                            continue
                    elif line[2] == 'news_tech':
                        line[2] = '科技'
                        c[7] += 1
                        if c[7] > 200000:
                            continue
                    elif line[2] == 'news_military':
                        line[2] = '军事'
                        c[8] += 1
                        if c[8] > 200000:
                            continue
                    elif line[2] == 'news_travel':
                        line[2] = '旅游'
                        c[9] += 1
                        if c[9] > 200000:
                            continue
                    elif line[2] == 'news_world':
                        line[2] = '国际'
                        c[10] += 1
                        if c[10] > 200000:
                            continue
                    elif line[2] == 'stock':
                        line[2] = '股票'
                        c[11] += 1
                        if c[11] > 200000:
                            continue
                    elif line[2] == 'news_agriculture':
                        line[2] = '三农'
                        c[12] += 1
                        if c[12] > 200000:
                            continue
                    elif line[2] == 'news_game':
                        line[2] = '游戏'
                        c[13] += 1
                        if c[13] > 200000:
                            continue
                    elif line[2] == 'news_entertainment':
                        line[2] = '娱乐'
                        c[14] += 1
                        if c[14] > 200000:
                            continue
                    if line[2] == '三农':
                        continue
             
                    self._all_labels.append(line[2])
                    pure_content = line[3].strip('\n ') + line[4].strip('\n ')
                    #print(pure_content)
                    line2 = line[2] + '\t' + line[3] + line[4]
                    #print(line)
                    """
                    if sys.argv[1] == 0:
                        write_file = date_dir + 'small_new_data.txt'
                    elif sys.argv[1] == 1:
                        write_file = data_dir + 'new_data.txt'
                    """
                    with open(self._write_file,'a+') as wf:
                        wf.write(line2)
                    #pure_content = re.sub('[{}]+'.format(punctuation), '', pure_content)
                    rule =re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
                    pure_content =rule.sub('', pure_content)
                    pure_content = pure_content.replace(' ', '')
                    all_cut_content = jieba.cut(pure_content)
                    ww = ' '.join(all_cut_content)
                    self._content.append(ww)
                    ww = ww.strip().split(' ')
                    line_clean = []
                    ww = [w for w in ww if w not in self._stop_words_voc]
                    for word in ww:
                        self._all_words_content.append(word)
   
    def _label_to_idx(self, label2idx):
        label_ids = [label2idx[label] for label in self._all_labels]

        return label_ids
        

    def _read_stop_words_file(self,file):
        with open(file,'r') as sf:
            for line in sf.readlines():
                self._stop_words_voc.append(line.strip('\n'))
   
    
    def _gen_voc(self, all_words_content):
        word_count = Counter(all_words_content) 
        sort_word_count = sorted(word_count.items(), key=lambda x: x[1],reverse=True)
        words = [i[0] for i in sort_word_count if i[1] >= 1]
        vocab, word_embedding = self._get_word_embedding(words)
        # 每一个词语对应一个向量
        self._word_embedding = word_embedding
        
        word2idx = dict(zip(vocab,list(range(len(vocab)))))               
        
        label2idx = dict(zip(self._label,list(range(len(self._label)))))
        return label2idx, word2idx

    def _get_word_embedding(self, words):
        
        word_vec = word2vec.Word2Vec.load( model_dir + 'word2vec.model')
        vocab = []
        word_embedding = []
        
        vocab.append('PAD')
        vocab.append('UNK')
        word_embedding.append(np.zeros(self._embedding_size))
        word_embedding.append(np.random.randn(self._embedding_size))
        for w in words:
            try:
                vec = word_vec[w]
                vocab.append(w)
                word_embedding.append(vec)
            except:
                print(w + "不存在于词向量中")


        return vocab, np.array(word_embedding)
    
    
   
    def _word_to_idx(self, word2idx):
        content_ids = [[word2idx.get(item,word2idx['UNK']) for item in line] for line in self._content]
        
        return content_ids
        
    
    def _get_train_dev_data(self, label_ids, content_ids, word2idx):
        ########  各类别都在一块聚集，所以应该分散处理，取出数据集：train:test = 9:1
        content1 = []
        train_content = []
        dev_content = []
        train_label = []
        dev_label = []
        for c in content_ids:
            if len(c) >= self._sentence_mean_length:
                content1.append(c[:self._sentence_mean_length])
            else:
                content1.append(c + [word2idx["PAD"]] * (self._sentence_mean_length - len(c)))
        #sep_idx = int(len(content1) * self._train_dev_rate)
        
        for i,w in enumerate(content1):
            if i % 10 == 0:
                dev_content.append(w)
            else:
                train_content.append(w)
        for j,c in enumerate(label_ids):
            if j % 10 == 0:
                dev_label.append(c)
            else:
                train_label.append(c)

        train_data = np.asarray(train_content, dtype="int64")
        train_labels = np.array(train_label, dtype = 'float32')
        dev_data = np.asarray(dev_content, dtype="int64")
        dev_labels = np.array(dev_label,dtype = 'float32')
        
        return train_data, train_labels, dev_data, dev_labels

    def data_generate(self):
        self._read_stop_words_file(self._stop_words_file)
        self._read_file(self._data_file)
        label2idx, word2idx = self._gen_voc(self._all_words_content)
        label_ids = self._label_to_idx(label2idx)
        content_ids = self._word_to_idx(word2idx)
        
        train_data, train_labels, dev_data, dev_labels = self._get_train_dev_data(label_ids, content_ids, word2idx)
        self._train_data = train_data
        self._dev_data = dev_data
        self._train_labels = train_labels
        self._dev_labels = dev_labels
        #print(self._train_data)
        #print(self._dev_data)        

#----------------------------------------------构建batch数据集---------------------------------------------------------------#

def next_batch(x, y, batch_size):

    sh = np.arange(len(x))
    np.random.shuffle(sh)
    
    #exit()
    x = x[sh]
    y = y[sh]
    #print(len(y))
    #exit()
    batch_num = (len(x) // batch_size) + 1

    for i in range(batch_num):
        start = i * batch_size
        end = start + batch_size

        batch_X = np.array(x[start:end],dtype = 'int64')
        batch_Y = np.array(y[start:end], dtype = 'float32')
        
        yield batch_X, batch_Y

#---------------------------------------------构建textCNN模型----------------------------------------------------------------#
class TextCNNModel(object):
    
    def __init__(self, config, word_embbeding):
        
        self.input_X = tf.placeholder(tf.int32, [None,config.sentence_mean_length], name = 'input_X')
        self.input_Y = tf.placeholder(tf.int32, [None], name = 'input_Y')
        
        self._dropout_keep_prob = tf.placeholder(tf.float32,name = 'dropout_keep_prob')

        l2_loss = tf.constant(0.0)

        with tf.device('/gpu:2'),tf.name_scope('embedding'):
            #[]
            #print(word_embbeding)
            self.W = tf.Variable(tf.cast(word_embedding, dtype = tf.float32, name = 'word2vec'), name = 'W')
           
            #exit()
            #[None, sequence_length, embedding_size]
            self.embedding_words = tf.nn.embedding_lookup(self.W, self.input_X)

            self.embeddiing_words_expanded = tf.expand_dims(self.embedding_words, -1)
        
        pool_outputs = []
        for i,filter_size in enumerate(config.model.filter_sizes):
            
            with tf.name_scope("conv-maxpool-%s" % filter_size):

                filter_shape = [filter_size, config.model.embedding_size, 1, config.model.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name = 'W')
                b = tf.Variable(tf.constant(0.1, shape = [config.model.num_filters]), name = 'b')
                conv = tf.nn.conv2d(self.embeddiing_words_expanded, W, strides = [1,1,1,1], padding = 'VALID', name = 'conv')
                
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name = 'relu')
                #过完卷积层的size：(n-f+2d)/s + 1 向下取整
                pooled = tf.nn.max_pool(
                    h, 
                    ksize = [1, config.sentence_mean_length - filter_size + 1, 1, 1], 
                    strides = [1,1,1,1],
                    padding = 'VALID',
                    name = 'pool')

                pool_outputs.append(pooled)
            
            num_filter_total = config.model.num_filters * len(config.model.filter_sizes)
            self.h_pool = tf.concat(pool_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filter_total])

            with tf.name_scope('dropout'):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self._dropout_keep_prob)
            #tf.reset_default_graph()
            #with tf.name_scope('output'):
            with tf.variable_scope(name_or_scope='output', reuse=tf.AUTO_REUSE): 
                #tf.reset_default_graph()
                output_W = tf.get_variable(
                    name='output_W', 
                    shape=[num_filter_total, config.num_class], 
                    initializer = tf.contrib.layers.xavier_initializer())
                output_b = tf.Variable(tf.constant(0.1, shape = [config.num_class]), name='output_b')
                l2_loss += tf.nn.l2_loss(output_W)
                l2_loss += tf.nn.l2_loss(output_b)
                self.logits = tf.nn.xw_plus_b(self.h_drop, output_W, output_b, name = 'logits')
                self.predictions = tf.argmax(self.logits, axis = -1, name = 'predictions')

                print(self.predictions)

            with tf.name_scope('loss'):
                print(self.input_Y)
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits, labels= self.input_Y)
                self.loss = tf.reduce_mean(losses) + config.model.l2_lamda * l2_loss


#----------------------------------------------评估指标：混淆矩阵------------------------------------------------------------#
"""
首先计算二分类，而多分类就是二分类的组合
"""
def mean(item:list) -> float:
    
    m = sum(item) / len(item) if len(item) > 0 else 0

    return m



def accuracy(pred_y, true_y):

    if isinstance(pred_y[0],list):
        pred_y = [item[0] for item in pred_y]

    correct = 0
    for i in range(len(pred_y)):
        if pred_y[i] == true_y[i]:
            correct +=1
    acc = correct / len(pred_y) if len(pred_y) > 0 else 0 

    return acc

def binary_precision(pred_y, true_y, positive = 1):
    corr = 0
    pred_corr = 0

    for i in range(len(pred_y)):
        if pred_y[i] == positive:
            pred_corr += 1
            if pred_y[i] == true_y[i]:
                corr += 1
    acc = corr / pred_corr if pred_corr > 0 else 0

    return acc
def binary_recall(pred_y, true_y, positive = 1):
    
    corr = 0
    alread_has_corr = 0
    for i in range(len(pred_y)):
        if true_y[i] == positive:
            alread_has_corr+= 1
            if pred_y[i] == true_y[i]:
                corr += 1

    acc = corr / alread_has_corr if alread_has_corr > 0 else 0

    return acc


def binary_f_beta(pred_y, true_y, beta = 1.0, positive = 1):
    precision = binary_precision(pred_y, true_y, positive)
    recall = binary_recall(pred_y, true_y, positive)
    
    try :
        f_beta = (1+beta*beta) * precision * recall / (beta*beta * (precision + recall))
    except:
        f_beta = 0
    return f_beta
def multi_precision(pred_y, true_y, labels):
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    precisions = [binary_precision(pred_y, true_y, label) for label in labels]
    precision = mean(precisions)

    return precision
def multi_recall(pred_y, true_y, labels):

    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]
    recalls = [binary_recall(pred_y, true_y, label) for label in labels]

    recall = mean(recalls)

    return recall


def multi_f_beta(pred_y, true_y, labels, beta = 1.0):
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    f_betas = [binary_f_beta(pred_y, true_y, beta, label) for label in labels]

    f_beta = mean(f_betas)

    return f_beta


def get_multi_metrics(pred_y, true_y, labels, beta = 1.0):
    
    acc = accuracy(pred_y, true_y)
    precision = multi_precision(pred_y, true_y, labels)
    recall = multi_recall(pred_y, true_y, labels)
    f_beta = multi_f_beta(pred_y, true_y, labels, beta)

    return acc, precision, recall, f_beta

#----------------------------------------------模型训练---------------------------------------------------------------------#
if __name__ == '__main__':
    config = Config()
    data = DataSet(config)
    data.data_generate()

    train_data = data._train_data
    train_labels = data._train_labels
    print("训练数据大小:" + str(len(train_data)))
    print("训练标签大小:" + str(len(train_labels)))
    dev_data = data._dev_data
    dev_labels = data._dev_labels

    word_embedding = data._word_embedding
    
    label_list = list(range(len(data._label)))
    
    #tf.reset_default_graph()
    graph1 = tf.Graph()
    with graph1.as_default():
        #tf.reset_default_graph()    
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth=True
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9 
        sess = tf.Session(config=session_conf)
        #exit()
        with sess.as_default():
            cnn = TextCNNModel(config, word_embedding)
            # TensorFlow handle the counting of training steps
            my_global_step = tf.Variable(0, name = 'globalstep', trainable=False)
            optimizer = tf.train.AdamOptimizer(config.training.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step = my_global_step)
      

            for g, v in grads_and_vars:
                if g is not None:
                    tf.summary.histogram('{}/grad/his'.format(v.name), g)
                    tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))            
            print("Writing to {}\n".format(out_dir))
        
            lossSummary = tf.summary.scalar("loss", cnn.loss)
            summaryOp = tf.summary.merge_all()
                        
            trainSummaryDir = os.path.join(out_dir, "train")
            trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)
                                                
            evalSummaryDir = os.path.join(out_dir, "eval")
            evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)
            sess.run(tf.global_variables_initializer())
       
       
            def train_step(batch_X, batch_Y):
                feed_dict = {
                    cnn.input_X: batch_X,
                    cnn.input_Y:batch_Y,
                    cnn._dropout_keep_prob: config.model.dropout_keep_prob
                }

                _, summary, step, loss, predictions  = sess.run([train_op, summaryOp, my_global_step, cnn.loss, cnn.predictions],feed_dict)

                time_str = datetime.datetime.now().isoformat()
            
                acc, precision, recall, f_beta = get_multi_metrics(pred_y = predictions, true_y = batch_Y, labels = label_list)
                trainSummaryWriter.add_summary(summary, step)

                return loss, acc, precision, recall, f_beta

            def dev_step(batch_X, batch_Y):
                feed_dict = {
                    cnn.input_X: batch_X,
                    cnn.input_Y:batch_Y,
                    cnn._dropout_keep_prob: config.model.dropout_keep_prob
                }

                summary, step, loss, predictions  = sess.run([summaryOp, my_global_step, cnn.loss, cnn.predictions],feed_dict)
                acc, precision, recall, f_beta = get_multi_metrics(pred_y = predictions, true_y = batch_Y, labels = label_list)
                evalSummaryWriter.add_summary(summary, step)

                return loss, acc, precision, recall, f_beta


            for i in range(config.training.training_epoches):
                print('=' * 20+ 'training epoch :' +str(i) +'=' * 20)
            
                for batch_train in next_batch(train_data, train_labels , config.batch_size):
                    loss, acc, precision, recall, f_beta = train_step(batch_train[0], batch_train[1])
                    current_step = tf.train.global_step(sess, my_global_step)
                
                    print("train: step:{}, loss: {}, acc: {}, precision: {}, recall: {}, f_beta: {}".format(current_step, loss, acc, precision, recall, f_beta)) 

                    if current_step % config.training.evaluate_every == 0:
                        print("\nEvaluation:")

                        losses = []
                        accs = []
                        precisions = []
                        recalls = []
                        f_betas = []


                        for batch_eval in next_batch(dev_data, dev_labels, config.batch_size):
                        
                            loss, acc, precision, recall, f_beta = dev_step(batch_eval[0], batch_eval[1])
                            losses.append(loss)
                            accs.append(acc)
                            precisions.append(precision)
                            recalls.append(recall)
                            f_betas.append(f_beta)
                        time_str = datetime.datetime.now().isoformat()
                    
                        print("{}, step:{}, loss: {}, acc: {}, precision: {}, recall: {}, f_beta: {}".format(time_str,current_step, mean(losses), mean(accs), mean(precisions), mean(recalls), mean(f_betas)))       
