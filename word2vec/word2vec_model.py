import logging
import gensim
from gensim.models import word2vec


dir2 = './handled_data/'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#sentence = word2vec.LineSentence(dir + 'unlabeled_seg_pure_data.txt')
sentence = word2vec.LineSentence(dir2 + 'seg_pure_data.txt')

model = gensim.models.Word2Vec(sentence, sg=1, iter=8) # 词向量维度为默认:100
model.save('/../save_model/word2vec.model')

#model.wv.save_word2vec_format(outp2, binary=False)
model.wv.save_word2vec_format('/../save_model/word2vec.txt', binary=False)
