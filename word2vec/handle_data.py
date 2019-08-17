import jieba
from tqdm import tqdm
from zhon.hanzi import punctuation
import re

ori_data_dir = '/../data/'
out_dir = './handled_data/'
def st_file(stop_words_file):
    stop_words_voc = []
    
    with open(stop_words_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n ')
            stop_words_voc.append(line)
    return stop_words_voc

def handle_data(data_file):
    data = []
    
    with open(data_file, 'r') as f:
        for line in tqdm(f.readlines()):
            line = line.strip('\n ')
            line = line.split('_!_')
            
            line = line[3] + line[4]
            #print(line)
            #print(punctuation)
            line = re.sub('[{}]+'.format(punctuation), '', line)
            rule =re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
            line =rule.sub('',line)
            line = line.replace(' ', '')
            #print(line)
            #print(type(line))
            #exit()
            data.append(line)
    return data


def seg_data(data_list, stop_list):
    a = []
    
    for i in tqdm(data_list):
        sentence_seg = jieba.cut(i)
        sentence_seg = ' '.join(sentence_seg)
        words = sentence_seg.split(' ')
        #print(type(words))
        #print(words)
        line_clean = ''
        for word in words:
            if word in stop_list:
                continue
            line_clean = line_clean + word + ' '
        #print(line_clean)
        #exit()
        a.append(line_clean)
    return a
    


if __name__ == '__main__':
    data_file = ori_data_dir + 'labeled_toutiao_cat_data.txt'
    stop_words_file = ori_data_dir + 'hgd_stop_words.txt'
    stop_words_voc = st_file(stop_words_file)
    data = handle_data(data_file)
    seg_pure_data = seg_data(data, stop_words_voc)

    with open(out_dir + 'seg_pure_data.txt', 'w') as f:
        for i in seg_pure_data:
            f.write(i + '\n')
