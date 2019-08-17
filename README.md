1、数据集：
   
   数据集来源：今日头条中文新闻（文本）分类数据集

   数据集说明参考：https://github.com/fate233/toutiao-text-classfication-dataset

2、数据集处理：

   说明：14分类，完整数据集每类20w数据
    
   输入原始数据集：labeled_toutiao_cat_data.txt 
   
   代码：word2vec/handle_data.py 
   
   生成数据 handled_data/seg_pure_data.txt 

3、训练词向量：

  输入处理后数据：seg_pure_data.txt
  
  代码：word2vec/word2vec_model.py
  
  训练后model：save_model/word2vec.model    saved_model/word2vec.txt
 
4、textCNN训练：
  
  小数据集：python text_cnn.py 0  
  完整数据集：python text_cnn.py 1
  
5、结果：

   train: loss: 0.5536235570907593, acc: 0.8205128205128205, precision: 0.7136054421768707, recall: 0.6882456828885399, f_beta: 0.6967303093486196

  
  （1）数据存在分类模棱两可的情况：即分到哪一类都太合适的情况
  
  （2）数据采用char级别训练可能效果更好
  
  （3）其他，bulabulabula 待研究

参考：
   
   http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

   https://www.cnblogs.com/jiangxinyang/p/10207273.html 
     
   https://hunto.github.io/nlp/2018/03/29/TextCNN%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E8%AF%A6%E8%A7%A3.html
     
     
   shout to 以上大佬s
          
                                                                           peace & love
   
                                  
 
