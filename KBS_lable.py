import time
import jieba
import pymysql
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer  # 词频计数
from sklearn.feature_extraction.text import TfidfVectorizer  # tf-idf 模块
from sklearn.model_selection import train_test_split  # 分割数据集

"""
功能：基于朴素贝叶斯算法建立机器学习模型对MySQL数据库答句智能标签，续接打标签
xm表：    id ask answer run    run默认为0，打上标签后为1
lable表：     id Ckey Ceid
"""


def Bayes_Model(id, sent):
    def stopwordslist():
        stopwords = [line.strip() for line in open(r'E:\xm\stop_words', encoding='UTF-8').readlines()]
        return stopwords

    def get_house_time_data():
        outputfile1 = r'C:\Users\Administrator\Desktop\samples\house_all.txt'
        outputfile2 = r'C:\Users\Administrator\Desktop\samples\time_all.txt'
        outputfile3 = r'C:\Users\Administrator\Desktop\samples\area_all.txt'
        return outputfile1, outputfile2, outputfile3

    house_outfile, time_outfile, area_outfile = get_house_time_data()
    house_data = pd.read_table(house_outfile, names=['label', 'chat'])
    time_data = pd.read_table(time_outfile, names=['label', 'chat'])
    area_data = pd.read_table(area_outfile, names=['label', 'chat'])
    house_data['labelnum'] = 1  # 房屋类型
    time_data['labelnum'] = 2  # 预计装修时间
    area_data['labelnum'] = 3  # 面积
    all_data = pd.concat([house_data, time_data, area_data])
    jieba.load_userdict(r'C:\Users\Administrator\Desktop\自定义词典.txt')
    stopwords = stopwordslist()
    corpus = []
    for i in range(len(all_data)):
        cor = jieba.lcut(all_data.iloc[i, 1])
        outstr = ''
        for j in cor:
            if j not in stopwords:
                if j != '\t':
                    outstr += j
                    outstr += " "
        corpus.append(outstr)
    vectorizer = CountVectorizer()
    corpusTotoken_count = vectorizer.fit_transform(corpus).todense()
    vectorizer = TfidfVectorizer()
    corpusTotoken_tfidf = vectorizer.fit_transform(corpus).todense()
    X_data = np.array(corpusTotoken_count)
    Y_data = np.array(all_data['labelnum'])
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3)
    GB = MultinomialNB()
    GB.fit(x_train, y_train)
    GB_y_predicted = GB.predict(x_train)
    GB_predicted = np.mean(GB_y_predicted == y_train)
    dic = {"1": "房屋类型", "2": "预计装修时间", "3": "面积"}
    sent_cut = jieba.lcut(sent)
    sent_cut_input = ' '.join(sent_cut)
    Xpredict = vectorizer.transform([sent_cut_input]).todense()
    prediction = dic[str(GB.predict(Xpredict)[0])]
    # for item in key_dic:
    tupe = (prediction, id)
    add_list = list(tupe)
    # print('预测标签：', prediction)
    return add_list


link = pymysql.Connect(host='localhost', port=3306, user='root', password='111111', db='xmjjdata', charset='utf8')
cursor = link.cursor()


# lable插入数据库
def batch_insert(list1):
    try:
        print(list1)
        sql_insert = "insert into lable(Ckey,Ceid) VALUES ('%s','%s')" % (list1[0], list1[1])
        cursor.execute(sql_insert)  # 执行插入
        cursor.execute('commit')

    except pymysql.Error as e:
        print('插入数据报错！', str(e))


try:
    sql_query1 = 'select * from xm where IFNULL(run,0)=0'
    cursor.execute(sql_query1)
    values = cursor.fetchall()  # 符合条件的所有数据，全部赋值给values
    res_list = []
    index = 0
    for i in values:
        if len(i[1]) <= 100:
            temp_list = Bayes_Model(i[0], i[1])
            batch_insert(temp_list)
            res_list.clear()
            upt_sql = 'update xm set run=1 where id=' + str(i[0])
            cursor.execute(upt_sql)  # 执行插入
            cursor.execute('commit')
        index = index + 1
        print('AI Runing ...' + str(index))
        time.sleep(0.1)
    print('ok')
except pymysql.Error as e:
    print('查询数据报错！', str(e))
