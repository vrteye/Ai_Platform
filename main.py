import jieba
import pymysql
import Levenshtein
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer  # 词频计数
from sklearn.feature_extraction.text import TfidfVectorizer  # tf-idf 模块
from sklearn.model_selection import train_test_split  # 分割数据集


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
# print(area_data)

house_data['labelnum'] = 1  # 房屋类型
time_data['labelnum'] = 2  # 预计装修时间
area_data['labelnum'] = 3  # 面积

# [1]:查看样本空间
all_data = pd.concat([house_data, time_data, area_data])
# print("all_data=", all_data)
jieba.load_userdict(r'C:\Users\Administrator\Desktop\自定义词典.txt')

# [2]:生成词袋
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
                # outstr = outstr.replace(" ", "|")
    corpus.append(outstr)

# [3]:计数统计，tf-idf模块
vectorizer = CountVectorizer()
corpusTotoken_count = vectorizer.fit_transform(corpus).todense()
vectorizer = TfidfVectorizer()
corpusTotoken_tfidf = vectorizer.fit_transform(corpus).todense()

X_data = np.array(corpusTotoken_count)
Y_data = np.array(all_data['labelnum'])

# [4]:训练阶段
x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3)

# LR 预测
LR = LogisticRegression()
LR.fit(x_train, y_train)
predictions_LR = LR.predict(x_test)
prob_LR = LR.predict_proba(x_test)
# print('LogisticRegression:', prob_LR)

# 模型评估
LR_y_predicted = LR.predict(x_train)
# print(metrics.classification_report(y_train, LR_y_predicted))

LR_predicted = np.mean(LR_y_predicted == y_train)
print('LR准确率：', LR_predicted)

# Bernoulli bayes 预测
GB = MultinomialNB()
GB.fit(x_train, y_train)
predictions_GB = GB.predict(x_test)
prob_GB = GB.predict_proba(x_test)
# print('predictions_GB:', predictions_GB)


# 模型评估
GB_y_predicted = GB.predict(x_train)
# print(metrics.classification_report(y_train, GB_y_predicted))

GB_predicted = np.mean(GB_y_predicted == y_train)
print('bayes准确率：', GB_predicted)

# print('MultinomialNB:', prob_GB)

# RandomForest预测
RF = RandomForestClassifier()
RF.fit(x_train, y_train)
predictions_RF = RF.predict(x_test)
prob_RF = RF.predict_proba(x_test)
# print('RandomForestClassifier:', prob_RF)

# 模型评估
RF_y_predicted = RF.predict(x_train)
# print(metrics.classification_report(y_train, GB_y_predicted))

RF_predicted = np.mean(RF_y_predicted == y_train)
print('RandomForest准确率：', RF_predicted)
# SVM预测
SV = SVC(kernel='linear', probability=True)
SV.fit(x_train, y_train)
predictions_SV = SV.predict(x_test)
prob_SV = SV.predict_proba(x_test)

# 模型评估
SV_y_predicted = SV.predict(x_train)
# print(metrics.classification_report(y_train, GB_y_predicted))

SV_predicted = np.mean(SV_y_predicted == y_train)
print('SVM准确率：', SV_predicted)

dic = {"1": "房屋类型", "2": "预计装修时间", "3": "面积"}

# [5]:测试阶段
while 1:
    sent = input("客户问：")
    link = pymysql.Connect(host='localhost', port=3306, user='root', password='111111', db='xmjjdata', charset='utf8')
    cursorde = link.cursor()
    sent_cut = jieba.lcut(sent)
    sent_cut_input = ' '.join(sent_cut)
    print("切词结果：", sent_cut_input)
    Xpredict = vectorizer.transform([sent_cut_input]).todense()
    print("预测的准确率=", GB.predict_proba(Xpredict))
    prediction = dic[str(GB.predict(Xpredict)[0])]
    print('预测标签：', prediction)
    # cx_sql = 'select answer from ask_answer where lable=%s ORDER BY RAND() LIMIT %s'  # 随机抽取一个
    cx_sql = 'select ask,answer from ask_answer where lable=%s'
    cursorde.execute(cx_sql, prediction)
    all = cursorde.fetchall()

    dict_all = dict(all)
    Leven = []
    for key in dict_all:
        Ratio = Levenshtein.ratio(sent, key)  # 编辑距离相似度
        Leven.append(Ratio)
    Max_Index = Leven.index(max(Leven))  # 相似度最高句子的索引
    value = list(dict_all.values())  # 索引对应的值
    Answer = value[Max_Index]
    print('答：', Answer)
