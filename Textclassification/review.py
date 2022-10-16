'''
任务描述：印尼语涉华新闻分类
'''
import pandas as pd
#读取文件
df_train=pd.read_json('ind_train.json',encoding='utf-8') #训练集 [1118 rows x 3 columns]
df_dev=pd.read_json('ind_dev.json',encoding='utf-8') #验证集 [280 rows x 3 columns]
df_test=pd.read_json('ind_test.json',encoding='utf-8') #测试机 [600 rows x 3 columns]

# print('****************************')
# print(df_dev)
# print('****************************')
# print(df_test)
# print('****************************')
# print(df_train)
# print('****************************')

# 查看一下一共有多少个标签是否涉华，每个标签的数目是多少
from collections import Counter
labels_t = Counter(df_train[1].tolist())
# print(df_train[1].tolist())
# print(labels_t)

# 得到训练数据
train_text = []  # 训练集新闻文本
train_labels = []  # 训练集新闻标签

for i in range(len(df_train)):
    x = list(df_train[0][i])
    #通过空格链接在一起
    text = ' '.join(x)
    train_text.append(text)
    train_labels.append(df_train[1][i])

# print(train_text)
# print('****************************')
# print(train_labels)
# print('****************************')
# print(list(df_train[0][1]))
# print(' '.join(list(df_train[0][1])))

dev_text = []  # 验证集新闻文本
dev_labels = []  # 验证集新闻标签
for i in range(len(df_dev)):
    x = list(df_dev[0][i])
    text = ' '.join(x)
    dev_text.append(text)
    dev_labels.append(df_dev[1][i])

test_text = []  # 测试集新闻文本
test_labels = []  # 测试集新闻标签
for i in range(len(df_test)):
    x = list(df_test[0][i])
    text = ' '.join(x)
    test_text.append(text)
    test_labels.append(df_test[1][i])



# 数据划分（没有单独的验证集，训练集划分出来）
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(train_text, train_labels, test_size=0.3, random_state=2)
# # train_text：待划分样本数据
# # train_labels：待划分样本数据的结果（标签）
# # test_size：测试数据占样本数据的比例，若整数则样本数量
# # random_state：设置随机数种子，填同一个数字的时候，每次得到的随机数组是一样的。若为0或不填，则每次得到数据都不一样


# 对文本进行分类  -->  将文本转换成向量，这里采用向量空间模型（VSM）
# 对于长文本，常用TfIdf做特征抽取


# 文本预处理
# 分词、特征降维（小写化、去停用词、按照特定的规则保留K个词）
import string
puns = string.punctuation
# 定义分词函数
def tokenizer(text):
    """
    给定一个文本text，返回词的序列words。
    定义这个函数的目的就是为了不直接采用空格分词，提升分词的精度。
    这里采用的算法是比较粗暴的，就是在标点符号前增加一个空格，然后直接按照空格切分。
    """
    for ch in puns:
        text = text.replace(ch, " "+ch+" ")
    return text.split()


# 停用词表 印尼语停用词
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
stopwords = StopWordRemoverFactory().get_stop_words()
# print(stopwords)

# 对文本进行特征抽取（将文本转成向量）
# 采用空间向量模型，使用TfIdf做特征抽取
# 在sklearn文本特征抽取工具sklearn.feature_extraction.text中有一个工具叫做TfidfVectorizer， 可以用作TfIdf特征抽取
from sklearn.feature_extraction.text import TfidfVectorizer
max_vocab = 30000   # 给定最大词数目为30000

# 新建一个特征抽取对象
vectorizer = TfidfVectorizer(
    lowercase=True,  # 是否将单词转为小写（如果为True的话就可以缩小特征空间）
    analyzer="word",  # 词级别分析（将文档在单词级别转成向量）
    tokenizer=tokenizer,  # 分词器，一个函数，给定一个文本返回词列表。
    stop_words=stopwords,  # 停用词表， 包含停用词的一个列表。
    max_features=max_vocab,  # 最大词数目，如果给定了最大词数目，则只会保留出现频率最高的max_features个词 如何在TfidfVectorizer模块中选择max_features参数的大小？
)

# 如果没有传进去tokenizer参数，sklearn会默认以正则表达式(?u)\b\w\w+\b对文本进行分词。具体含义见：https://blog.csdn.net/steven_ffd/article/details/84881063

# 利用训练文本构建特征空间
vectorizer.fit(train_text)

# 查看一下特征空间
# print(vectorizer.get_feature_names())
# 看看特征空间中有多少个词，共有all word number: 25341个，max_vocab = 30000   # 保留出现频率最高的max_features个词
# print("all word number:", len(vectorizer.get_feature_names()))

# 对文本进行编码（向量化）
train_sequences = vectorizer.transform(train_text)
dev_sequences = vectorizer.transform(dev_text)
print(type(train_sequences))
print(type(dev_sequences))



# 这个时候train_sequences就是一个矩阵，
# 查看train_sequences的大小
print(train_sequences.shape)


# 矩阵的行数代表有多少个训练样本
# 矩阵的列数代表特征空间中有多少个单词
# train_sequences[i][j] 代表的就是第i个训练样本对应的第j个特征空间中的词的tfidf值

# 构建分类器
# SVM、线性SVM
from sklearn.svm import SVC, LinearSVC
# 朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
# 决策树
from sklearn.tree import DecisionTreeClassifier
# 随机森林、GBDT
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 创建分类器对象，这里创建的是决策树分类器
clf = DecisionTreeClassifier()

# 需要用哪个模型，只需要替换这一行代码：clf = XXX 就可以了！！
# clf = GaussianNB()  # 贝叶斯
# clf = RandomForestClassifier()  # 随机森林
# clf = GradientBoostingClassifier()  # GBDT


# 分类器训练, 这里train_sequences是输入，train_labels是期望的输出。
clf.fit(train_sequences, train_labels)

# 至此，分类器已经训练好啦！！！
# 对验证集进行预测，predict的时候不需要给定输出，只需要给输入即可。
pred_labels = clf.predict(dev_sequences)    #看一下验证结果！
# print(pred_labels)



# 分类器的评估
# 准确率、精度、召回率、f1值
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 计算准确率,其中dev_labels是真实结果，pred_labels是分类器的预测结果， 返回浮点数
acc = accuracy_score(dev_labels, pred_labels)  # 准确率
print("accuracy:", acc)

p = precision_score(dev_labels, pred_labels)   # 精度
print("precision_score:", p)


# 更方便的方法: 使用分类报告集成以上所有评估指标
from sklearn.metrics import classification_report
# 直接使用classification_report， digits=4意思是保留小数点后4位
report = classification_report(dev_labels, pred_labels, digits=4)
print("==========  report  ==========")
print(report)




# 最后就模型的保存和加载啦！

# 我们这里有2个东西需要保存，分别是特征抽取模型 vectorizer 和分类模型 clf
# 我们使用pickle进行保存
import pickle
#
# 保存vectorizer
# 需要以二进制的方式打开文件，这时候不需要指定编码
with open("vectorizer_re.bin", "wb") as f:
    pickle.dump(vectorizer, f)

# 保存分类模型
with open("model_re.bin", "wb") as f:
    pickle.dump(clf, f)

# 加载模型
with open("vectorizer_re.bin", "rb") as f:
    vectorizer = pickle.load(f)
with open("model_re.bin", "rb") as f:
    clf = pickle.load(f)

# 测试一下使用有没有什么问题
train_sequences = vectorizer.transform(train_text)
test_sequences = vectorizer.transform(test_text)

# 预测
pred_train_labels = clf.predict(train_sequences)
pred_test_labels = clf.predict(test_sequences)

# 评估
print("========== Training Set ==========")
report = classification_report(train_labels, pred_train_labels, digits=4)
print(report)

print("========== Test Set ==========")
report = classification_report(test_labels, pred_test_labels, digits=4)
print(report)


'''
分类任务的的基本处理流程:
1. 查看训练数据
2. 生成训练数据
3. 将文本转换成向量（特征抽取）
4. 构建分类器，进行训练和预测
5. 对分类器的评估
'''