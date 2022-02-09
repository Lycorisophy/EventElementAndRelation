import re
import os
from bs4 import BeautifulSoup

def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

# 对原始数据进行预处理
org_path = "data/CEC"
preprocessed_path = "cut_result"
labels = os.listdir(org_path)
for label in labels:
    files = os.listdir(org_path+'/'+label)
    for file in files:
        f1 = open(org_path+'/'+label+'/'+file, "r", encoding='utf-8')
        xml = f1.read()
        f1.close()
        soup = BeautifulSoup(xml, 'lxml')
        content = soup.content.get_text(strip=True).replace(' ', '')
        f2 = open(preprocessed_path + '/' + label + '/' + file[:-4] + '.txt', "w", encoding='utf-8')
        f2.writelines('\n'.join(cut_sent(content)))
        f2.close()
print('over')
