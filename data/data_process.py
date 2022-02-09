import glob
import re
import os
from bs4 import BeautifulSoup
from language_model.transformers import ElectraTokenizer
import numpy as np


def remove_punctuation(line):
    space_punctuation = '''，。、�；!！.,.：:()（）"“”《》·丶．/・-'''
    line1 = re.sub(u"（.*?）", "", line)  # 去除括号内注释
    line2 = re.sub("[%s]+" % space_punctuation, " ", line1)  # 去除标点、特殊字母
    return line2


def stopwordslist():
    stopwords = [line.strip() for line in open('stop_word.txt', encoding='UTF-8').readlines()]
    return stopwords


def remove_stopword(sentence_depart):
    stopwords = stopwordslist()
    outstr = ''
    for word in sentence_depart:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr


try:
    import jieba


    def seg_depart(sentence):
        sentence_depart = jieba.cut(sentence.strip())
        return remove_stopword(sentence_depart)
except ImportError:
    def seg_depart(sentence):
        return sentence


def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    return para.split("\n")


def pos_depart(sentence, segmentor, postagger, recognizer):
    try:
        stopwords = stopwordslist()
        line = remove_punctuation(sentence)  # 去除标点、特殊字母
        seg_list = list(segmentor.segment(line))  # 分词
        postags = postagger.postag(seg_list)
        netags = recognizer.recognize(seg_list, postags)  # 命名实体识别
        n_words = v_words = o_words = ''
        try:
            sum1 = sum2 = sum3 = 0
            for postag, word, netag in zip(postags, seg_list, netags):
                if word not in stopwords and word != '\t':
                    if postag[0] == 'n' and netag == 'O':  # 词性标注
                        n_words += word
                        n_words += ' '
                        sum1 += 1
                    elif postag[0] == 'v':
                        v_words += word
                        v_words += ' '
                        sum2 += 1
                    else:
                        o_words += word
                        o_words += ' '
                        sum3 += 1
            count = [sum1, sum2, sum3]
            return n_words, v_words, o_words, count
        except IndexError:
            return
    except KeyError:
        return


def text2tokens(tokenizer, text):
    tokens_a = tokenizer.tokenize(text)
    tokens = []
    tokens.append("[CLS]")
    for token in tokens_a:
        tokens.append(token)
    tokens.append("[SEP]")
    tokens.append("[SEP]")
    return tokens


def text2tokens_noCS(tokenizer, text):
    tokens_a = tokenizer.tokenize(text)
    tokens = []
    for token in tokens_a:
        tokens.append(token)
    return tokens


def xml2bio(path):
    tokenizer = ElectraTokenizer.from_pretrained('chinese_rbtl3_pytorch/vocab.txt')
    labels = os.listdir(path)
    Time = re.compile('<time.*/time>')
    Location = re.compile('<location.*/location>')
    E_object = re.compile('<object.*/object>')
    Participant = re.compile('<participant.*/participant>')
    Means = re.compile('<mean.*/mean>')
    Denoter = re.compile('<denoter.*/denoter>')
    f1 = open("bio/train.txt", 'w', encoding='utf-8')
    for label in labels:
        files = os.listdir(path + '/' + label)
        for file in files:
            f = open(path + '/' + label + '/' + file, 'r', encoding='utf-8')
            xml = f.read()
            f.close()
            soup = BeautifulSoup(xml, 'lxml')
            for event in soup.find_all('sentence'):
                event = str(event)
                try:
                    time = re.sub(u"<.*?>", "", Time.findall(event)[0])
                except IndexError:
                    time = ""
                try:
                    location = re.sub(u"<.*?>", "", Location.findall(event)[0])
                except IndexError:
                    location = ""
                try:
                    e_object = re.sub(u"<.*?>", "", E_object.findall(event)[0])
                except IndexError:
                    e_object = ""
                try:
                    participant = re.sub(u"<.*?>", "", Participant.findall(event)[0])
                except IndexError:
                    participant = ""
                try:
                    means = re.sub(u"<.*?>", "", Means.findall(event)[0])
                except IndexError:
                    means = ""
                try:
                    denoter = re.sub(u"<.*?>", "", Denoter.findall(event)[0])
                except IndexError:
                    denoter = ""
                content = re.sub(u"<.*?>", "", str(event))
                content = content.replace('\n', '')
                content = content.replace('\t', '')
                tokens = text2tokens_noCS(tokenizer, content.replace(' ', ''))
                bio = []
                for i in range(len(tokens)):
                    bio.append('o')
                tokens_t = text2tokens_noCS(tokenizer, time)
                tokens_l = text2tokens_noCS(tokenizer, location)
                tokens_o = text2tokens_noCS(tokenizer, e_object)
                tokens_p = text2tokens_noCS(tokenizer, participant)
                tokens_m = text2tokens_noCS(tokenizer, means)
                tokens_d = text2tokens_noCS(tokenizer, denoter)
                for idx, token in enumerate(tokens):
                    tmp = tokens[idx:idx + len(tokens_t)]
                    if tmp == tokens_t and len(tmp) != 0:
                        bio[idx] = 'B-Time'
                        for i in range(len(tokens_t) - 1):
                            bio[idx + 1 + i] = 'I-Time'
                    tmp = tokens[idx:idx + len(tokens_l)]
                    if tmp == tokens_l and len(tmp) != 0:
                        bio[idx] = 'B-Location'
                        for i in range(len(tokens_l) - 1):
                            bio[idx + 1 + i] = 'I-Location'
                    tmp = tokens[idx:idx + len(tokens_o)]
                    if tmp == tokens_o and len(tmp) != 0:
                        bio[idx] = 'B-Object'
                        for i in range(len(tokens_o) - 1):
                            bio[idx + 1 + i] = 'I-Object'
                    tmp = tokens[idx:idx + len(tokens_p)]
                    if tmp == tokens_p and len(tmp) != 0:
                        bio[idx] = 'B-Participant'
                        for i in range(len(tokens_p) - 1):
                            bio[idx + 1 + i] = 'I-Participant'
                    tmp = tokens[idx:idx + len(tokens_m)]
                    if tmp == tokens_m and len(tmp) != 0:
                        bio[idx] = 'B-Means'
                        for i in range(len(tokens_m) - 1):
                            bio[idx + 1 + i] = 'I-Means'
                    tmp = tokens[idx:idx + len(tokens_d)]
                    if tmp == tokens_d and len(tmp) != 0:
                        bio[idx] = 'B-Denoter'
                        for i in range(len(tokens_d) - 1):
                            bio[idx + 1 + i] = 'I-Denoter'
                for token, bi in zip(tokens, bio):
                    f1.writelines(token)
                    f1.writelines('\t')
                    f1.writelines(bi)
                    f1.writelines('\n')
                f1.writelines('\n')
    f1.close()
    return


try:
    from pyltp import SentenceSplitter


    def xml2txt(path):
        labels = os.listdir(path)
        for label in labels:
            files = os.listdir(path + '/' + label)
            f1 = open("result/" + label + ".txt", 'w', encoding='utf-8')
            for file in files:
                f = open(path + '/' + label + '/' + file, 'r', encoding='utf-8')
                xml = f.read()
                f.close()
                soup = BeautifulSoup(xml, 'lxml')
                content = soup.content.get_text(strip=True).replace(' ', '')
                f1.writelines(SentenceSplitter.split(content))
                f1.writelines('\n')
            f1.close()
        return
except ImportError:
    def xml2txt(path):
        labels = os.listdir(path)
        for label in labels:
            files = os.listdir(path + '/' + label)
            f1 = open("result/" + label + ".txt", 'w', encoding='utf-8')
            for file in files:
                f = open(path + '/' + label + '/' + file, 'r', encoding='utf-8')
                xml = f.read()
                f.close()
                soup = BeautifulSoup(xml, 'lxml')
                content = soup.content.get_text(strip=True).replace(' ', '')
                f1.writelines(content)
                f1.writelines('\n')
            f1.close()
        return


def xml2event(path):
    labels = os.listdir(path)
    for label in labels:
        files = os.listdir(path + '/' + label)
        f1 = open("event/" + label + ".txt", 'w', encoding='utf-8')
        for file in files:
            f = open(path + '/' + label + '/' + file, 'r', encoding='utf-8')
            xml = f.read()
            f.close()
            soup = BeautifulSoup(xml, 'lxml')
            for event in soup.find_all('event'):
                content = re.sub(u"<.*?>", "", str(event))
                content = content.replace('\n', '')
                content = content.replace('\t', '')
                f1.writelines(content.replace(' ', ''))
                f1.writelines('\n')
        f1.close()
    return


def xml2d_event(path):
    labels = os.listdir(path)
    Time = re.compile('<time.*/time>')
    Location = re.compile('<location.*/location>')
    E_object = re.compile('<object.*/object>')
    Participant = re.compile('<participant.*/participant>')
    Means = re.compile('<means.*/means>')
    Denoter = re.compile('<denoter.*/denoter>')
    for label in labels:
        files = os.listdir(path + '/' + label)
        f1 = open("d_event/" + label + ".csv", 'w', encoding='utf-8')
        f1.writelines("\"Time\",\"Location\",\"Object\",\"Participant\",\"Means\",\"Denoter\"")
        f1.writelines('\n')
        for file in files:
            f = open(path + '/' + label + '/' + file, 'r', encoding='utf-8')
            xml = f.read()
            f.close()
            soup = BeautifulSoup(xml, 'lxml')
            for event in soup.find_all('event'):
                event = str(event)
                try:
                    time = re.sub(u"<.*?>", "", Time.findall(event)[0])
                except IndexError:
                    time = ""
                try:
                    location = re.sub(u"<.*?>", "", Location.findall(event)[0])
                except IndexError:
                    location = ""
                try:
                    e_object = re.sub(u"<.*?>", "", E_object.findall(event)[0])
                except IndexError:
                    e_object = ""
                try:
                    participant = re.sub(u"<.*?>", "", Participant.findall(event)[0])
                except IndexError:
                    participant = ""
                try:
                    means = re.sub(u"<.*?>", "", Means.findall(event)[0])
                except IndexError:
                    means = ""
                try:
                    denoter = re.sub(u"<.*?>", "", Denoter.findall(event)[0])
                except IndexError:
                    denoter = ""

                f1.writelines("\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\""
                              .format(time, location, e_object, participant, means, denoter))
                f1.writelines('\n')
        f1.close()
    return


def denoter_masked(path):
    labels = os.listdir(path)
    Denoter = re.compile('<denoter.*/denoter>')
    for label in labels:
        files = os.listdir(path + '/' + label)
        f1 = open("denoter_masked/" + label + ".txt", 'w', encoding='utf-8')
        for file in files:
            f = open(path + '/' + label + '/' + file, 'r', encoding='utf-8')
            xml = f.read()
            f.close()
            soup = BeautifulSoup(xml, 'lxml')
            for event in soup.find_all('event'):
                event = str(event)
                denoter = re.sub(u"<.*?>", "", Denoter.findall(event)[0])
                content = re.sub(u"<.*?>", "", event)
                content = content.replace('\n', '')
                content = content.replace('\t', '')
                content = content.replace(' ', '')
                f1.writelines("^{}^   #{}#".format(content, denoter))
                f1.writelines('\n')
        f1.close()
    return


# 按行读取文本
def read_lines(file_name):
    lines = open(file_name, encoding='utf-8').read().strip().split('\n')
    return [one_line for one_line in lines]


# 创建文件夹
def mkdir_path(path):
    path = path.strip()  # 去除首位空格
    path = path.rstrip("\\")  # 去除尾部 \ 符号
    is_exists = os.path.exists(path)  # 判断路径是否存在
    if not is_exists:  # 判断结果
        os.makedirs(path)  # 如果不存在则创建目录
        print(path + ' 创建成功')
        return True
    else:
        return False


def bio2xml(data_path, result_path, tag_to_tag):
    mkdir_path(result_path)
    Text = re.compile('.+\t')
    Label = re.compile('\t.+')
    Name = re.compile('\\\\.*\.txt')
    all_filenames = glob.glob(data_path + "/*.txt")
    for idx, filename in enumerate(all_filenames):
        words = []
        bios = []
        cnt = 0
        text = ""
        name = re.sub(u"\\\\.*?\\\\", "", Name.findall(filename)[0])
        name = re.sub(u"\\\\", "", name)
        name = re.sub(u".txt", "", name)
        f = open(result_path + "/" + name + ".xml", 'w', encoding='utf-8')
        f.writelines("<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
        f.writelines("\n")
        f.writelines("<Body>")
        f.writelines("\n")
        f.writelines("\t<Content>")
        f.writelines("\n")
        lines = read_lines(filename)
        for line in lines:
            if line != "":
                text += line
                words.append(re.sub(u'\t', '', Text.findall(line)[0]))
                tag = re.sub(u'\t', '', Label.findall(line)[0])
                try:
                    bios.append(tag_to_tag[tag])
                except KeyError:
                    bios.append(tag)
            else:
                f.writelines("\t\t<Sentence>")
                f.writelines("\n")
                if len(bios) == 0:
                    break
                last_bio = bios[0]
                if last_bio != 'Other':
                    f.writelines("\t\t\t<{}>".format(last_bio))
                else:
                    f.writelines("\t\t\t")
                if not (np.array(bios) == 'Other').all():
                    cnt += 1
                    # f.writelines("<Event eid=\"e{}\">".format(cnt))
                    for word, bio in zip(words, bios):
                        if bio != 'Other' and bio != last_bio and last_bio != 'Other':
                            f.writelines("</{}>".format(last_bio))
                            last_bio = bio
                            f.writelines("\n")
                            f.writelines("\t\t\t<{}>".format(bio))
                            f.writelines(word)
                        elif bio != 'Other' and last_bio == 'Other':
                            f.writelines("\n")
                            last_bio = bio
                            f.writelines("\t\t\t<{}>".format(bio))
                            f.writelines(word)
                        elif bio != 'Other' and bio == last_bio:
                            f.writelines(word)
                        elif bio == 'Other' and bio != last_bio:
                            f.writelines("</{}>".format(last_bio))
                            last_bio = bio
                            f.writelines(word)
                        elif bio == 'Other' and bio == last_bio:
                            f.writelines(word)
                    final_bio = bios[-1]
                    if final_bio != 'Other':
                        f.writelines("</{}>".format(final_bio))
                    # f.writelines("</Event>")
                else:
                    f.writelines(text)
                f.writelines("\n")
                f.writelines("\t\t</Sentence>")
                f.writelines("\n")
                words = []
                bios = []
                text = ""
        f.writelines("\t</Content>")
        f.writelines("\n")
        f.writelines("</Body>")
        f.writelines("\n")
        f.close()
    return True


if __name__ == "__main__":
    START_TAG = "[CLS]"
    STOP_TAG = "[SEP]"
    MASK_TAG = "[MASK]"
    tag_to_tag = {"B-Time": "Time", "I-Time": "Time", "B-Location": "Location", "I-Location": "Location",
                  "B-Object": "Object", "I-Object": "Object", "B-Participant": "Participant",
                  "I-Participant": "Participant", "B-Means": "Means", "I-Means": "Means", "B-Denoter": "Denoter",
                  "I-Denoter": "Denoter", "o": "Other", START_TAG: "Other", STOP_TAG: "Other", MASK_TAG: "Other"}
    data_paths = glob.glob("../log/*")
    result_paths = glob.glob("../xml_result/*")
    for data_path, result_path in zip(data_paths, result_paths):
        bio2xml(data_path, result_path, tag_to_tag)
