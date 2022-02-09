import re
import os
from bs4 import BeautifulSoup
from pyltp import SentenceSplitter
from language_model.transformers import ElectraTokenizer


# 按行读取文本
def read_lines(file_name):
    lines = open(file_name, encoding='utf-8').read().strip().split('\n')
    return [one_line for one_line in lines]


def text2tokens(tokenizer, text):
    tokens_a = tokenizer.tokenize(text)
    tokens = ["[CLS]"]
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
    tokenizer = ElectraTokenizer.from_pretrained('vocab.txt')
    labels = os.listdir(path)
    Time = re.compile('<time.*/time>')
    Location = re.compile('<location.*/location>')
    E_object = re.compile('<object.*/object>')
    Participant = re.compile('<participant.*/participant>')
    Means = re.compile('<means.*/means>')
    Denoter = re.compile('<denoter.*/denoter>')
    for label in labels:
        files = os.listdir(path + '/' + label)
        f1 = open("bio/" + label + ".txt", 'w', encoding='utf-8')
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
                        for i in range(len(tokens_t)-1):
                            bio[idx+1+i] = 'I-Time'
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


def xml2txt(path):
    labels = os.listdir(path)
    for label in labels:
        files = os.listdir(path+'/'+label)
        f1 = open("result/"+label+".txt", 'w', encoding='utf-8')
        for file in files:
            f = open(path+'/'+label+'/'+file, 'r', encoding='utf-8')
            xml = f.read()
            f.close()
            soup = BeautifulSoup(xml, 'lxml')
            content = soup.content.get_text(strip=True).replace(' ', '')
            f1.writelines(SentenceSplitter.split(content))
            f1.writelines('\n')
        f1.close()
    return


def xml2event(path):
    labels = os.listdir(path)
    for label in labels:
        files = os.listdir(path+'/'+label)
        f1 = open("event/"+label+".txt", 'w', encoding='utf-8')
        for file in files:
            f = open(path+'/'+label+'/'+file, 'r', encoding='utf-8')
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
        files = os.listdir(path+'/'+label)
        f1 = open("d_event/"+label+".csv", 'w', encoding='utf-8')
        f1.writelines("\"Time\",\"Location\",\"Object\",\"Participant\",\"Means\",\"Denoter\"")
        f1.writelines('\n')
        for file in files:
            f = open(path+'/'+label+'/'+file, 'r', encoding='utf-8')
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
        files = os.listdir(path+'/'+label)
        f1 = open("denoter_masked/"+label+".txt", 'w', encoding='utf-8')
        for file in files:
            f = open(path+'/'+label+'/'+file, 'r', encoding='utf-8')
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


if __name__ == "__main__":
    path = "CEC"
    xml2bio(path)
    print("finish")
