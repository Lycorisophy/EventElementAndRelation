import re
import os
from bs4 import BeautifulSoup
import random
from language_model.transformers import ElectraTokenizer
import numpy as np


def text2tokens_noCS(tokenizer, text):
    tokens_a = tokenizer.tokenize(text)
    tokens = []
    for token in tokens_a:
        tokens.append(token)
    return tokens


def get_relation(path):
    labels = os.listdir(path)
    f1 = open("relation.csv", 'w', encoding='utf-8')
    f1.writelines("\"{}\"\t\"{}\"\t\"{}\"".format("event1", "event2", "reltype"))
    f1.writelines('\n')
    Relation = re.compile('<erelation.*</erelation>')
    Eid = re.compile('eid=\"e\d*\"')
    Reltype = re.compile('reltype=\"\w*\"')
    count, success, wrong, manual = 0, 0, 0, 0
    for label in labels:
        files = os.listdir(path + '/' + label)
        for file in files:
            count += 1
            f = open(path + '/' + label + '/' + file, 'r', encoding='utf-8')
            xml = f.read()
            f.close()
            soup = BeautifulSoup(xml, 'lxml')
            body = str(soup.find_all('body')[0])
            relations = Relation.findall(body)
            body = body.replace('\n', '')
            for relation in relations:
                success += 1
                eid = Eid.findall(relation)
                reltype = str(Reltype.findall(relation)[0])
                reltype = re.sub(u"reltype=\"", "", reltype)
                reltype = re.sub(u"\"", "", reltype)
                try:
                    str1 = '<event '+eid[0]+'.*</event>'
                    Event1 = re.compile(str1)
                    event1 = Event1.findall(body)[0]
                    str2 = '<event ' + eid[1] + '.*</event>'
                    Event2 = re.compile(str2)
                    event2 = Event2.findall(body)[0]
                except:
                    wrong += 1
                    if reltype != 'Thoughtcontent':
                        manual += 1
                        print(body)
                        print(relation)
                event1 = re.sub(u"<.*?>", "", event1)
                event1 = event1.replace('\t', '')
                event2 = re.sub(u"<.*?>", "", event2)
                event2 = event2.replace('\t', '')
                f1.writelines("\"{}\"\t\"{}\"\t\"{}\"".format(event1, event2, reltype))
                f1.writelines('\n')
    f1.close()
    print("文件数：{}，关系数：{}，标注失败：{}, 需手动标注：{}, 其余关系为Thoughtcontent".format(count, success, wrong, manual))
    return


def get_bio(event1, tokenizer):
    Time = re.compile('<time.*/time>')
    Location = re.compile('<location.*/location>')
    E_object = re.compile('<object.*/object>')
    Participant = re.compile('<participant.*/participant>')
    Means = re.compile('<means.*/means>')
    Denoter = re.compile('<denoter.*/denoter>')
    try:
        time1 = re.sub(u"<.*?>", "", Time.findall(event1)[0])
    except IndexError:
        time1 = ""
    try:
        location1 = re.sub(u"<.*?>", "", Location.findall(event1)[0])
    except IndexError:
        location1 = ""
    try:
        e_object1 = re.sub(u"<.*?>", "", E_object.findall(event1)[0])
    except IndexError:
        e_object1 = ""
    try:
        participant1 = re.sub(u"<.*?>", "", Participant.findall(event1)[0])
    except IndexError:
        participant1 = ""
    try:
        means1 = re.sub(u"<.*?>", "", Means.findall(event1)[0])
    except IndexError:
        means1 = ""
    try:
        denoter1 = re.sub(u"<.*?>", "", Denoter.findall(event1)[0])
    except IndexError:
        denoter1 = ""
    tokens_t1 = text2tokens_noCS(tokenizer, time1)
    tokens_l1 = text2tokens_noCS(tokenizer, location1)
    tokens_o1 = text2tokens_noCS(tokenizer, e_object1)
    tokens_p1 = text2tokens_noCS(tokenizer, participant1)
    tokens_m1 = text2tokens_noCS(tokenizer, means1)
    tokens_d1 = text2tokens_noCS(tokenizer, denoter1)
    event1 = re.sub(u"<.*?>", "", event1)
    event1 = event1.replace('\t', '')
    event1 = event1.replace(' ', '')
    tokens1 = text2tokens_noCS(tokenizer, event1)
    bio1 = []
    for i in range(len(tokens1)):
        bio1.append(0)
    for idx, token in enumerate(tokens1):
        tmp = tokens1[idx:idx + len(tokens_t1)]
        if tmp == tokens_t1 and len(tmp) != 0:
            bio1[idx] = 1
            for i in range(len(tokens_t1) - 1):
                bio1[idx + 1 + i] = 2
        tmp = tokens1[idx:idx + len(tokens_l1)]
        if tmp == tokens_l1 and len(tmp) != 0:
            bio1[idx] = 3
            for i in range(len(tokens_l1) - 1):
                bio1[idx + 1 + i] = 4
        tmp = tokens1[idx:idx + len(tokens_o1)]
        if tmp == tokens_o1 and len(tmp) != 0:
            bio1[idx] = 5
            for i in range(len(tokens_o1) - 1):
                bio1[idx + 1 + i] = 6
        tmp = tokens1[idx:idx + len(tokens_p1)]
        if tmp == tokens_p1 and len(tmp) != 0:
            bio1[idx] = 7
            for i in range(len(tokens_p1) - 1):
                bio1[idx + 1 + i] = 8
        tmp = tokens1[idx:idx + len(tokens_m1)]
        if tmp == tokens_m1 and len(tmp) != 0:
            bio1[idx] = 9
            for i in range(len(tokens_m1) - 1):
                bio1[idx + 1 + i] = 10
        tmp = tokens1[idx:idx + len(tokens_d1)]
        if tmp == tokens_d1 and len(tmp) != 0:
            bio1[idx] = 11
            for i in range(len(tokens_d1) - 1):
                bio1[idx + 1 + i] = 12
    return event1, bio1


def get_rel_and_role(path, tokenizer):
    labels = os.listdir(path)
    E1, E2, B1, B2, R = [], [], [], [], []
    tE1, tE2, tB1, tB2, tR = [], [], [], [], []
    Relation = re.compile('<erelation.*</erelation>')
    Eid = re.compile('eid=\"e\d*\"')
    Reltype = re.compile('reltype=\"\w*\"')
    count, success, wrong, manual = 0, 0, 0, 0
    for label in labels:
        files = os.listdir(path + '/' + label)
        for file in files:
            count += 1
            f = open(path + '/' + label + '/' + file, 'r', encoding='utf-8')
            xml = f.read()
            f.close()
            soup = BeautifulSoup(xml, 'lxml')
            body = str(soup.find_all('body')[0])
            relations = Relation.findall(body)
            body = body.replace('\n', '')
            for relation in relations:
                success += 1
                eid = Eid.findall(relation)
                reltype = str(Reltype.findall(relation)[0])
                reltype = re.sub(u"reltype=\"", "", reltype)
                reltype = re.sub(u"\"", "", reltype)
                try:
                    str1 = '<event '+eid[0]+'.*</event>'
                    Event1 = re.compile(str1)
                    event1 = Event1.findall(body)[0]
                    str2 = '<event ' + eid[1] + '.*</event>'
                    Event2 = re.compile(str2)
                    event2 = Event2.findall(body)[0]
                except:
                    wrong += 1
                    if reltype != 'Thoughtcontent':
                        manual += 1
                        # print(body)
                        # print(relation)
                if reltype != 'Thoughtcontent':
                    event1, bio1 = get_bio(event1, tokenizer)
                    event2, bio2 = get_bio(event2, tokenizer)
                    r = random.randint(1, 10)
                    if r <= 8:
                        E1.append(event1)
                        E2.append(event2)
                        B1.append(bio1)
                        B2.append(bio2)
                        R.append(reltype)
                    else:
                        tE1.append(event1)
                        tE2.append(event2)
                        tB1.append(bio1)
                        tB2.append(bio2)
                        tR.append(reltype)
    # np.save('RnR_data/train/e1.npy', E1)
    # np.save('RnR_data/train/e2.npy', E2)
    # np.save('RnR_data/train/b1.npy', B1)
    # np.save('RnR_data/train/b2.npy', B2)
    # np.save('RnR_data/train/r.npy', R)
    # np.save('RnR_data/test/e1.npy', tE1)
    # np.save('RnR_data/test/e2.npy', tE2)
    # np.save('RnR_data/test/b1.npy', tB1)
    # np.save('RnR_data/test/b2.npy', tB2)
    # np.save('RnR_data/test/r.npy', tR)
    return [E1, E2, B1, B2, R], [tE1, tE2, tB1, tB2, tR]


def get_all_rel_and_role(path, tokenizer):
    labels = os.listdir(path)
    E1, E2, B1, B2, R = [], [], [], [], []
    Relation = re.compile('<erelation.*</erelation>')
    Eid = re.compile('eid=\"e\d*\"')
    Reltype = re.compile('reltype=\"\w*\"')
    count, success, wrong, manual = 0, 0, 0, 0
    for label in labels:
        files = os.listdir(path + '/' + label)
        for file in files:
            count += 1
            f = open(path + '/' + label + '/' + file, 'r', encoding='utf-8')
            xml = f.read()
            f.close()
            soup = BeautifulSoup(xml, 'lxml')
            body = str(soup.find_all('body')[0])
            relations = Relation.findall(body)
            body = body.replace('\n', '')
            for relation in relations:
                success += 1
                eid = Eid.findall(relation)
                reltype = str(Reltype.findall(relation)[0])
                reltype = re.sub(u"reltype=\"", "", reltype)
                reltype = re.sub(u"\"", "", reltype)
                try:
                    str1 = '<event '+eid[0]+'.*</event>'
                    Event1 = re.compile(str1)
                    event1 = Event1.findall(body)[0]
                    str2 = '<event ' + eid[1] + '.*</event>'
                    Event2 = re.compile(str2)
                    event2 = Event2.findall(body)[0]
                    if reltype != 'Thoughtcontent':
                        manual += 1
                        event1, bio1 = get_bio(event1, tokenizer)
                        event2, bio2 = get_bio(event2, tokenizer)
                        E1.append(event1)
                        E2.append(event2)
                        B1.append(bio1)
                        B2.append(bio2)
                        R.append(reltype)
                except:
                    wrong += 1
    # np.save('RnR_data/all/e1.npy', E1)
    # np.save('RnR_data/all/e2.npy', E2)
    # np.save('RnR_data/all/b1.npy', B1)
    # np.save('RnR_data/all/b2.npy', B2)
    # np.save('RnR_data/all/r.npy', R)
    return [E1, E2, B1, B2, R]


def get_all_rel_role_con(path, tokenizer):
    labels = os.listdir(path)
    E1, E2, B1, B2, R, C1, C2 = [], [], [], [], [], [], []
    Relation = re.compile('<erelation.*</erelation>')
    Eid = re.compile('eid=\"e\d*\"')
    Reltype = re.compile('reltype=\"\w*\"')
    count, success, wrong, manual = 0, 0, 0, 0
    for label in labels:
        files = os.listdir(path + '/' + label)
        for file in files:
            count += 1
            f = open(path + '/' + label + '/' + file, 'r', encoding='utf-8')
            xml = f.read()
            f.close()
            soup = BeautifulSoup(xml, 'lxml')
            body = str(soup.find_all('body')[0])
            relations = Relation.findall(body)
            body = body.replace('\n', '')
            for relation in relations:
                success += 1
                eid = Eid.findall(relation)
                reltype = str(Reltype.findall(relation)[0])
                reltype = re.sub(u"reltype=\"", "", reltype)
                reltype = re.sub(u"\"", "", reltype)
                try:
                    eid1 = eid[0]
                    eid2 = eid[1]

                    str1 = '<event '+eid1+'.*</event>'
                    Event1 = re.compile(str1)
                    event1 = Event1.findall(body)[0]
                    str2 = '<event ' + eid2 + '.*</event>'
                    Event2 = re.compile(str2)
                    event2 = Event2.findall(body)[0]

                    # 提取第一句的下一句和第二句的上一句
                    eid3 = str(int("".join(filter(str.isdigit, eid1)))+1)
                    eid4 = str(int("".join(filter(str.isdigit, eid2)))-1)
                    str3 = '<event eid="e'+eid3+'".*</event>'
                    Event3 = re.compile(str3)
                    event3 = Event3.findall(body)[0]
                    str4 = '<event eid="e'+eid4+'".*</event>'
                    Event4 = re.compile(str4)
                    event4 = Event4.findall(body)[0]
                    event3 = re.sub(u"<.*?>", "", event3)
                    event3 = event3.replace('\t', '')
                    event3 = event3.replace(' ', '')
                    event4 = re.sub(u"<.*?>", "", event4)
                    event4 = event4.replace('\t', '')
                    event4 = event4.replace(' ', '')

                    if reltype != 'Thoughtcontent':
                        manual += 1
                        event1, bio1 = get_bio(event1, tokenizer)
                        event2, bio2 = get_bio(event2, tokenizer)
                        E1.append(event1)
                        E2.append(event2)
                        B1.append(bio1)
                        B2.append(bio2)
                        R.append(reltype)
                        C1.append(event3)
                        C2.append(event4)
                except:
                    wrong += 1
    np.save('RRC_data/all/e1.npy', E1)
    np.save('RRC_data/all/e2.npy', E2)
    np.save('RRC_data/all/b1.npy', B1)
    np.save('RRC_data/all/b2.npy', B2)
    np.save('RRC_data/all/r.npy', R)
    np.save('RRC_data/all/c1.npy', C1)
    np.save('RRC_data/all/c2.npy', C2)
    return [E1, E2, B1, B2, R, C1, C2]


if __name__ == "__main__":
    tokenizer = ElectraTokenizer.from_pretrained('D:\事件关系抽取\pretrained_model/pytorch_electra_180g_large/vocab.txt')
    data_path = "CEC"
    get_all_rel_role_con(data_path, tokenizer)
