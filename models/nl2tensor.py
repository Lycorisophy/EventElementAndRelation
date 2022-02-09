import torch
import re


# 按行读取文本
def read_lines(file_name):
    lines = open(file_name, encoding='utf-8').read().strip().split('\n')
    return [one_line for one_line in lines]


# 切分句子
def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


# 中文字符转token
def convert_single_example(tokenizer, text_a, max_seq_length):
    tokens_a = tokenizer.tokenize(text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    while len(input_ids) > max_seq_length:
        input_ids.pop()
        input_mask.pop()
        segment_ids.pop()
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return input_ids, input_mask, segment_ids


def convert_single_with_pos(tokenizer, text_a, max_seq_length, first_text=True):
    tokens_a = tokenizer.tokenize(text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    tokens = []
    pos_ids = []
    tokens.append("[CLS]")
    pos = 0
    pos_ids.append(pos)
    for token in tokens_a:
        tokens.append(token)
        pos += 1
        pos_ids.append(pos)
    tokens.append("[SEP]")
    pos += 1
    pos_ids.append(pos)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    mask_ids = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        mask_ids.append(0)
        pos += 1
        pos_ids.append(pos)
    while len(input_ids) > max_seq_length:
        input_ids.pop()
        mask_ids.pop()
        pos_ids.pop()
    token_type = [0] * len(input_ids) if first_text else [1] * len(input_ids)
    assert len(input_ids) == max_seq_length
    assert len(token_type) == max_seq_length
    assert len(pos_ids) == max_seq_length
    assert len(mask_ids) == max_seq_length
    return [input_ids, token_type, pos_ids, mask_ids]


def convert_single_with_ele(tokenizer, text_a, max_seq_length, ee, first_text=True):
    tokens_a = tokenizer.tokenize(text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    tokens = []
    pos_ids = []
    tokens.append("[CLS]")
    pos = 0
    pos_ids.append(pos)
    for token in tokens_a:
        tokens.append(token)
        pos += 1
        pos_ids.append(pos)
    tokens.append("[SEP]")
    pos += 1
    pos_ids.append(pos)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    mask_ids = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        mask_ids.append(0)
        pos += 1
        pos_ids.append(pos)
        ee.append(0)
    while len(input_ids) > max_seq_length:
        input_ids.pop()
        mask_ids.pop()
        pos_ids.pop()
    while len(ee) < max_seq_length:
        ee.append(0)
    while len(ee) > max_seq_length:
        ee.pop()
    token_type = [0] * len(input_ids) if first_text else [1] * len(input_ids)
    assert len(input_ids) == max_seq_length
    assert len(token_type) == max_seq_length
    assert len(pos_ids) == max_seq_length
    assert len(mask_ids) == max_seq_length
    return [input_ids, token_type, pos_ids, mask_ids, ee]



def convert_double_example(tokenizer, text_a, text_b, max_seq_length):
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = tokenizer.tokenize(text_b)
    half_seq_len = max_seq_length//2
    if len(tokens_a) > half_seq_len - 1:
        tokens_a = tokens_a[0:(half_seq_len - 1)]
    if len(tokens_b) > half_seq_len - 1:
        tokens_b = tokens_b[0:(half_seq_len - 1)]
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    while len(input_ids) > max_seq_length:
        input_ids.pop()
        input_mask.pop()
        segment_ids.pop()
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return input_ids, input_mask, segment_ids


def convert_single_list(arrs, length, num):
    tokens = []
    tokens.append(num)
    for arr in arrs:
        tokens.append(arr)
    tokens.append(num+1)
    while len(tokens) < length:
        tokens.append(num+2)
    while len(tokens) > length:
        tokens.pop()
    assert len(tokens) == length
    return tokens


def text2tensor(tokenizer, l_text, max_len):
    output_ids, output_mask, segment_ids = convert_single_example(tokenizer, l_text, max_len)
    tokens_tensor = torch.tensor([output_ids])
    masks_tensor = torch.tensor([output_mask])
    segments_tensors = torch.tensor([segment_ids])
    return tokens_tensor, masks_tensor, segments_tensors


def texts2tensor(tokenizer, l_text, r_text, max_len):
    output_ids, output_mask, segment_ids = convert_double_example(tokenizer, l_text, r_text, max_len)
    tokens_tensor = torch.tensor([output_ids])
    masks_tensor = torch.tensor([output_mask])
    segments_tensors = torch.tensor([segment_ids])
    return tokens_tensor, masks_tensor, segments_tensors


def sent2vec(l_model, tokenizer, l_text, max_len):
    tokens_tensor, masks_tensor, segments_tensors = text2tensor(tokenizer, l_text, max_len)
    embedding, _ = l_model(tokens_tensor, segments_tensors, masks_tensor, False)
    return embedding


def text2ids(tokenizer, l_text, max_len):
    tokens_tensor, masks_tensor, segments_tensors = text2tensor(tokenizer, l_text, max_len)
    return tokens_tensor.numpy(), masks_tensor.numpy(), segments_tensors.numpy()


def texts2ids(tokenizer, l_text, r_text, max_len):
    tokens_tensor, masks_tensor, segments_tensors = texts2tensor(tokenizer, l_text, r_text, max_len)
    return tokens_tensor.numpy(), masks_tensor.numpy(), segments_tensors.numpy()

# 文本转embedding和attention_mask
def text2vector(model, tokenizer, text, max_len):
    tokens_tensor, masks_tensor, segments_tensors = text2tensor(tokenizer, text, max_len)
    embedding = model(tokens_tensor, segments_tensors, masks_tensor)
    return embedding


