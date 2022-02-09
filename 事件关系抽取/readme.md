# Event Relation Extraction
## 目录说明
1. checkpoint
    1.  ckeckpoint **新训练模型保存路径**
    2.  ckeckpoint+'数字' **历史训练模型保存路径**
2. config
    1.  relation_classify_args.txt **代码全局参数设置(临时文件，没有用)**
    2.  relation_classify_config.json **模型全局参数设置（重要文件，不要随意修改）**
3. data
    1.  CEC **CEC2.0中文突发事件原始语料**
    2.  rel_data  **CEC2.0关系语料**
    3.  RnR_data    **带角色标注的关系语料(numpy格式)**
    4.  RRC_data    **带角色标注和上下文特征的关系语料**
    5.  data_get.py     **数据预处理代码**
4. garbage  **垃圾内容回收站**
5. language_model
    1.transformers  *开源语言模型+transformer框架整合包[Github](https://github.com/huggingface/transformers)*
6. log **实验日志文件**[妥善保存]
7. models **定义了实验所需的网络模型**
8. nn **定义了实验所需的网络模型的组件**
9.  pretrained_model **预训练的嵌入层语言模型和所需字典**
10. utils **一些控制类和函数**
11. 相关文档
12. relation_classify_***_train.py  **包含了数据读取、模型训练和测试部分**
13. requirements.txt    **项目依赖的包**
14. 实验结果比较.xlsx

## 新闻
**2020年11月25日**
- **嗨，我是宋杨** 
****Welcome to my team!****
>Enjoy yourself! 
>[My Github](https://github.com/LySoY). 

## 如何开始？
### 1. 安装环境

需要安装**Python 3.6**.

* 运行 `pip install -r requirements.txt` 安装依赖包.

### 2. (可选项)下载预训练模型


### 3. (可选项) 测试环境

`python relation_classify_base_train.py`


### 4. (可选项)  配置数据集


### 5. 运行程序

### 6. (可选项) 使用GPU
You will need to ensure GPU drivers are properly installed 
and that your CUDA version matches your PyTorch and Tensorflow installations.
