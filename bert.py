#coding=utf-8
import torch
import os
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np
from typing import Literal

# 注意：需确保dataset.dataset.TextDataset存在，若不存在需自定义
from .dataset.dataset import TextDataset, transform

# 预训练模型路径
PRETRAINED_PATH = 'bert-base-chinese'

class LastClassifier(nn.Module):
    def __init__(self, hidden_dim=768, output_dim=2):
        super().__init__()
        self.fn1 = nn.Linear(768, hidden_dim)
        self.ac1 = nn.Hardtanh()
        self.fn2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        '''
        映射到非线性算子空间
        但不使用 Dropout 增大随机性（数据太少）
        '''
        out = self.fn1(x)
        out = self.ac1(out)
        out = self.fn2(out)
        return out
    
class LastWeakClassifier(nn.Module):
    def __init__(self, output_dim=2):
        super().__init__()

        '''
        直接使用 Bert 内置的分类器也可以
        为了区分我们单独做了这一部分
        '''
        self.fn1 = nn.Linear(768, output_dim)

    def forward(self, x):
        '''
        单仿射变换直接输出特征
        '''
        out = self.fn1(x)
        return out
    

def train_bert_model(
    episode = 28,
    batch_size = 32,
    max_len = 50,
    learning_rate = 2e-6,
    weight_decay = 1e-3,
    num_labels=2,
    dropout_prop=0.3,
    pretraind_path='bert-base-chinese',
    save_path=os.path.join('models', 'classifier_only.pth'),
    save_txt=os.path.join('runs', 'bert-report.txt'),
    dataset=os.path.join('dataset', 'chat', 'data.txt'),
    human_test=False,
    classifier:Literal['normal', 'weak']='normal'
):
    '''
    由于我们这里存在端到端一步到位的 Bert 模型
    所以我们需要使用至少两层 Classifier 作为最终分类器
    而针对 deepseek 的优化输出则是直接使用一个仿射变换矩阵即可

    BertForSequenceClassification(
    (bert): BertModel(...)
    (dropout): Dropout(p=0.1, inplace=False)
    (classifier): LastClassifier(
        (fn1): Linear(in_features=768, out_features=768, bias=True)
        (ac1): Hardtanh(min_val=-1.0, max_val=1.0)
        (fn2): Linear(in_features=768, out_features=2, bias=True)
        (ac2): Softmax(dim=1)
    ))

    中间省略的 bert 层就是半边 Transformer, 已经预训练好了我们不需要动
    训练的时候也是直接冻结 bert 参数的, 我们做后面的微调
    '''
    # 设备初始化
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"模型运行在 {device} 上")

    tokenizer = BertTokenizer.from_pretrained(pretraind_path)
    config = BertConfig.from_pretrained(
        pretraind_path,
        num_labels=num_labels,
        hidden_dropout_prob=dropout_prop
    )
    model = BertForSequenceClassification.from_pretrained(pretraind_path, config=config).to(device)

    # 冻结BERT主体参数
    for param in model.bert.parameters():
        param.requires_grad = False

    if classifier == 'normal':
        model.classifier = LastClassifier(hidden_dim=768, output_dim=num_labels).to(device)
    elif classifier == 'weak':
        model.classifier = LastWeakClassifier(output_dim=num_labels).to(device)

    # 加载数据集
    ad_data = []
    ad_labels = []

    with open(dataset, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            part = line.strip().split('\t')
            if len(part) != 2:
                continue
            ad_labels.append(int(part[0]))
            ad_data.append(part[1])

    print("数据集统计：", pd.DataFrame({"label": ad_labels, "data": ad_data})['label'].value_counts())

    # 划分训练/测试集, 根源上解决混淆问题
    X_train, X_test, y_train, y_test = train_test_split(ad_data, ad_labels, test_size=0.1, random_state=42)
    train_dataset = TextDataset(X_train, y_train, tokenizer, max_len=max_len)
    test_dataset = TextDataset(X_test, y_test, tokenizer, max_len=max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    '''
    正则化 AdamW 反正在 NLP 挺受欢迎的
    如果你后续发现双层感知机训练的有一点慢可以直接换成 Adam
    '''
    optimizer = AdamW(
        model.classifier.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    model.train()
    for epoch in range(episode):
        losses = []
        optimizer.zero_grad()  # 每个 epoch 清空梯度
        
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{episode}') as tbar:
            for batch in tbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['label'].to(device)

                out = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(out.logits, labels)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                tbar.set_postfix(loss=np.mean(losses))

    # 测试评估, 使用未出现过的测试集进行测试 (sklearn 保证现场分类的基本没啥问题)
    model.eval()
    all_content, all_labels, all_predictions = [], [], []
    with torch.no_grad():
        with tqdm(test_loader, desc='Testing') as tbar:
            for batch in tbar:
                all_content.extend(batch['input_text'])
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['label'].to(device)

                out = model(input_ids, attention_mask, token_type_ids)
                preds = torch.argmax(out.logits, dim=1).cpu().numpy()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(preds)

    # 保存评估结果
    with open(save_txt, 'w+', encoding='utf-8') as f:
        cm = confusion_matrix(all_labels, all_predictions)
        f.write("混淆矩阵:\n" + str(cm) + "\n\n")
        f.write("分类报告:\n" + classification_report(all_labels, all_predictions) + "\n")
        
        pf = pd.DataFrame({
            'True': all_labels,
            'Pred': all_predictions,
            'Text': all_content
        })
        f.write("测试案例:\n" + pf.to_string() + "\n")

    # 绘制混淆矩阵
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join('runs', 'confusion_matrix.png'))  # 保存图片而非仅显示
    plt.show()

    sel = input(f"(Y) 进行模型保存: {save_path}\n(N) 测试后再保存\n(y/N)")
    if sel.lower().startswith('y'):
        torch.save(model.classifier.state_dict(), save_path)
        print(f"分类头参数已保存到: {save_path}")

    if human_test:
        print("\n--------------------- 人工测试模式（输入q退出）---------------------")
        while True:
            text = input("输入要判断的内容: ")
            if text in ['', 'q', 'Q']:
                break
            # 文本编码
            inputs = transform(text, tokenizer=tokenizer)

            input_ids = torch.tensor(inputs['input_ids']).unsqueeze(0).to(device)
            attention_mask = torch.tensor(inputs['attention_mask']).unsqueeze(0).to(device)
            token_type_ids = torch.tensor(inputs['token_type_ids']).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(input_ids, attention_mask, token_type_ids)
                predicted = torch.argmax(out.logits, 1).cpu().numpy()[0]
            print(f"预测结果: {predicted}")

    # 仅保存分类头参数
    if not sel.lower().startswith('y'):
        torch.save(model.classifier.state_dict(), save_path)
        print(f"分类头参数已保存到: {save_path}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', '--epoch', type=int, default=60, help='训练次数')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--max_len', type=int, default=50, help='句子最大长度')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--num_labels', type=int, default=2, help='分类类别数（常规/违规为2）')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='权重衰减率')
    parser.add_argument('--dropout_prop', '--dropout', '--drop', type=float, default=0.3, help='丢弃率（浮点数）')
    parser.add_argument('--pretraind_path', type=str, default='bert-base-chinese', help='预训练模型路径')
    parser.add_argument('--save_txt', type=str, default=os.path.join('runs', 'bert-report.txt'), help='训练报告路径')
    parser.add_argument('--human_test', type=bool, default=True, help='是否启用人工测试')
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()

    train_bert_model(**vars(opt), 
                     dataset=os.path.join('dataset', 'chat', 'data.txt'),
                     save_path=os.path.join('models', 'classifier_only.pth'),
                     classifier='normal')
    
    # if not os.path.exists(os.path.join('dataset', 'deepseek')):
    #     os.makedirs(os.path.exists('dataset', 'deepseek'), exist_ok=True)

    # llm_manager = LargeLanguageModelManager(llm_model='deepseek-r1', 
    #                                         config={'llm': {
    #                                             "llm-server": 'remote',
    #                                             "chatglm-api": "",
    #                                             "prompt": r'E:\pandownload1\Projects\Nanoka-Nonebot2\nnk-bot\nnk_bot\plugins\nonebot_plugin_nanokabot_review\dataset\prompt.txt',
    #                                             "llama-path": r'E:\pandownload1\ML\Police\Project\models\DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf'
    #                                         }})

    # with open(os.path.join('dataset', 'chat', 'data.txt'), 'r', encoding='utf-8') as f:
    #     with open(os.path.join('dataset', 'chat', 'cvt.txt'), 'w+', encoding='utf-8') as savefile:
    #         while line := f.readline():
    #             code, content = line.split('\t')
    #             # 由于我们不清楚最后有什么类型的输出，所以我们尽可能准确地进行映射
    #             try:
    #                 code = int(code)
    #                 resp = asyncio.run(llm_manager.ask_function([content]))[0]

    #                 if re.search('think', resp) is not None:
    #                     resp = str(resp.split('think')[-1])
    #                     bias = resp.find('\n')
    #                     if bias < 5:  # 容差
    #                         resp = resp[bias+1:]

    #                 resp = resp.replace('\n', ' ').strip()

    #                 if len(resp) > 300:
    #                     code = -1

    #                 savefile.write("{}\t{}\n".format(code, resp))
    #             except Exception as e:
    #                 print("Exception:", e)
    
    # input("请确认最终训练集 {}\n按 Enter 键继续...".format(os.path.join('dataset', 'chat', 'cvt.txt')))

    # train_bert_model(**vars(opt), 
    #                  dataset=os.path.join('dataset', 'chat', 'data.txt'),
    #                  save_path=os.path.join('models', 'weak_classifier_only.pth'),
    #                  classifier='weak')
