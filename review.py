#coding=utf-8
import torch
import os
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from .dataset.dataset import transform
from .bert import LastClassifier, LastWeakClassifier
from .agent import LargeLanguageModelManager
import asyncio
import re

class Review():
    def __init__(self, 
                 split_length:int=40, 
                 pretraind_path='bert-base-chinese',
                 classifier_path=r'E:\pandownload1\Projects\Nanoka-Nonebot2\nnk-bot\nnk_bot\plugins\nonebot_plugin_nanokabot_review\models\classifier_only.pth',
                 weak_classifier_path=r'E:\pandownload1\Projects\Nanoka-Nonebot2\nnk-bot\nnk_bot\plugins\nonebot_plugin_nanokabot_review\models\weak_classifier_only.pth'):
        '''
        一键管理我们的审查行为, 用于管理我们的一个 Deepseek 后端和两个 Bert 模型
        使用 GPU 计算的 Bert 速度还是很可观的, 但是 Deepseek 则使用 CPU 虽然 4 线程还是有点吃力
        而且 Deepseek R1 效果说实在一般，我建议还是姑且做一些简单的算法环节吧
        '''
        self._length = split_length
        self.llm_manager = LargeLanguageModelManager(llm_model='deepseek-r1', 
                                                     config={'llm': {
                                                        "llm-server": 'remote',
                                                        "chatglm-api": "",
                                                        "prompt": r'E:\pandownload1\Projects\Nanoka-Nonebot2\nnk-bot\nnk_bot\plugins\nonebot_plugin_nanokabot_review\dataset\prompt.txt',
                                                        "llama-path": r'E:\pandownload1\ML\Police\Project\models\DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf'
                                                        }})
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"模型运行在 {self.device} 上")

        self.pretraind_path = pretraind_path
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretraind_path)
        self.bert_config = BertConfig.from_pretrained(
            pretraind_path,
            num_labels=2,
            hidden_dropout_prob=0.3
        )
        self.bert_model = BertForSequenceClassification.from_pretrained(pretraind_path, config=self.bert_config).to(self.device)
        self.bert_model.classifier = LastClassifier(hidden_dim=768, output_dim=2).to(self.device)

        # 标准预测模型, 端到端版本, 还有一个是对接 DeekSeek 的后期处理版本
        self.bert_model.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
        self.bert_model.eval()

        # 简易模型, 用于对接 LLM
        self.weak_bert_model = BertForSequenceClassification.from_pretrained(pretraind_path, config=self.bert_config).to(self.device)
        self.weak_bert_model.classifier = LastWeakClassifier(output_dim=2).to(self.device)

        if os.path.exists(weak_classifier_path):
            self.weak_bert_model.classifier.load_state_dict(torch.load(weak_classifier_path, map_location=self.device))
        else:
            # 没有预训练? 随便了，反正我也不用 Deepseek R1
            sample_weights = torch.zeros(size=(2, 768), device=self.device)
            sample_weights[0][0] = 1
            sample_weights[1][1] = 1
            # 我没找到 OrderMap 定义, 凑合用字典了, 将计算图补全, 目前是取的后 766 个维度为切割平面
            torch_dict_map = {
                'fn1.weight': sample_weights,
                'fn1.bias': torch.zeros(size=(2,), device=self.device)
            }
            self.weak_bert_model.classifier.load_state_dict(torch_dict_map)

        self.weak_bert_model.eval()

    @torch.no_grad()
    def bert_predict(self, text, max_len=512):
        '''
        使用 Bert 进行文本分类
        这个表现还凑合，就是样本太少了，长篇广告识别的效果还可以
        其他的黑话什么的还是要往里加，不然不能识别出来

        汉语谐音识别说实话真应该用拼音描述吧 (((
        '''
        if len(text) > max_len:
            text = text[:max_len] # 截断设计, 不过三百多字还没写完也是神人了
        inputs = transform(text, tokenizer=self.bert_tokenizer)

        input_ids = torch.tensor(inputs['input_ids']).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(inputs['attention_mask']).unsqueeze(0).to(self.device)
        token_type_ids = torch.tensor(inputs['token_type_ids']).unsqueeze(0).to(self.device)

        out = self.bert_model(input_ids, attention_mask, token_type_ids)
        predicted = torch.argmax(out.logits, 1).cpu().numpy()[0]
        return predicted
    
    @torch.no_grad()
    def deepseek_predict(self, text, max_len=512):
        '''
        使用 Deepseek + Bert 进行文本分类
        说实话这个效果真的很烂, 基本没眼看
        '''
        if len(text) > max_len:
            text = text[:max_len]

        resp = asyncio.run(self.llm_manager.ask_function([text]))[0]
        # 在此之前, 你要洗数据, 将思考数据转换为输出数据
        if re.search('think', resp) is not None:
            resp = str(resp.split('think')[-1])
            bias = resp.find('\n')
            if bias < 5:  # 容差
                resp = resp[bias+1:]

        resp = resp.replace('\n', ' ').strip()

        inputs = transform(resp, tokenizer=self.bert_tokenizer)

        input_ids = torch.tensor(inputs['input_ids']).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(inputs['attention_mask']).unsqueeze(0).to(self.device)
        token_type_ids = torch.tensor(inputs['token_type_ids']).unsqueeze(0).to(self.device)

        out = self.weak_bert_model(input_ids, attention_mask, token_type_ids)
        predicted = torch.argmax(out.logits, 1).cpu().numpy()[0]
        return predicted

    def func(self, text: list):
        '''
        我们根据句子长度决定是否使用 LLM 去判断
        较长的句子一般也不会太频繁的请求, 所以怎么处理归最终部署管理即可
        为了防止逆天使用超长模型还是加了判断
        '''
        ret = None
        if len(text) > self._length:
            print("使用 Deepseek + Bert 判断")
            ret = self.deepseek_predict(text=text, max_len=512)
        else:
            print("使用 Bert 进行判断")
            # min 一下, 谁知道有没有人用逆天模型
            ret = self.bert_predict(text=text, max_len=min(512, self._length))
        return ret

if __name__ == '__main__':
    '''
    主函数测试 Deepseek 和 Bert 的工作状态
    墓前一切良好, 除了 Deepseek 的输出经常抽风
    如果你有 chatglm 的付费 API 还是可以考虑使用的
    '''
    x = Review(split_length=80)

    while line := input("输入需要识别的内容 (q 键结束):"):
        if line.strip() == 'q':
            break

        text = x.func(line)
        print("输出:", text)

    del x