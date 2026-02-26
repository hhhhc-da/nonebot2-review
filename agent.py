# coding = utf-8
import os
from typing import Literal
from zhipuai import ZhipuAI
from copy import copy
from llama_cpp import Llama

'''
统一接口管理大语言模型的交互

--------------------------------------
                Models
--------------------------------------
- 本地部署的 DeepSeek-R1 模型 (通过 llama-cpp 接口访问)
--------------------------------------

--------------------------------------
                Classes
--------------------------------------
- LargeLanguageModelManager 类: 统一管理大语言模型的接口, 支持模型切换和资源管理
- DeepSeekServe 类: 专门负责与 DeepSeek 模型进行交互, 提供流式输出接口
--------------------------------------

模型统一使用 Llama-2 的聊天格式, 以保证输入输出的一致性和兼容性
请避免占用太多核心数导致其他服务运行出现异常
'''

class LargeLanguageModelManager():
    '''
    大语言模型接口类, 用于与语言模型进行交互
    同时负责管理语言模型相关内容
    '''
    def __init__(self, 
                 llm_model:Literal['deepseek-r1', 'zhipuai']='zhipuai', 
                 config:dict={'llm': {
                     "llm-server": 'remote',
                     "chatglm-api": "",
                     "prompt": r'E:\pandownload1\Projects\Nanoka-Nonebot2\nnk-bot\nnk_bot\plugins\nonebot_plugin_nanokabot_review\dataset\prompt.txt',
                     "llama-path": r'E:\pandownload1\ML\Police\Project\models\DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf'
                     }
                 }):
        self.config = config

        self.llm_model = llm_model
        self.serve = DeepSeekServe() # 如果不使用我们就不创建

        self.api_key = self.config['llm']['chatglm-api']
        self.chatglm_model = ZhipuAI(api_key=self.api_key) # 只是一个网络包封装, 不占用太多内存

        self.ask_function = None

        if self.llm_model == 'zhipuai':
            self.ask_function = self.chatglm_response
            
        elif self.llm_model == 'deepseek-r1':
            self.serve.create_deepseek(
                chat_format='llama-2', 
                llama_path=self.config['llm']['llama-path']
            )
            self.ask_function = self.deepseek_response

        prompt_path = self.config['llm']['prompt']
        self.prompt = []
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                while line := f.readline().strip():
                    if line.startswith("(System)"):
                        self.prompt.append({"role": "system", "content": line[len("(System)"):].strip()})
                    elif line.startswith("(User)"):
                        self.prompt.append({"role": "user", "content": line[len("(User)"):].strip()})
                    elif line.startswith("(DeepSeek)"):
                        self.prompt.append({"role": "assistant", "content": line[len("(DeepSeek)"):].strip()})
                    else:
                        print(f"无法识别的行格式: {line}")

        print(f"成功加载 Prompt 模板: \n{self.prompt}\n")

    def change_llm_model(self, new_model:Literal['deepseek-r1', 'zhipuai']):
        '''
        切换语言模型, 这里会清理当前模型资源, 然后重新初始化新的模型
        '''
        if new_model == self.llm_model:
            print(f"当前已经是 {new_model} 模型, 无需切换")
            return
        
        # 从本地模型切换出去之后要即时回收内存
        if self.llm_model == 'deepseek-r1':
            self.serve.remove_deepseek()
        
        self.llm_model = new_model
        if self.llm_model == 'zhipuai':
            self.ask_function = self.chatglm_response

        elif self.llm_model == 'deepseek-r1':
            self.serve.create_deepseek(
                chat_format='llama-2', 
                llama_path=self.config['llm']['llama-path']
            )
            self.ask_function = self.deepseek_response

    async def deepseek_response(self, texts: list):
        '''
        通过批量生成的问题来逐个询问 DeepSeek
        因为我们后端 DeepSeek 是单线程的, 所以实在没有办法进行多线程优化, 只能逐个生成了
        '''
        reply = []
        extra_ends = '判断这段文本是否违规，如果是，违规内容是什么？如果是，回答“是”，否则回答“否”。'

        for text in texts:
            print("开始询问 DeepSeek:", text + extra_ends)
            
            response = await self.request_deepseek(text + extra_ends)
            response = response.strip()
            reply.append(response)

        return reply
    
    async def request_deepseek(self, question):
        '''
        根据输入的内容进行对话, 这里的对话是单轮的, 不涉及上下文管理
        但是如果后续需要增加上下文管理功能, 可以在这里进行修改, 比如增加一个历史消息列表, 每次对话都将历史消息作为输入的一部分
        '''
        # message = [
        #     {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
        #     {"role": "user", "content": question}
        # ]
        message = copy(self.prompt)
        message.append({"role": "user", "content": question})

        full_response = ""
        print("\n开始生成响应...")
        async for text_chunk in self.serve.generate(messages=message):
            full_response += text_chunk
            print(text_chunk, end='', flush=True)
        print('\n')
        
        return full_response


    def chatglm_response(self, texts: list):
        '''
        通过批量生成的问题来逐个询问
        使用 concurrent.futures 进行并行生成, 提高效率
        '''
        reply = []
        extra_ends = '判断这段文本是否违规，如果是，违规内容是什么？如果是，回答“是”，否则回答“否”。'

        for text in texts:
            print("开始询问 ChatGLM:", text + extra_ends)
            
            # 获取我们的答案, 正则表达式保正确性
            response = self.request_zhipuai_chatglm(text + extra_ends).strip()
            reply.append(response)

            print("ChatGLM 回复信息:", response, "\n")

        return reply

    def request_zhipuai_chatglm(self, content='') -> str:
        """
        请求 ZhipuAI 的 ChatGLM-4-Flash 模型进行对话
        直接返回单句话的回复内容, 不进行任何格式化处理
        """
        if len(content) == 0:
            return False, "内容不能为空"

        message = copy(self.prompt)
        message.append({"role": "user", "content": content})

        response = self.chatglm_model.chat.completions.create(
            model="glm-4-flash",
            messages=message,
        )

        return response.choices[0].message.content
    
class DeepSeekServe():
    '''
    DeepSeek 后端接口类, 使用 llama-cpp 原生的 create_chat_completion 接口
    兼容 ZhipuAI/OpenAI 格式的 messages, 支持流式输出
    '''
    def __init__(self):
        '''
        这里不该插手 prompt 的内容, 只负责提供 Llama 接口
        '''
        self.llama = None
        self.chat_format = None  
        self.llama_path = None

    def create_deepseek(self, 
                        chat_format:Literal['llama-2']='llama-2',
                        llama_path=os.path.join("models", "DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf")
                        ):
        '''
        创建 Llama 模型实例 (使用原生 chat 接口)
        '''
        self.chat_format = chat_format
        self.llama_path = os.path.abspath(llama_path)

        if not os.path.exists(self.llama_path):
            raise FileNotFoundError(f"Llama 模型文件不存在: {self.llama_path}")
        
        self.llama = Llama(
            model_path=self.llama_path,
            n_ctx=512,
            n_threads=2,
            chat_format=self.chat_format,
            verbose=False
        )
        print(f"成功加载 DeepSeek 模型 (chat 格式: {self.chat_format})")

    def remove_deepseek(self):
        '''
        清理模型资源, 保证不占用其他应用的资源
        '''
        if hasattr(self, 'llama') and self.llama is not None:
            del self.llama
            self.llama = None

        self.chat_format = None
        self.llama_path = None

    async def generate(self, messages, temperature:float=0.4, max_tokens:int=512, stop_tokens:list=["<tool_call>"], stream:bool=True):
        '''
        DeepSeek 生成接口
        '''
        if self.llama is None:
            raise RuntimeError("DeepSeek 模型未初始化, 请先调用 create_deepseek")

        try:
            # 调用原生chat接口, stream 表示流式输出
            result = self.llama.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_tokens,
                stream=stream
            )

            if stream:
                full_response = ""
                for chunk in result:
                    if chunk["choices"][0]["finish_reason"] is None:
                        text = chunk["choices"][0]["delta"].get("content", "")
                        if text:
                            full_response += text
                            yield text
            else:
                text = result["choices"][0]["message"]["content"].strip()
                # print("完整响应:", text)
                yield text
                
        except Exception as e:
            print(f"(Deepseek Genarate) Exception: {e}")
            if str(e) == "Close":
                return
            raise
