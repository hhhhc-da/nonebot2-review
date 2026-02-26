#coding: utf-8
from typing import Annotated
from nonebot import get_plugin_config
from nonebot.plugin import PluginMetadata

from nonebot import on_command, on_fullmatch, on_regex, on_message, require
from nonebot.adapters.onebot.v11 import (
    GROUP,
    GROUP_ADMIN,
    GROUP_OWNER,
    GroupMessageEvent,
    Message,
    MessageSegment,
)
from nonebot.log import logger
from nonebot.matcher import Matcher
from nonebot.params import CommandArg, Depends, RegexStr
from nonebot.permission import SUPERUSER

from .config import Config
from .review import Review

__help_version__ = "v0.1.0"
__help_usages__ = f"""
[帮助] 查看 NanokaBot 的所有指令内容""".strip()

__plugin_meta__ = PluginMetadata(
    name="nonebot_plugin_nanokabot_help",
    description="用来返回可以的帮助信息",
    usage=__help_usages__,
    config=Config,
)

cfg = get_plugin_config(Config)
reviewer = Review(split_length=800) # 800 字分割, 实际上肯定达不到

review_handle = on_message(priority=99, block=False, permission=GROUP)

@review_handle.handle()
async def handle_all_msg(event: GroupMessageEvent):
    '''
    内容审查响应函数, 仅在需要回复时回复
    '''
    sender_id = event.get_user_id()
    msg_content = event.get_plaintext().strip()
    group_id = event.group_id
    print(type(group_id))

    if not (group_id == 1043289075):
        print("非有效群聊消息")
        return 

    if not msg_content:
        return 

    print(f"【内容审查】发送者：{sender_id} | 群号：{group_id} | 内容：{msg_content}")

    is_illegal = reviewer.func(msg_content)

    if is_illegal != 0:
        reply_msg = Message(f"⚠️ 检测到违规内容：{msg_content}\n来自用户 {sender_id}")
        await review_handle.send(reply_msg)

    return
