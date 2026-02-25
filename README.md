# 内容审查机器人
使用 nonebot 的内容审查 bot，基于 Bert 和 Llama

下载 bert-base-chinese 的时候建议用镜像站

```
git clone https://hf-mirror.com/google-bert/bert-base-chinese
```

在此之前你需要先安装好 `requirements.txt` 内的所有内容到你的 `.venv` 环境内

需要自己打包成文件夹 `nonebot_plugin_{这里填你要起的名字}` 然后放进 `plugin` 文件夹就可以了

`nb create` 之后会自动帮你创建 `plugin` 文件夹的
