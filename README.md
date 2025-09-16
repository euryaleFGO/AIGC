# AIGC
ComfyUI文生图/图生图工作流入门学习
## 项目简介
本项目是一个基于AI的图像生成与处理工具，集成了ComfyUI工作流配置和DeepSeek API调用功能，支持文本生成图像（Text-to-Image）、图像生成图像（Image-to-Image）等AIGC相关任务，适用于动漫风格等图像创作场景。

## 功能特点
- 支持通过ComfyUI工作流配置进行图像生成
- 集成DeepSeek API，可扩展AI对话能力
- 包含多种预设工作流：基础图像生成、动漫风格生成、图生图处理
- 支持LoRA模型加载，可定制化图像风格

## 环境配置

### 前置要求
- Python 3.10
- 相关依赖库（建议使用虚拟环境）
- ComfyUI 环境（用于运行图像生成工作流）

### 安装步骤
1. 克隆本项目到本地
   ```bash
   git clone <项目仓库地址>
   cd AIGC
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt  
   ```

3. 配置环境变量
   - 复制`.env`文件模板并修改
     ```bash
     cp .env.example .env 
     ```
   - 编辑`.env`文件，填写必要信息：
     ```
     DEEPSEEK_API_KEY=你的DeepSeek API密钥
     BASE_URL=https://api.deepseek.com
     MODEL=deepseek-chat
     ```

## 工作流说明
项目`workflow`目录下包含多个预设的ComfyUI工作流配置文件：

1. `API.json`：基础图像生成工作流
   - 使用`waiNSFWIllustrious_v150.safetensors`模型
   - 包含LoRA加载（Marin_Kitagawa角色风格）
   - 分辨率：832x1216

2. `AnimeAPI.json`：动漫风格生成工作流
   - 使用`novaAnimeXL_ilV110.safetensors`模型
   - 专注于动漫角色生成
   - 分辨率：832x1216

3. `image2imageAPI.json`：图生图工作流
   - 支持基于输入图像生成新图像
   - 集成WD14标签器，自动提取图像标签
   - 分辨率：1024x1024

### 运行工作流
将工作流文件导入ComfyUI中即可运行，需确保相关模型文件（.safetensors）已放置在ComfyUI的对应模型目录下。

## 目录结构
```
AIGC/
├── .idea/               # IDE配置文件
├── workflow/            # ComfyUI工作流配置
│   ├── API.json
│   ├── AnimeAPI.json
│   └── image2imageAPI.json
├── config.py            # 配置文件
├── .env                 # 环境变量
└── .gitignore           # Git忽略文件
```

## 注意事项
- 模型文件（.safetensors）需自行获取并正确放置
- DeepSeek API密钥需要从官方平台申请
- 生成图像的质量和风格受提示词、模型和参数影响，建议多次调试
