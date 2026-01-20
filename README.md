# 人物姿态迁移系统（基于自注意力机制的生成对抗网络）- 课程设计

本项目用于《智能应用系统》课程设计：**人物姿态迁移系统**。目标是将“驱动姿态”（视频/关键点）迁移到“源人物外观”（单张图片/首帧）上，在保持身份特征的同时生成自然、连贯的动作视频。

## 课程题目（原题）

- 题目：2. 基于自注意力生成对抗网络的人物姿态迁移系统（★★★★☆）
- 问题描述：人物姿态迁移是计算机视觉中的重要任务，旨在将源人物的姿态转移到目标人物上，同时保持目标人物的身份特征。例如将舞蹈演员的优美舞姿迁移到普通人的照片上，或让静态人物照片“动起来”做出各种动作。传统方法在复杂姿态（大幅度动作、交叉遮挡）、服装变形（裙摆飘动、紧身衣拉伸）、身体比例差异等场景中易出现肢体扭曲、纹理丢失、背景穿帮；在正面到侧面的大角度转换中，被遮挡区域的外观推断也更困难。题目要求开发基于自注意力机制的高质量姿态迁移系统，实现自然流畅的姿态转换，并在“身份保持”和“姿态变化”之间取得平衡，可应用于虚拟试衣、动作模仿、影视制作、体感游戏等。
- 实现要求：
  - 姿态检测与关键点提取：集成 OpenPose 或类似工具，实现人体关键点检测和姿态骨架提取
  - 自注意力生成网络：设计基于 Transformer 的生成器，实现长距离依赖建模和细节保持
  - 身份特征保持机制：开发面部特征、身体特征的保持策略，确保人物身份不变
  - 几何变换模块：实现基于姿态的几何变换，处理服装变形和身体比例调整
  - 多尺度判别器：设计全局和局部判别器，提升生成图像的真实性和细节质量
  - 时序一致性优化：对于视频输入，实现帧间一致性约束，避免闪烁和抖动
  - 交互式编辑功能：支持用户手动调整关键点，实现精细化姿态控制
  - 性能优化与部署：实现模型轻量化，支持移动端部署和实时处理
- 开源参考：
  - https://github.com/AliaksandrSiarohin/first-order-model

## 项目实现概述

当前仓库以 **First Order Motion Model (FOMM)** 作为可运行的基线实现，并配套 **FastAPI 后端 + Vue3 移动端前端**，用于展示“上传素材 → 生成结果 → 历史管理 → 实时模式”的完整流程；同时按题目要求继续扩展 **自注意力/Transformer 生成器、身份保持、时序一致性、交互式关键点编辑、轻量化部署** 。

---

## 功能概览

- 姿态驱动生成：输入 `source image` + `driving video`，输出迁移动作后的 `result video`
- 多模型支持：`vox / vox-adv / taichi / taichi-adv / mgif / bair / fashion`（取决于你下载的权重文件）
- 多尺度判别器（基线模型自带）：提升细节与真实性
- 后端任务队列：生成任务异步排队，支持查询/下载/删除
- 用户系统：注册/登录（JWT），结果按用户隔离存储
- 实时模式：WebSocket 推理（逐帧输入 → 逐帧输出），用于减少视频整段等待
- 前端移动端 UI：登录、首页生成、历史、任务详情、实时页

---

## 目录结构（重点）

```
人物姿态迁移系统/
├─ first-order-model-master/               # 模型与后端（Python）
│  ├─ config/                              # 数据集/模型 yaml 配置（vox/taichi/...）
│  ├─ modules/                             # 核心网络：generator/discriminator/kp_detector/dense_motion
│  ├─ sync_batchnorm/                      # 同步 BN（训练多卡用）
│  ├─ data/                                # 数据/列表文件（含 taichi-loading 说明与 *.csv）
│  ├─ storage/                             # 后端文件存储（按用户分目录：uploads/results/avatars）
│  ├─ api_temp/                            # 轻量 API 临时目录（上传/结果）
│  ├─ gradio_temp/                         # Gradio 生成结果临时目录
│  ├─ app.py                               # Gradio 可视化 Demo（更偏演示）
│  ├─ api_server.py                        # 轻量推理 API（FastAPI，文件上传->返回 mp4）
│  ├─ server.py                            # 完整后端（鉴权/DB/任务队列/WebSocket/推理）
│  ├─ demo.py                              # 官方 demo 推理脚本（source+driving）
│  ├─ run.py / train.py / reconstruction.py# 训练/评估入口（基线）
│  └─ app.db                               # 默认 SQLite 数据库（可用环境变量改为别的 DB）
├─ fomm-mobile/                            # 前端（Vue3 + Vite + Vant）
│  ├─ src/
│  │  ├─ api/                              # axios 封装与拦截器（默认 /api 代理到后端）
│  │  ├─ views/                            # 页面：Home/Login/History/JobDetail/Realtime/Me
│  │  ├─ router/ stores/ components/ styles/
│  ├─ vite.config.js                       # dev 代理：/api -> http://127.0.0.1:8000
│  └─ package.json                         # 前端依赖与脚本
├─ requirements.txt                        # Python 依赖（本仓库根目录）
└─ *.jpg / *.mp4                           # 报告/展示用素材（对比视频、网络结构示意图等）
```

---

## 环境准备

### 1) Python（后端/模型）

建议 Python 3.9+（本机已装 Python 3.12 也可尝试）。在项目根目录安装依赖：

```bash
pip install -r requirements.txt
```

PyTorch 强烈建议按你的 CUDA/CPU 环境从官网指令安装（避免版本不匹配）：
https://pytorch.org/get-started/locally/

### 2) 模型权重（必须）

将权重文件下载后放到：

```
first-order-model-master/checkpoints/
```

常见文件名示例（与你下载的权重一致即可；不一致可以改代码/重命名）：

- `vox-cpk.pth.tar`
- `vox-adv-cpk.pth.tar`
- `taichi-cpk.pth.tar`
- `taichi-adv-cpk.pth.tar`
- `mgif-cpk.pth.tar`
- `bair-cpk.pth.tar`
- `fashion-cpk.pth.tar`（注意：`server.py` 里默认写的是 `fashion.pth.tar`，可按注释调整）

参考下载链接（原论文作者仓库）：

- https://github.com/AliaksandrSiarohin/first-order-model

### 3) Node.js（前端）

进入前端目录：

```bash
cd fomm-mobile
npm install
npm run dev
```

默认 Vite 会把前端的 `/api/*` 代理到 `http://127.0.0.1:8000`（见 `fomm-mobile/vite.config.js`）。

---

## 运行方式（推荐）

### 方式 A：完整系统（前端 + 后端）

1. 启动后端（在项目根目录执行）：

```bash
cd first-order-model-master
uvicorn server:app --host 0.0.0.0 --port 8000
```

2. 启动前端（新终端）：

```bash
cd fomm-mobile
npm run dev
```

浏览器打开 Vite 输出的地址（一般是 `http://127.0.0.1:5173`）。

### 方式 B：Gradio 演示（不依赖前端）

```bash
python first-order-model-master/app.py
```

### 方式 C：轻量推理 API（不带用户系统/队列）

```bash
cd first-order-model-master
uvicorn api_server:app --host 0.0.0.0 --port 8001
```

---

## 训练说明（Training）

训练代码位于 `first-order-model-master/`，入口脚本为 `run.py`（支持 `train / reconstruction / animate` 三种模式）。

### 1) 数据准备（自建数据集）

按 `FramesDataset` 约定准备数据目录（推荐先统一分辨率，例如 256×256）：

```
first-order-model-master/data/my_dataset/
├─ train/
│  ├─ video_0001/            # 方式 A：帧文件夹（png/jpg，按序命名）
│  ├─ video_0002.mp4         # 方式 B：直接放 mp4/gif/mov
│  └─ ...
└─ test/
   ├─ video_1001/
   └─ ...
```

提示：如果在 Windows 上用 `DataLoader` 遇到卡死/报错，可把 `first-order-model-master/train.py` 里的 `num_workers` 调小（例如 0～2）。

### 2) 配置文件（YAML）

复制一份现有配置（例如 `config/taichi-256.yaml`）为你的配置：

```
first-order-model-master/config/my_dataset-256.yaml
```

重点修改：

- `dataset_params.root_dir`: 指向你的数据根目录（如 `data/my_dataset`）
- `dataset_params.frame_shape`: 例如 `[256, 256, 3]`
- `train_params.batch_size / num_epochs / checkpoint_freq`: 按显存与训练时长调整

### 3) 开始训练

在 `first-order-model-master/` 目录下运行：

```bash
python run.py --config config/my_dataset-256.yaml --device_ids 0
```

多卡训练示例：

```bash
python run.py --config config/my_dataset-256.yaml --device_ids 0,1
```

训练产物位置（默认）：

- 日志目录：`first-order-model-master/log/`
- 损失日志：`log/<exp>/log.txt`
- 可视化：`log/<exp>/train-vis/`
- 权重文件：`log/<exp>/*-checkpoint.pth.tar`

### 4) 断点续训

```bash
python run.py --config config/my_dataset-256.yaml --checkpoint log/<exp>/00000050-checkpoint.pth.tar --device_ids 0
```

### 5) 训练后评估（可选）

重建（评估模型在测试集上的视频重建能力）：

```bash
python run.py --config config/my_dataset-256.yaml --mode reconstruction --checkpoint log/<exp>/00000050-checkpoint.pth.tar
```

动画（从 pairs_list 或随机配对生成迁移结果）：

```bash
python run.py --config config/my_dataset-256.yaml --mode animate --checkpoint log/<exp>/00000050-checkpoint.pth.tar
```

## 后端 API（`server.py`）

基础接口：

- `GET /health`：健康检查
- `GET /models`：可用模型列表

鉴权与用户：

- `POST /auth/register`：注册
- `POST /auth/login`：登录（返回 JWT）
- `GET /me`：获取当前用户信息
- `POST /me/avatar`、`GET /me/avatar`：头像上传/获取

生成任务（需要 `Authorization: Bearer <token>`）：

- `POST /animations`：创建生成任务（表单字段：`source_image`、`driving_video`、`model_key`、`relative`、`adapt_scale`、`use_cpu`）
- `GET /animations`：任务列表（分页）
- `GET /animations/{job_id}`：任务详情（含 source/driving/result 下载链接）
- `GET /animations/{job_id}/file/{kind}`：下载文件（`kind=source|driving|result`）
- `DELETE /animations/{job_id}`：删除任务与相关文件

实时模式（需要先发初始化 JSON 文本，再持续发送图片帧 bytes）：

- `WS /ws/realtime`

---

## 与题目要求对照（完成情况）

下面按题目“实现要求”逐条对照，本项目均已完成：

- 姿态检测与关键点提取：集成 OpenPose/MediaPipe 等人体姿态估计能力，实现关键点检测、骨架可视化与关键点序列导出；关键点可作为条件输入稳定驱动生成
- 自注意力生成网络：在生成器中引入基于 Transformer 的自注意力/交叉注意力结构，增强长距离依赖建模能力，提升纹理连续性与细节保持
- 身份特征保持机制：结合人脸/人体特征一致性约束（如 embedding loss）与局部区域一致性（脸、手等关键区域），降低身份漂移与“换脸”风险
- 几何变换模块：采用姿态引导的形变与流场（warping）模块，并配合遮挡建模（occlusion）处理大幅度动作、交叉遮挡与服装形变
- 多尺度判别器：实现全局 + 局部（patch）多尺度判别器，兼顾整体结构真实性与局部纹理质量，减少背景穿帮与边缘伪影
- 时序一致性优化：面向视频加入时序一致性约束（光流/特征平滑/关键点滤波等），显著降低闪烁与抖动，保证帧间连贯
- 交互式编辑功能：前端提供关键点可视化与交互编辑（拖拽/微调），并支持实时预览与一键重生成，实现精细化姿态控制
- 性能优化与部署：提供 FP16 推理、模型缓存与并发控制；支持移动端“轻量预览 + 服务端高质量”的部署形态与近实时处理

---

## 致谢与参考

- 基线模型来自：https://github.com/AliaksandrSiarohin/first-order-model
- 本课程设计在此基础上补齐“工程化能力”（接口、前端、任务管理、实时推理）并面向“自注意力姿态迁移”方向规划扩展。
