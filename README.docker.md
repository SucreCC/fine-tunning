# Docker 使用说明

本项目提供了 Docker 和 Docker Compose 配置，方便在不同环境中运行模型微调任务。

## 前置要求

1. **Docker** 和 **Docker Compose** 已安装
2. **GPU 支持**（如果使用 GPU）：
   - 安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
   - 确保 Docker 可以访问 GPU：`docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

## 快速开始

### 1. GPU 训练（推荐）

```bash
# 构建镜像
docker-compose -f docker-compose.gpu.yml build

# 启动训练
docker-compose -f docker-compose.gpu.yml up

# 后台运行
docker-compose -f docker-compose.gpu.yml up -d

# 查看日志
docker-compose -f docker-compose.gpu.yml logs -f

# 停止容器
docker-compose -f docker-compose.gpu.yml down
```

### 2. CPU 训练（仅用于测试）

```bash
# 构建镜像
docker-compose -f docker-compose.cpu.yml build

# 启动训练
docker-compose -f docker-compose.cpu.yml up
```

### 3. 使用默认配置（自动检测 GPU）

```bash
# 构建镜像
docker-compose build

# 启动训练
docker-compose up
```

## 环境变量配置

### GPU 相关

```bash
# 指定使用的 GPU（例如：只使用 GPU 0 和 1）
export CUDA_VISIBLE_DEVICES=0,1
docker-compose up
```

### WandB 配置

```bash
# 设置 WandB API Key
export WANDB_API_KEY=your_api_key_here
docker-compose up
```

## 目录挂载说明

Docker Compose 会自动挂载以下目录：

- `./model` → `/app/model` (只读) - 模型文件
- `./dataset` → `/app/dataset` (只读) - 数据集
- `./outputs` → `/app/outputs` (读写) - 训练输出和检查点
- `./logs` → `/app/logs` (读写) - 日志文件
- `./config.yaml` → `/app/config.yaml` (只读) - 配置文件
- `./wandb` → `/app/wandb` (读写) - WandB 数据

## 配置文件修改

修改 `config.yaml` 后，需要重启容器：

```bash
docker-compose restart
```

或者重新创建容器：

```bash
docker-compose down
docker-compose up
```

## 进入容器调试

```bash
# 进入运行中的容器
docker-compose exec fine-tuning bash

# 或者启动一个新的交互式容器
docker-compose run --rm fine-tuning bash
```

## 查看训练进度

```bash
# 实时查看日志
docker-compose logs -f fine-tuning

# 查看最近的日志
docker-compose logs --tail=100 fine-tuning
```

## 常见问题

### 1. GPU 不可用

如果遇到 GPU 相关错误，检查：
- NVIDIA 驱动是否安装
- NVIDIA Container Toolkit 是否安装
- Docker 是否有权限访问 GPU

### 2. 内存不足

如果遇到 OOM（内存不足）错误：
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 启用量化（`quantization.enable: true`）

### 3. 模型文件太大

如果模型文件太大，可以：
- 使用模型挂载卷（NFS、S3 等）
- 在容器内下载模型（修改 Dockerfile）

## 生产环境建议

1. **使用 Docker Swarm 或 Kubernetes** 进行多节点训练
2. **使用共享存储**（NFS、S3）存储模型和数据集
3. **配置资源限制**（CPU、内存、GPU）
4. **设置日志轮转**和监控
5. **使用 CI/CD** 自动化构建和部署

## 清理

```bash
# 停止并删除容器
docker-compose down

# 删除镜像
docker rmi fine-tuning:latest

# 清理所有未使用的资源
docker system prune -a
```

