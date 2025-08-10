# 🚀 BERT 中文情感分析微调项目

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.35.2+-green.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个基于 BERT 预训练模型的中文情感分析微调项目，支持模型训练、评估和推理。该项目使用 Hugging Face Transformers 库，针对中文文本进行情感分析任务。

## ✨ 主要特性

- 🎯 **中文情感分析**: 基于 BERT-base-chinese 预训练模型
- 🚀 **高效训练**: 支持 GPU 加速，包含早停机制
- 📊 **全面评估**: 提供详细的性能指标和可视化结果
- 🔧 **易于使用**: 简洁的 API 接口，支持批量处理
- 📝 **详细日志**: 完整的训练和评估日志记录
- 🎨 **结果可视化**: 使用 matplotlib 和 seaborn 生成美观的图表

## 🏗️ 项目结构

```
fine_tuning/
├── bert-trainer.py          # 模型训练脚本
├── bert-eval.py             # 模型评估和推理脚本
├── requirements.txt          # 项目依赖
├── sentiment_model/         # 训练好的模型文件
├── evaluation_results/      # 评估结果输出
├── results/                 # 训练结果和日志
└── README.md               # 项目说明文档
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于 GPU 加速)
- 至少 8GB 内存

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/jzx05/bert-chinese-sentiment.git
cd bert-chinese-sentiment/fine_tuning

# 安装依赖包
pip install -r requirements.txt
```

### 模型训练

```bash
# 开始训练
python bert-trainer.py
```

训练完成后，模型将保存在 `sentiment_model/` 目录中。

### 模型评估

```bash
# 运行评估
python bert-eval.py
```

评估结果将保存在 `evaluation_results/` 目录中。

## 📚 使用说明

### 训练配置

在 `bert-trainer.py` 中，你可以修改 `TrainingConfig` 类来自定义训练参数：

```python
@dataclass
class TrainingConfig:
    model_name: str = "google-bert/bert-base-chinese"
    num_labels: int = 2                    # 标签数量
    max_length: int = 512                  # 最大序列长度
    batch_size: int = 16                   # 批次大小
    num_epochs: int = 3                    # 训练轮数
    learning_rate: float = 2e-5            # 学习率
    warmup_steps: int = 500                # 预热步数
    weight_decay: float = 0.01             # 权重衰减
```

### 评估配置

在 `bert-eval.py` 中，你可以修改 `EvaluationConfig` 类来自定义评估参数：

```python
@dataclass
class EvaluationConfig:
    model_path: str = "./sentiment_model"  # 模型路径
    device: str = 'cpu'                    # 设备类型 (cpu/cuda)
    batch_size: int = 32                   # 批次大小
    max_length: int = 512                  # 最大序列长度
    save_results: bool = True              # 是否保存结果
    output_dir: str = "./evaluation_results"  # 输出目录
```

### 自定义测试用例

你可以在 `bert-eval.py` 的 `create_test_cases()` 函数中添加自己的测试文本：

```python
def create_test_cases() -> Tuple[List[str], List[str]]:
    # 添加你的测试文本
    custom_texts = [
        "这个产品真的很棒！",
        "服务态度太差了",
        # ... 更多文本
    ]
    return custom_texts, []
```

## 📊 性能指标

项目支持以下评估指标：

- **准确率 (Accuracy)**: 整体预测正确率
- **精确率 (Precision)**: 预测为正例中实际为正例的比例
- **召回率 (Recall)**: 实际正例中被正确预测的比例
- **F1 分数**: 精确率和召回率的调和平均数
- **置信度分布**: 模型预测的置信度统计
- **处理时间**: 单条文本的平均处理时间

## 🔧 高级功能

### GPU 加速

如果你的机器有 NVIDIA GPU，可以通过以下方式启用 GPU 加速：

```python
# 在评估配置中设置
config.device = 'cuda'
```

### 批量处理

支持批量文本处理，提高推理效率：

```python
# 批量分析文本
texts = ["文本1", "文本2", "文本3"]
results = analyzer.analyze_batch(texts)
```

### 早停机制

训练过程中包含早停机制，防止过拟合：

```python
from transformers import EarlyStoppingCallback

callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
```

## 📈 结果示例

训练完成后，你将看到类似以下的输出：

```
============================================================
                    基础情感分析结果
============================================================
 1. 😊 今天天气真好，心情愉快
    情感: 正面
    置信度: 0.923
    处理时间: 0.045s

 2. 😞 这部电影太无聊了，浪费时间
    情感: 负面
    置信度: 0.891
    处理时间: 0.038s
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Hugging Face](https://huggingface.co/) - 提供优秀的 Transformers 库
- [Google Research](https://research.google/) - BERT 预训练模型
- [PyTorch](https://pytorch.org/) - 深度学习框架

## 📞 联系方式

- 项目主页: [GitHub Repository](https://github.com/jzx05/bert-chinese-sentiment)
- 问题反馈: [Issues](https://github.com/jzx05/bert-chinese-sentiment/issues)

---

⭐ 如果这个项目对你有帮助，请给它一个星标！
