import os
import re
import logging
from typing import Dict, Any
from dataclasses import dataclass

import torch
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """训练配置类"""
    model_name: str = "google-bert/bert-base-chinese"  # 使用google-bert/bert-base-chinese模型
    num_labels: int = 2
    max_length: int = 512
    batch_size: int = 16
    eval_batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    output_dir: str = "./results"
    model_save_dir: str = "./sentiment_model"
    seed: int = 42

def set_seed(seed: int) -> None:
    """设置随机种子确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def clean_text(text: str) -> str:
    """清洗文本数据"""
    if not isinstance(text, str):
        return ""
    
    # 保留中文、英文、数字和基本标点
    text = re.sub(r'[^\u4e00-\u9fff\w\s，。！？；：""''（）【】]', '', text)
    text = re.sub(r'\s+', ' ', text)  # 合并多个空格
    return text.strip()

def tokenize_function(examples: Dict[str, Any], tokenizer, max_length: int) -> Dict[str, Any]:
    """分词函数"""
    return tokenizer(
        examples['text'], 
        padding='max_length', 
        truncation=True, 
        max_length=max_length,
        return_tensors=None  # 确保返回列表而不是张量
    )

def compute_metrics(eval_pred) -> Dict[str, float]:
    """计算评估指标"""
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    labels = eval_pred.label_ids
    
    # 基础准确率
    accuracy = accuracy_score(labels, predictions)
    
    # 精确率、召回率、F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    """主训练函数"""
    config = TrainingConfig()
    
    # 设置随机种子
    set_seed(config.seed)
    
    logger.info("开始BERT中文情感分析模型训练")
    logger.info(f"配置: {config}")
    
    try:
        # 1. 加载模型和分词器
        logger.info("加载模型和分词器...")
        tokenizer = BertTokenizer.from_pretrained(config.model_name)
        model = BertForSequenceClassification.from_pretrained(
            config.model_name, 
            num_labels=config.num_labels
        )
        
        # 2. 加载数据集
        logger.info("加载ChnSentiCorp数据集...")
        dataset = load_dataset('lansinuote/ChnSentiCorp')  
        logger.info(f"数据集大小: 训练集={len(dataset['train'])}, 验证集={len(dataset['test'])}")
        
        # 3. 数据预处理
        logger.info("开始数据预处理...")
        dataset = dataset.map(
            lambda x: {'text': clean_text(x['text'])}, 
            desc="清洗文本"
        )
        
        # 过滤空文本
        dataset = dataset.filter(lambda x: len(x['text']) > 0)
        logger.info(f"预处理后数据集大小: 训练集={len(dataset['train'])}, 验证集={len(dataset['test'])}")
        
        # 4. 分词处理
        logger.info("开始分词处理...")
        encoded_dataset = dataset.map(
            lambda x: tokenize_function(x, tokenizer, config.max_length),
            batched=True,
            desc="分词处理"
        )
        
        # 5. 设置训练参数
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.eval_batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=100,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to=None,  # 禁用wandb等外部工具
            dataloader_pin_memory=False,  # Windows兼容性
        )
        
        # 6. 创建训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset['train'],
            eval_dataset=encoded_dataset['test'],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # 7. 开始训练
        logger.info("开始训练...")
        trainer.train()
        
        # 8. 评估模型
        logger.info("评估模型性能...")
        eval_results = trainer.evaluate()
        logger.info(f"最终评估结果: {eval_results}")
        
        # 9. 保存模型
        logger.info(f"保存模型到 {config.model_save_dir}...")
        os.makedirs(config.model_save_dir, exist_ok=True)
        model.save_pretrained(config.model_save_dir)
        tokenizer.save_pretrained(config.model_save_dir)
        
        # 保存训练配置
        import json
        with open(os.path.join(config.model_save_dir, 'training_config.json'), 'w', encoding='utf-8') as f:
            json.dump(config.__dict__, f, ensure_ascii=False, indent=2)
        
        logger.info("训练完成！模型已保存")
        
        # 10. 显示最终结果
        print("\n" + "="*50)
        print("训练完成！")
        print(f"模型保存位置: {config.model_save_dir}")
        print(f"最终准确率: {eval_results.get('eval_accuracy', 'N/A'):.4f}")
        print(f"最终F1分数: {eval_results.get('eval_f1', 'N/A'):.4f}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()