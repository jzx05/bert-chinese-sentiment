import os
import json
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """评估配置类"""
    model_path: str = r"E:\Codes\bert-base-chinese\sentiment_model"
    device: str = 'cpu'  # 'cpu' 或 'cuda'
    batch_size: int = 32
    max_length: int = 512
    save_results: bool = True
    output_dir: str = "./evaluation_results"

class SentimentAnalyzer:
    """情感分析器类"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self.label_map = {0: "负面", 1: "正面"}
        
        self._load_model()
    
    def _load_model(self):
        """加载模型和分词器"""
        try:
            logger.info(f"加载模型从: {self.config.model_path}")
            
            # 检查模型路径
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"模型路径不存在: {self.config.model_path}")
            
            # 加载分词器和模型
            self.tokenizer = BertTokenizer.from_pretrained(self.config.model_path)
            self.model = BertForSequenceClassification.from_pretrained(
                self.config.model_path, 
                num_labels=2
            )
            
            # 创建pipeline
            self.classifier = pipeline(
                "text-classification", 
                model=self.model, 
                tokenizer=self.tokenizer, 
                device=self.config.device,
                batch_size=self.config.batch_size
            )
            
            logger.info("模型加载成功！")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def analyze_single_text(self, text: str) -> Dict[str, Any]:
        """分析单个文本"""
        try:
            start_time = time.time()
            result = self.classifier(text)
            
            # 解析结果
            label_id = int(result[0]['label'].replace('LABEL_', ''))
            label = self.label_map[label_id]
            confidence = result[0]['score']
            processing_time = time.time() - start_time
            
            return {
                'text': text,
                'label': label,
                'label_id': label_id,
                'confidence': confidence,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"文本分析失败: {text}, 错误: {e}")
            return {
                'text': text,
                'label': '错误',
                'label_id': -1,
                'confidence': 0.0,
                'processing_time': 0.0,
                'error': str(e)
            }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量分析文本"""
        logger.info(f"开始批量分析 {len(texts)} 个文本...")
        
        results = []
        for i, text in enumerate(texts):
            if i % 100 == 0:
                logger.info(f"进度: {i}/{len(texts)}")
            
            result = self.analyze_single_text(text)
            results.append(result)
        
        logger.info("批量分析完成！")
        return results
    
    def get_performance_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算性能统计"""
        valid_results = [r for r in results if r['label_id'] != -1]
        
        if not valid_results:
            return {}
        
        # 基础统计
        total_texts = len(valid_results)
        avg_confidence = np.mean([r['confidence'] for r in valid_results])
        avg_processing_time = np.mean([r['processing_time'] for r in valid_results])
        
        # 标签分布
        label_counts = {}
        for r in valid_results:
            label = r['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # 置信度分布
        confidence_ranges = {
            '高置信度 (>0.8)': len([r for r in valid_results if r['confidence'] > 0.8]),
            '中置信度 (0.6-0.8)': len([r for r in valid_results if 0.6 <= r['confidence'] <= 0.8]),
            '低置信度 (<0.6)': len([r for r in valid_results if r['confidence'] < 0.6])
        }
        
        return {
            'total_texts': total_texts,
            'avg_confidence': avg_confidence,
            'avg_processing_time': avg_processing_time,
            'label_distribution': label_counts,
            'confidence_distribution': confidence_ranges
        }

def create_test_cases() -> Tuple[List[str], List[str]]:
    """创建测试用例"""
    
    # 基础测试用例
    basic_texts = [
        # 负面情感
        "我今天心情很糟糕",
        "这部电影太无聊了，浪费时间",
        "服务态度很差，不会再来了",
        "产品质量太差了，很失望",
        "今天工作特别累，想休息",
        "这个决定太愚蠢了",
        "交通堵塞让人烦躁",
        "价格太贵了，不值得",
        
        # 正面情感
        "今天天气真好，心情愉快",
        "这家餐厅的菜很好吃，推荐！",
        "学习新知识很有趣",
        "朋友聚会很开心",
        "工作顺利完成，很有成就感",
        "这个想法很棒",
        "风景太美了，让人心旷神怡",
        "孩子很聪明，学习进步很快"
    ]
    
    # 复杂情感测试用例
    complex_texts = [
        "虽然今天下雨了，但是和朋友聊天很开心",
        "这部电影剧情一般，但是演员演技不错",
        "工作很忙很累，但是看到成果很有成就感",
        "价格有点贵，但是质量确实很好",
        "学习很辛苦，但是为了梦想值得",
        "天气不太好，但是心情还不错",
        "路程有点远，但是风景很美",
        "开始有点困难，但是慢慢就适应了"
    ]
    
    return basic_texts, complex_texts

def save_results(results: List[Dict[str, Any]], config: EvaluationConfig):
    """保存评估结果"""
    if not config.save_results:
        return
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 保存详细结果
    results_file = os.path.join(config.output_dir, "evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存性能统计
    stats = SentimentAnalyzer(config).get_performance_stats(results)
    stats_file = os.path.join(config.output_dir, "performance_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"结果已保存到: {config.output_dir}")

def print_results(results: List[Dict[str, Any]], title: str):
    """打印评估结果"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    for i, result in enumerate(results, 1):
        if 'error' in result:
            print(f"{i:2d}. ❌ {result['text']}")
            print(f"    错误: {result['error']}")
        else:
            emoji = "😞" if result['label'] == "负面" else "😊"
            print(f"{i:2d}. {emoji} {result['text']}")
            print(f"    情感: {result['label']}")
            print(f"    置信度: {result['confidence']:.3f}")
            print(f"    处理时间: {result['processing_time']:.3f}s")
        print()

def main():
    """主函数"""
    # 配置
    config = EvaluationConfig()
    
    try:
        # 创建分析器
        analyzer = SentimentAnalyzer(config)
        
        # 获取测试用例
        basic_texts, complex_texts = create_test_cases()
        
        # 分析基础文本
        logger.info("开始基础情感分析...")
        basic_results = analyzer.analyze_batch(basic_texts)
        print_results(basic_results, "基础情感分析结果")
        
        # 分析复杂文本
        logger.info("开始复杂情感分析...")
        complex_results = analyzer.analyze_batch(complex_texts)
        print_results(complex_results, "复杂情感分析结果")
        
        # 计算性能统计
        all_results = basic_results + complex_results
        stats = analyzer.get_performance_stats(all_results)
        
        # 显示性能统计
        print(f"\n{'='*60}")
        print("性能统计")
        print(f"{'='*60}")
        print(f"总文本数: {stats['total_texts']}")
        print(f"平均置信度: {stats['avg_confidence']:.3f}")
        print(f"平均处理时间: {stats['avg_processing_time']:.3f}s")
        print(f"\n标签分布:")
        for label, count in stats['label_distribution'].items():
            print(f"  {label}: {count}")
        print(f"\n置信度分布:")
        for range_name, count in stats['confidence_distribution'].items():
            print(f"  {range_name}: {count}")
        
        # 保存结果
        save_results(all_results, config)
        
        logger.info("评估完成！")
        
    except Exception as e:
        logger.error(f"评估过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()





