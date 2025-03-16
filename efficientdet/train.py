#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用EfficientDet训练钢材表面缺陷检测模型
基于谷歌官方的AutoML库
"""

import os
import sys
import yaml
import argparse
import tempfile
import subprocess
import shutil
from datetime import datetime

def setup_automl():
    """设置AutoML仓库环境"""
    # 检查是否已经克隆了AutoML仓库
    if not os.path.exists("automl"):
        print("克隆Google AutoML仓库...")
        subprocess.run(["git", "clone", "https://github.com/google/automl.git"], check=True)
    
    # 将AutoML目录添加到PYTHONPATH
    automl_path = os.path.abspath("automl")
    sys.path.append(automl_path)
    os.environ["PYTHONPATH"] = f"{automl_path}:{os.environ.get('PYTHONPATH', '')}"
    
    # 切换到AutoML仓库目录
    os.chdir(automl_path)
    
    print("AutoML仓库设置完成")

def load_config(config_path):
    """加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_hparams_file(config):
    """准备EfficientDet训练的超参数文件
    
    Args:
        config: 配置字典
    
    Returns:
        超参数文件路径
    """
    # 创建临时文件来存储hparams
    fd, hparams_path = tempfile.mkstemp(suffix='.yaml')
    os.close(fd)
    
    # 只提取AutoML库真正支持的参数
    hparams = {
        'train_batch_size': config.get('batch_size', 64),
        'num_examples_per_epoch': config.get('num_examples_per_epoch', 120000),
        'num_epochs': config.get('num_epochs', 50),
        'learning_rate': config.get('learning_rate', 0.08),
        'optimizer': config.get('optimizer', 'sgd'),
        'momentum': config.get('momentum', 0.9),
        'weight_decay': config.get('weight_decay', 4e-5),
        'mixed_precision': config.get('mixed_precision', False),
        'label_map': config.get('label_map', {}),
    }
    
    # 写入临时文件
    with open(hparams_path, 'w') as f:
        yaml.dump(hparams, f)
    
    print(f"已创建超参数文件: {hparams_path}")
    return hparams_path

def train_efficientdet(config_path, pretrained_ckpt=None):
    """使用EfficientDet训练钢材表面缺陷检测模型
    
    Args:
        config_path: 配置文件路径
        pretrained_ckpt: 预训练检查点路径
    """
    # 设置AutoML环境
    setup_automl()
    
    # 加载配置
    config = load_config(config_path)
    
    # 创建必要的目录
    os.makedirs(config['model_dir'], exist_ok=True)
    
    # 构建训练命令 - 根据AutoML源码使用正确的参数
    train_cmd = [
        "python", "efficientdet/main.py",
        "--mode=train_and_eval",
        f"--model_name={config.get('model_name', 'efficientdet-d0')}",
        f"--model_dir={config['model_dir']}",
        f"--train_file_pattern={config['train_file_pattern']}",
        f"--val_file_pattern={config['val_file_pattern']}",
        f"--num_examples_per_epoch={config.get('num_examples_per_epoch', 120000)}",
        f"--train_batch_size={config.get('batch_size', 64)}",
        f"--eval_batch_size={config.get('batch_size', 64)}",
        f"--eval_samples={config.get('eval_samples', 5000)}",
        f"--num_epochs={config.get('num_epochs', 50)}",
        "--hparams=" + f"num_classes={config.get('num_classes', 90)},"
                     + f"learning_rate={config.get('learning_rate', 0.08)},"
                     + f"optimizer={config.get('optimizer', 'sgd')},"
                     + f"momentum={config.get('momentum', 0.9)},"
                     + f"weight_decay={config.get('weight_decay', 4e-5)},"
                     + f"mixed_precision={str(config.get('mixed_precision', False)).lower()}"
    ]
    
    # 如果指定了预训练检查点，添加到命令中
    if pretrained_ckpt:
        train_cmd.append(f"--ckpt={pretrained_ckpt}")
    
    # 使用子进程运行训练
    print("开始EfficientDet训练...")
    print(f"训练命令: {' '.join(train_cmd)}")
    subprocess.run(train_cmd, check=True)
    
    print(f"训练完成，模型保存在 {config['model_dir']}")

def export_model(config_path):
    """导出训练好的EfficientDet模型
    
    Args:
        config_path: 配置文件路径
    """
    # 设置AutoML环境
    setup_automl()
    
    # 加载配置
    config = load_config(config_path)
    
    # 创建导出目录
    os.makedirs(config['export_dir'], exist_ok=True)
    
    # 构建导出命令
    export_cmd = [
        "python", "efficientdet/main.py",
        "--mode=export",
        f"--model_dir={config['model_dir']}",
        f"--export_dir={config['export_dir']}"
    ]
    
    # 使用子进程运行导出
    print("导出模型...")
    subprocess.run(export_cmd, check=True)
    
    print(f"模型导出完成，保存在 {config['export_dir']}")

def evaluate_model(config_path):
    """评估训练好的EfficientDet模型
    
    Args:
        config_path: 配置文件路径
    """
    # 设置AutoML环境
    setup_automl()
    
    # 加载配置
    config = load_config(config_path)
    
    # 创建评估目录
    os.makedirs(config['eval_dir'], exist_ok=True)
    
    # 准备超参数文件
    hparams_path = prepare_hparams_file(config)
    
    # 构建评估命令
    eval_cmd = [
        "python", "efficientdet/main.py",
        "--mode=eval",
        f"--model_dir={config['model_dir']}",
        f"--hparams={hparams_path}"
    ]
    
    # 使用子进程运行评估
    print("评估模型...")
    subprocess.run(eval_cmd, check=True)
    
    # 在评估完成后,删除临时超参数文件
    os.remove(hparams_path)
    
    print("模型评估完成")

def main():
    parser = argparse.ArgumentParser(description="使用EfficientDet训练钢材表面缺陷检测模型")
    parser.add_argument("--dataset", type=str, choices=["neu", "gc10", "combined"], help="数据集类型: neu (NEU-DET), gc10 (GC10-DET), 或 combined (合并数据集)")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "export", "train_and_export"],
                        help="操作模式: train (训练), eval (评估), export (导出), train_and_export (训练并导出)")
    parser.add_argument("--ckpt", type=str, default=None, help="预训练检查点路径，用于迁移学习")
    
    args = parser.parse_args()
    
    # 使用默认配置文件路径(如果没有指定)
    if args.config is None:
        if args.dataset == "neu":
            args.config = "efficientdet/configs/neu_det_config.yaml"
        elif args.dataset == "gc10":
            args.config = "efficientdet/configs/gc10_det_config.yaml"
        elif args.dataset == "combined":
            args.config = "efficientdet/configs/combined_det_config.yaml"
        else:
            parser.error("必须指定--dataset或--config参数")
    
    # 切换回项目根目录(因为setup_automl会切换到automl目录)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # 根据操作模式执行不同操作
    if args.mode == "train":
        train_efficientdet(os.path.join(project_root, args.config), args.ckpt)
    elif args.mode == "eval":
        evaluate_model(os.path.join(project_root, args.config))
    elif args.mode == "export":
        export_model(os.path.join(project_root, args.config))
    elif args.mode == "train_and_export":
        train_efficientdet(os.path.join(project_root, args.config), args.ckpt)
        export_model(os.path.join(project_root, args.config))

if __name__ == "__main__":
    main() 