#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将NEU-DET和GC10-DET数据集转换为COCO格式
支持两种数据集：
1. NEU-DET：东北大学钢材表面缺陷数据集
2. GC10-DET：钢板表面缺陷数据集
"""

import os
import json
import argparse
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import shutil
from tqdm import tqdm
import glob
import random

# NEU-DET类别定义
NEU_CATEGORIES = [
    {"id": 1, "name": "crazing", "supercategory": "defect"},
    {"id": 2, "name": "inclusion", "supercategory": "defect"},
    {"id": 3, "name": "patches", "supercategory": "defect"},
    {"id": 4, "name": "pitted_surface", "supercategory": "defect"},
    {"id": 5, "name": "rolled-in_scale", "supercategory": "defect"},
    {"id": 6, "name": "scratches", "supercategory": "defect"}
]

# NEU-DET类别名称到ID的映射
NEU_CATEGORY_MAP = {
    "crazing": 1,
    "inclusion": 2,
    "patches": 3,
    "pitted_surface": 4,
    "rolled-in_scale": 5,
    "scratches": 6
}

# GC10-DET类别定义
GC10_CATEGORIES = [
    {"id": 1, "name": "punching", "supercategory": "defect"},
    {"id": 2, "name": "weld", "supercategory": "defect"},
    {"id": 3, "name": "crescent_gap", "supercategory": "defect"},
    {"id": 4, "name": "water_spot", "supercategory": "defect"},
    {"id": 5, "name": "oil_spot", "supercategory": "defect"},
    {"id": 6, "name": "silk_spot", "supercategory": "defect"},
    {"id": 7, "name": "inclusion", "supercategory": "defect"},
    {"id": 8, "name": "rolled_pit", "supercategory": "defect"},
    {"id": 9, "name": "crease", "supercategory": "defect"},
    {"id": 10, "name": "waist_folding", "supercategory": "defect"}
]

# GC10-DET类别名称到ID的映射
GC10_CATEGORY_MAP = {
    "punching": 1,
    "weld": 2,
    "crescent_gap": 3,
    "water_spot": 4,
    "oil_spot": 5,
    "silk_spot": 6,
    "inclusion": 7,
    "rolled_pit": 8,
    "crease": 9,
    "waist_folding": 10
}

def convert_neu_det_to_coco(input_path, output_path, train_ratio=0.8, seed=42):
    """将NEU-DET数据集转换为COCO格式
    
    Args:
        input_path: NEU-DET数据集根目录
        output_path: 输出COCO格式JSON文件路径
        train_ratio: 训练集比例，默认0.8
        seed: 随机种子，默认42
    """
    random.seed(seed)
    
    # 检查数据集路径
    images_dir = os.path.join(input_path, "images")
    annotations_dir = os.path.join(input_path, "annotations")
    
    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        raise FileNotFoundError(f"NEU-DET数据集格式错误，请确保有images和annotations文件夹: {input_path}")
    
    # 获取所有图像和标注文件
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    
    if not image_files:
        raise FileNotFoundError(f"没有找到图像文件: {images_dir}")
    
    # 随机打乱并分配训练集和验证集
    random.shuffle(image_files)
    train_count = int(len(image_files) * train_ratio)
    
    train_files = image_files[:train_count]
    val_files = image_files[train_count:]
    
    # 创建COCO格式数据
    train_coco = create_neu_coco_format(train_files, annotations_dir)
    val_coco = create_neu_coco_format(val_files, annotations_dir)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 分别保存训练集和验证集的COCO格式数据
    train_output = output_path.replace(".json", "_train.json")
    val_output = output_path.replace(".json", "_val.json")
    
    with open(train_output, 'w') as f:
        json.dump(train_coco, f)
    
    with open(val_output, 'w') as f:
        json.dump(val_coco, f)
    
    print(f"NEU-DET数据集转换完成: 训练集{len(train_files)}张，验证集{len(val_files)}张")
    return train_output, val_output

def create_neu_coco_format(image_files, annotations_dir):
    """创建NEU-DET数据集的COCO格式
    
    Args:
        image_files: 图像文件列表
        annotations_dir: 标注文件目录
    
    Returns:
        COCO格式数据字典
    """
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": NEU_CATEGORIES
    }
    
    ann_id = 1
    
    for img_id, img_file in enumerate(tqdm(image_files, desc="处理NEU-DET图像")):
        img_filename = os.path.basename(img_file)
        img = Image.open(img_file)
        width, height = img.size
        
        coco_format["images"].append({
            "id": img_id,
            "file_name": img_filename,
            "width": width,
            "height": height
        })
        
        # 查找对应的XML标注文件
        xml_file = os.path.join(annotations_dir, img_filename.replace('.jpg', '.xml'))
        
        if os.path.exists(xml_file):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                category = obj.find('name').text
                category_id = NEU_CATEGORY_MAP.get(category)
                
                if category_id is None:
                    print(f"警告：未知类别 {category} 在文件 {xml_file} 中")
                    continue
                
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                width = xmax - xmin
                height = ymax - ymin
                
                coco_format["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [xmin, ymin, width, height],
                    "area": width * height,
                    "iscrowd": 0
                })
                
                ann_id += 1
    
    return coco_format

def convert_gc10_det_to_coco(input_path, output_path, train_ratio=0.8, seed=42):
    """将GC10-DET数据集转换为COCO格式
    
    Args:
        input_path: GC10-DET数据集路径
        output_path: 输出COCO格式文件路径
        train_ratio: 训练集比例，默认0.8
        seed: 随机种子，默认42
    
    Returns:
        train_output: 训练集输出路径
        val_output: 验证集输出路径
    """
    random.seed(seed)
    
    # 检查数据集路径
    images_dir = os.path.join(input_path, "img")
    annotations_dir = os.path.join(input_path, "ann")
    
    # 验证目录存在
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"没有找到图像目录: {images_dir}")
    if not os.path.exists(annotations_dir):
        raise FileNotFoundError(f"没有找到标注目录: {annotations_dir}")
    
    # 读取meta.json获取类别信息
    meta_file = os.path.join(input_path, "meta.json")
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            meta_data = json.load(f)
            print(f"加载元数据文件: {meta_file}")
    
    # 获取所有图像文件
    all_images = []
    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
    
    for img_file in image_files:
        # 获取图像文件名
        img_filename = os.path.basename(img_file)
        # 构建对应的标注文件名（格式为：图像名.jpg.json）
        ann_filename = img_filename + ".json"  # 如 "img_01_xxx.jpg.json"
        ann_path = os.path.join(annotations_dir, ann_filename)
        
        if os.path.exists(ann_path):
            # 读取标注文件
            try:
                with open(ann_path, 'r') as f:
                    annotation = json.load(f)
                
                # 根据标注文件中的信息确定类别
                # 因缺少具体的标注格式细节，此处使用简化处理
                # 实际应按标注文件格式提取defect_type或类别信息
                # 此处暂时使用默认类别，后续可根据实际情况修改
                category_id = 1  # 默认类别ID
                category_name = "defect"  # 默认类别名
                
                all_images.append((img_file, category_id, category_name))
            except Exception as e:
                print(f"处理标注文件时出错: {ann_path}, 错误: {e}")
    
    if not all_images:
        raise FileNotFoundError(f"没有找到有效的图像和标注文件: {input_path}")
    
    print(f"共找到{len(all_images)}个有效的图像-标注对")
    
    # 随机打乱并分配训练集和验证集
    random.shuffle(all_images)
    train_count = int(len(all_images) * train_ratio)
    
    train_images = all_images[:train_count]
    val_images = all_images[train_count:]
    
    # 创建COCO格式数据
    train_coco = create_gc10_coco_format(train_images)
    val_coco = create_gc10_coco_format(val_images)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 分别保存训练集和验证集的COCO格式数据
    train_output = output_path.replace(".json", "_train.json")
    val_output = output_path.replace(".json", "_val.json")
    
    with open(train_output, 'w') as f:
        json.dump(train_coco, f)
    
    with open(val_output, 'w') as f:
        json.dump(val_coco, f)
    
    print(f"GC10-DET数据集转换完成: 训练集{len(train_images)}张，验证集{len(val_images)}张")
    return train_output, val_output

def create_gc10_coco_format(image_info_list):
    """创建GC10-DET数据集的COCO格式
    
    Args:
        image_info_list: 图像信息列表，包含(图像路径, 类别ID, 类别名称)元组
    
    Returns:
        COCO格式数据字典
    """
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": GC10_CATEGORIES
    }
    
    ann_id = 1
    
    for img_id, (img_file, category_id, _) in enumerate(tqdm(image_info_list, desc="处理GC10-DET图像")):
        img_filename = os.path.basename(img_file)
        img = Image.open(img_file)
        width, height = img.size
        
        coco_format["images"].append({
            "id": img_id,
            "file_name": img_filename,
            "width": width,
            "height": height
        })
        
        # GC10-DET数据集中，整个图像是一个缺陷
        coco_format["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": category_id,
            "bbox": [0, 0, width, height],  # 整个图像范围
            "area": width * height,
            "iscrowd": 0
        })
        
        ann_id += 1
    
    return coco_format

def main():
    parser = argparse.ArgumentParser(description="将钢材表面缺陷数据集转换为COCO格式")
    parser.add_argument("--dataset", type=str, required=True, choices=["neu", "gc10"], 
                        help="数据集类型: neu (NEU-DET) 或 gc10 (GC10-DET)")
    parser.add_argument("--input_path", type=str, required=True, help="输入数据集路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出COCO格式JSON文件路径")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例，默认0.8")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认42")
    
    args = parser.parse_args()
    
    if args.dataset == "neu":
        convert_neu_det_to_coco(args.input_path, args.output_path, args.train_ratio, args.seed)
    elif args.dataset == "gc10":
        convert_gc10_det_to_coco(args.input_path, args.output_path, args.train_ratio, args.seed)

if __name__ == "__main__":
    main() 