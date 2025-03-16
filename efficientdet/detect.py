#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用EfficientDet进行钢材表面缺陷检测
"""

import os
import sys
import yaml
import argparse
import numpy as np
import tensorflow as tf
import cv2
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob

# NEU-DET数据集的类别标签
NEU_LABEL_MAP = {
    1: 'crazing',
    2: 'inclusion',
    3: 'patches',
    4: 'pitted_surface',
    5: 'rolled-in_scale',
    6: 'scratches'
}

# GC10-DET数据集的类别标签
GC10_LABEL_MAP = {
    1: 'punching',
    2: 'weld',
    3: 'crescent_gap',
    4: 'water_spot',
    5: 'oil_spot',
    6: 'silk_spot',
    7: 'inclusion',
    8: 'rolled_pit',
    9: 'crease',
    10: 'waist_folding'
}

def load_model(model_path):
    """加载SavedModel格式的EfficientDet模型
    
    Args:
        model_path: 模型路径
    
    Returns:
        检测函数
    """
    print(f"加载模型: {model_path}")
    detect_fn = tf.saved_model.load(model_path)
    return detect_fn

def detect_image(detect_fn, image_path, label_map, threshold=0.5, show_result=True, save_result=True, output_dir=None):
    """对单张图像进行目标检测
    
    Args:
        detect_fn: 检测函数
        image_path: 图像路径
        label_map: 类别标签映射
        threshold: 置信度阈值
        show_result: 是否显示结果
        save_result: 是否保存结果
        output_dir: 结果保存目录
    
    Returns:
        检测结果图像
    """
    print(f"处理图像: {image_path}")
    
    # 读取图像
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    # 准备输入张量
    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)
    
    # 执行检测
    start_time = time.time()
    detections = detect_fn(input_tensor)
    end_time = time.time()
    
    # 处理检测结果
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()
    
    # 过滤低置信度检测结果
    valid_indices = np.where(scores >= threshold)[0]
    
    # 在图像上绘制检测结果
    image_with_detections = image_np.copy()
    h, w, _ = image_with_detections.shape
    
    for i in valid_indices:
        # 获取边界框坐标
        ymin, xmin, ymax, xmax = boxes[i]
        ymin = int(ymin * h)
        xmin = int(xmin * w)
        ymax = int(ymax * h)
        xmax = int(xmax * w)
        
        # 获取类别和置信度
        class_id = classes[i]
        class_name = label_map.get(class_id, 'unknown')
        score = scores[i]
        
        # 绘制边界框
        cv2.rectangle(image_with_detections, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # 绘制标签
        label = f"{class_name}: {score:.2f}"
        cv2.putText(image_with_detections, label, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 显示检测结果
    if show_result:
        plt.figure(figsize=(12, 8))
        plt.imshow(image_with_detections)
        plt.axis('off')
        plt.title(f"检测时间: {end_time - start_time:.3f}秒")
        plt.show()
    
    # 保存检测结果
    if save_result and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, cv2.cvtColor(image_with_detections, cv2.COLOR_RGB2BGR))
        print(f"结果已保存: {output_path}")
    
    print(f"检测到 {len(valid_indices)} 个目标, 耗时: {end_time - start_time:.3f}秒")
    
    # 返回检测结果
    result = {
        'image': image_with_detections,
        'boxes': boxes[valid_indices],
        'classes': classes[valid_indices],
        'scores': scores[valid_indices],
        'time': end_time - start_time
    }
    
    return result

def process_directory(detect_fn, directory, label_map, threshold=0.5, output_dir=None):
    """处理目录中的所有图像
    
    Args:
        detect_fn: 检测函数
        directory: 图像目录
        label_map: 类别标签映射
        threshold: 置信度阈值
        output_dir: 结果保存目录
    """
    # 获取目录中的所有图像文件
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp']
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(directory, f"*.{ext}")
        image_files.extend(glob.glob(pattern))
        # 检查大写扩展名
        pattern = os.path.join(directory, f"*.{ext.upper()}")
        image_files.extend(glob.glob(pattern))
    
    if not image_files:
        print(f"目录中没有发现图像文件: {directory}")
        return
    
    print(f"发现 {len(image_files)} 个图像文件")
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个图像文件
    for image_path in image_files:
        detect_image(
            detect_fn,
            image_path,
            label_map,
            threshold=threshold,
            show_result=False,
            save_result=True,
            output_dir=output_dir
        )
    
    print(f"已处理 {len(image_files)} 个图像文件，结果保存在 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="使用EfficientDet进行钢材表面缺陷检测")
    parser.add_argument("--model_path", type=str, required=True, help="SavedModel格式的模型路径")
    parser.add_argument("--dataset", type=str, choices=["neu", "gc10"], default="neu",
                        help="数据集类型，决定使用的标签映射: neu (NEU-DET) 或 gc10 (GC10-DET)")
    parser.add_argument("--image_path", type=str, help="输入图像路径或目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="检测置信度阈值，默认0.5")
    parser.add_argument("--output_dir", type=str, help="检测结果保存目录")
    parser.add_argument("--show", action="store_true", help="显示检测结果")
    
    args = parser.parse_args()
    
    # 选择标签映射
    label_map = NEU_LABEL_MAP if args.dataset == "neu" else GC10_LABEL_MAP
    
    # 加载模型
    detect_fn = load_model(args.model_path)
    
    # 检查输入是文件还是目录
    if not args.image_path:
        parser.error("必须指定--image_path参数")
    
    if os.path.isfile(args.image_path):
        # 处理单个图像
        detect_image(
            detect_fn,
            args.image_path,
            label_map,
            threshold=args.threshold,
            show_result=args.show,
            save_result=args.output_dir is not None,
            output_dir=args.output_dir
        )
    elif os.path.isdir(args.image_path):
        # 处理目录
        process_directory(
            detect_fn,
            args.image_path,
            label_map,
            threshold=args.threshold,
            output_dir=args.output_dir
        )
    else:
        parser.error(f"找不到图像路径: {args.image_path}")

if __name__ == "__main__":
    main() 