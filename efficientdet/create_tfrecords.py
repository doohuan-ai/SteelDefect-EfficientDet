#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将COCO格式数据转换为TFRecord格式
用于EfficientDet模型训练
"""

import os
import json
import hashlib
import io
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
from tqdm import tqdm
import contextlib2

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_tf_example(image_info, annotations, image_dir, include_masks=False):
    """创建一个TF Example
    
    Args:
        image_info: 图像信息字典
        annotations: 对应图像的标注列表
        image_dir: 图像目录路径
        include_masks: 是否包含分割掩码
    
    Returns:
        TF Example
    """
    img_path = os.path.join(image_dir, image_info['file_name'])
    
    # 打开图像并读取内容
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    
    # 计算图像哈希值
    key = hashlib.sha256(encoded_jpg).hexdigest()
    
    # 读取图像尺寸
    width = image_info['width']
    height = image_info['height']
    
    # 提取边界框信息
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    areas = []
    is_crowds = []
    
    for ann in annotations:
        category_id = ann['category_id']
        # COCO格式下bbox为[x,y,width,height]，转换为[xmin,ymin,xmax,ymax]
        bbox = ann['bbox']
        xmin = bbox[0] / width
        ymin = bbox[1] / height
        xmax = (bbox[0] + bbox[2]) / width
        ymax = (bbox[1] + bbox[3]) / height
        
        # 确保坐标在[0,1]范围内
        xmin = max(0.0, min(xmin, 1.0))
        ymin = max(0.0, min(ymin, 1.0))
        xmax = max(0.0, min(xmax, 1.0))
        ymax = max(0.0, min(ymax, 1.0))
        
        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)
        
        classes.append(category_id)
        classes_text.append(str(category_id).encode('utf8'))
        
        areas.append(ann['area'])
        is_crowds.append(ann['iscrowd'])
    
    # 创建TF Example
    feature_dict = {
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(image_info['file_name'].encode('utf8')),
        'image/source_id': bytes_feature(str(image_info['id']).encode('utf8')),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature('jpg'.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
        'image/object/area': float_list_feature(areas),
        'image/object/is_crowd': int64_list_feature(is_crowds),
    }
    
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example

def create_tf_record(output_path, num_shards, image_dir, annotations_file, include_masks=False):
    """创建TFRecord文件
    
    Args:
        output_path: 输出TFRecord文件路径前缀
        num_shards: 分片数量
        image_dir: 图像目录路径
        annotations_file: COCO格式标注文件路径
        include_masks: 是否包含分割掩码
    """
    # 加载COCO格式标注
    with tf.io.gfile.GFile(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # 创建图像ID到图像信息的映射
    images = {img['id']: img for img in coco_data['images']}
    
    # 为每个图像收集对应的标注
    annotations_index = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_index:
            annotations_index[image_id] = []
        annotations_index[image_id].append(ann)
    
    # 获取所有图像ID并排序
    image_ids = sorted(list(images.keys()))
    
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 为每个分片创建TFRecord写入器
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = []
        for i in range(num_shards):
            output_file = f"{output_path}-{i:05d}-of-{num_shards:05d}.tfrecord"
            output_tfrecords.append(
                tf_record_close_stack.enter_context(tf.io.TFRecordWriter(output_file))
            )
        
        # 遍历所有图像，创建TF Example并写入TFRecord
        for i, image_id in enumerate(tqdm(image_ids, desc="创建TFRecord")):
            # 确定应该写入哪个分片
            shard_idx = i % num_shards
            
            # 获取图像信息
            image_info = images[image_id]
            
            # 获取对应的标注
            image_annotations = annotations_index.get(image_id, [])
            
            # 创建TF Example
            tf_example = create_tf_example(image_info, image_annotations, image_dir, include_masks)
            
            # 写入TFRecord
            output_tfrecords[shard_idx].write(tf_example.SerializeToString())
    
    print(f"成功创建TFRecord: {output_path}*")

def main():
    parser = argparse.ArgumentParser(description="将COCO格式数据转换为TFRecord格式")
    parser.add_argument("--image_dir", type=str, required=True, help="图像目录路径")
    parser.add_argument("--annotations_file", type=str, required=True, help="COCO格式标注文件路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出TFRecord文件路径前缀")
    parser.add_argument("--num_shards", type=int, default=10, help="分片数量，默认10")
    parser.add_argument("--include_masks", action="store_true", help="是否包含分割掩码")
    
    args = parser.parse_args()
    
    create_tf_record(
        args.output_path,
        args.num_shards,
        args.image_dir,
        args.annotations_file,
        args.include_masks
    )

if __name__ == "__main__":
    main() 