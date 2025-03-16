# 基于EfficientDet算法的钢材表面缺陷检测

<div align="center">
  <img src="assets/doohuan_logo.png" alt="多焕智能Logo" width="300"/>
  <h1>多焕智能（DooHuan AI）</h1>
  <p>
    <b>工业智能视觉检测系统</b>
  </p>
  <p>
    <a href="https://www.doohuan.com">
      <img alt="官网" src="https://img.shields.io/badge/官网-doohuan.com-blue?style=flat-square" />
    </a>
    <a href="https://github.com/doohuan-ai">
      <img alt="GitHub" src="https://img.shields.io/badge/GitHub-doohuan--ai-lightgrey?style=flat-square&logo=github" />
    </a>
    <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
  </p>
  <hr>
</div>

基于EfficientDet的钢材表面缺陷检测系统，使用NEU-DET和GC10-DET两个公开数据集，能够检测16种不同类型的钢材表面缺陷。

## 项目特点

- 使用Google官方的EfficientDet目标检测模型
  - 精度高，可根据资源选择不同规模（D0-D7）
  - 资源利用率高，适合不同计算资源限制
- 联合两个专业钢材表面缺陷数据集
- 高精度检测16种常见钢材表面缺陷类型
- 支持实时检测和批量分析
- 提供数据集转换工具，支持COCO格式

## 数据集介绍

本项目使用了两个公开的钢材表面缺陷数据集：

1. **NEU-DET**：东北大学钢材表面缺陷数据集
   - 6种常见缺陷类型：轧制鳞片、斑块、裂纹、点蚀表面、夹杂物和划痕
   - 每种缺陷300张图像，共1800张图像
   - 图像尺寸：200×200像素

2. **GC10-DET**：钢板表面缺陷数据集
   - 10种缺陷类型：冲孔、焊接线、新月形缝隙、水斑、油斑、丝状斑点、夹杂、轧制坑、折痕和腰褶
   - 共2300张高质量标注图像

## 文件夹结构

```
├── datasets/                   # 数据集目录
│   ├── NEU-DET/               # NEU-DET数据集
│   │   ├── images/            # 图像文件
│   │   │   ├── train/         # 训练集图像
│   │   │   └── val/           # 验证集图像
│   │   ├── annotations/       # COCO格式标注
│   │   └── tfrecords/         # TFRecord格式数据
│   └── GC10-DET/              # GC10-DET数据集
│       ├── images/            # 图像文件
│       │   ├── train/         # 训练集图像
│       │   └── val/           # 验证集图像
│       ├── annotations/       # COCO格式标注
│       └── tfrecords/         # TFRecord格式数据
├── efficientdet/             # EfficientDet实现相关文件
│   ├── convert_to_coco.py    # 数据集转COCO格式脚本
│   ├── create_tfrecords.py   # 创建TFRecord文件脚本
│   ├── configs/              # 配置文件目录
│   │   ├── neu_det_config.yaml # NEU-DET配置
│   │   └── gc10_det_config.yaml # GC10-DET配置
│   ├── train.py              # EfficientDet训练脚本
│   └── detect.py             # EfficientDet目标检测推理脚本
├── requirements.txt          # 项目依赖
└── README.md                 # 项目说明文档
```

## 安装与使用

### 环境配置

1. 克隆仓库：
```bash
git clone https://github.com/doohuan-ai/SteelDefect-EfficientDet.git
cd SteelDefect-EfficientDet
```

2. 创建并激活环境：
```bash
# 使用conda
conda create -n steeldefect python=3.9
conda activate steeldefect

# 或使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

### 数据集准备

如果您已有NEU-DET和GC10-DET数据集：

#### COCO格式转换

```bash
# 转换NEU-DET为COCO格式
python efficientdet/convert_to_coco.py --dataset neu --input_path /mnt/hdd/datasets/steel-surface-defect/NEU-DET --output_path datasets/NEU-DET/annotations/neu_det.json

# 转换GC10-DET为COCO格式
python efficientdet/convert_to_coco.py --dataset gc10 --input_path /mnt/hdd/datasets/steel-surface-defect/GC10-DET --output_path datasets/GC10-DET/annotations/gc10_det.json

# 将COCO格式数据转换为TFRecord格式
python efficientdet/create_tfrecords.py --image_dir=/mnt/hdd/datasets/steel-surface-defect/NEU-DET/images --annotations_file=datasets/NEU-DET/annotations/neu_det_train.json --output_path=datasets/NEU-DET/tfrecords/neu_train.tfrecord
```

```bash
python efficientdet/create_tfrecords.py --image_dir=/mnt/hdd/datasets/steel-surface-defect/NEU-DET/images --annotations_file=datasets/NEU-DET/annotations/neu_det_val.json --output_path=datasets/NEU-DET/tfrecords/neu_val.tfrecord

python efficientdet/create_tfrecords.py --image_dir=/mnt/hdd/datasets/steel-surface-defect/GC10-DET/img --annotations_file=datasets/GC10-DET/annotations/gc10_det_train.json --output_path=datasets/GC10-DET/tfrecords/gc10_train.tfrecord
python efficientdet/create_tfrecords.py --image_dir=/mnt/hdd/datasets/steel-surface-defect/GC10-DET/img --annotations_file=datasets/GC10-DET/annotations/gc10_det_val.json --output_path=datasets/GC10-DET/tfrecords/gc10_val.tfrecord
```

如果您没有数据集，可以从以下链接下载：
- NEU-DET: http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/
- GC10-DET: https://www.kaggle.com/c/severstal-steel-defect-detection

### 模型训练

```bash
# 单独训练每个数据集
# 训练NEU-DET数据集模型
python efficientdet/train.py --dataset neu --mode train

# 训练GC10-DET数据集模型（从头开始）
python efficientdet/train.py --dataset gc10 --mode train

# 或者使用迁移学习方式训练合并模型（推荐）
# 先训练NEU-DET模型
python efficientdet/train.py --dataset neu --mode train
# 再使用NEU-DET模型作为预训练权重训练GC10-DET模型
python efficientdet/train.py --dataset gc10 --mode train --ckpt trained_models/efficientdet-neu/model.ckpt-XXXXX
```

EfficientDet训练参数（可在配置文件中修改）：
- `model_name`：选择模型规模，从efficientdet-d0到efficientdet-d7
- `batch_size`：训练批次大小
- `num_epochs`：训练轮数
- `learning_rate`：学习率
- 其他参数详见配置文件

### 模型推理

```bash
# 单张图片推理
python efficientdet/detect.py --model_path exported_models/neu_det --image_path path/to/image.jpg --threshold 0.5
```

更多参数选项请参考推理脚本。

## EfficientDet模型优势

EfficientDet的主要优势：
- 精度高，特别是D3以上模型
- 可扩展性好，有D0-D7多种规模可选
- 资源利用效率高，适合不同计算资源限制
- 采用BiFPN特征融合，提高了特征提取能力
- 复合缩放策略，平衡模型各组件的大小

## 性能指标

在联合数据集上的性能：

### EfficientDet-D0性能
- mAP50: 待测试
- mAP50-95: 待测试

## 许可证

本项目采用MIT许可证。

## 关于我们

**多焕智能（DooHuan AI）** 是一家专注于工业视觉检测和人工智能解决方案的公司，致力于为制造业提供高精度、高效率的智能检测系统。

## 联系方式

- **公司官网**：[多焕智能官网](https://www.doohuan.com)
- **GitHub**：[doohuan-ai](https://github.com/doohuan-ai)
- **邮箱**：reef@doohuan.com

## 致谢

- 感谢东北大学提供NEU-DET数据集
- 感谢Severstal提供GC10-DET数据集
- 感谢Google提供EfficientDet开源实现 