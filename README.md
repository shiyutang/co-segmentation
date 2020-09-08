# 基于语义分割的超像素整形变化检测方法

## 简介： 
我们使用了预先在雄安和部分石家庄数据上训练的语义分割模型，backbone 为 resnet152的 PSPnet，对两个不同时相的数据进行预测，根据预测结果的差距生成变化检测结果；随后使用在两张原图上生成的超像素的融合超像素用于变化结果的整形。

通过确保每个超像素内部为单个实体，即超像素准确度超过95%的基础上，我们使用两项方法进行筛选来减少误差：
1. 减少配准误差：忽略单个超像素中不超过 ignore_pixels 像素的 
2. 减少分割误差：单个超像素中变化不超过 threshold 的，当作没有变化

**优势**：

- **准确，鲁棒**：语义分割网络综合初高级语义信息进行分类，并有全局和局部综合感受野，较大程度缓解季节，光照，无背景信息等造成的类别误判

- **轻量**：在分割网络后只需要增加超像素生成和整形模块，运行速度大大提升

## 使用方法：
1. 按照requirments.txt 安装好环境
2. 将数据按照格式放在不同的文件夹下：
    1. 所有图片分别放在img/和label/文件夹，利用命名保证原图和标签顺序对应，并成对的图像相互临近：这样的数据集 config["Data"]["path"] 中需要包含 chengdu
        ```bash        
        ├─label
        │ └─chengdu
        ├─img
        │ └─chengdu
        ```
    2. 一对对的存放，每一对中放有两张原图和 mask（对应彩色标签）和 grey mask（黑色标签）：这样的数据集 config["Data"]["path"] 中需要包含 xiongan
       ```bash
        └─xiongan
            ├─1
            │  ├─gray_mask
            │  └─mask
            ├─2...
       ```
            
3. 并设置好预训练模型的位置并在 config.json 中设置好其他参数（参数含义详见后面）

4. 运行 inference.py 

```bash 
  python3 inference.py -c config.json
```

## 参数含义和调整参数：
```json
{
  "Exp": {
    "name": "shijiazhuang_thresh_0.5_nseg_1500_0907", # 实验名称，用于区分输出结果
    "use_gpu": true,                                  # 是否使用 GPU
    "checkpoint": "./best_model.pth",                 # 需要加载的模型的位置
    "classes": 8,                                     # 检测几类的变化
    "palette": [0, 0, 0,                              # 每一类变化对应的颜色
                150, 250, 0,
                0, 250, 0,
                0, 100, 0,
                200, 0, 0,
                255, 255, 255,
                0, 0, 200,
                0, 150, 250]
  },
  "Data": {
    "path": "./origin_img/shijiazhuang"               # 数据的输入路径
  },
  "Output": {
    "record_SP_Acc": true,                            # 是否将超像素的精度记录在 SP_outpath 中     
    "SP_outpath": "./result.txt"                      # 超像素精度的输出位置
    "img_outpath": "./outputs/shijiazhuang_thresh_0.5_nseg_1500_0907", # 图像的输出位置
  },
  "SP_setting": {
    "if_merge": false,                                # 是否融合超像素
    "n_segments": [1500],                             # 单张图片分割多少超像素
    "compactness": 10,                                # 超像素参数
    "merge_regions": [0],                             # 合并后保留多少个区域
    "threshold": 0.5,                                 # 超过 threshold 之后才认为是变化
    "ignore_pixels": 20                               # 像素个数小于这个的超像素忽略
  },
  "Transform": {
    "scales": [0.75, 1.0, 1.25, 1.5, 1.75, 2.0], 
    "normalizeX": [0.45734706, 0.43338275, 0.40058118], # 归一化参数
    "normalizeY": [0.23965294, 0.23532275, 0.2398498]
  }
}

```
