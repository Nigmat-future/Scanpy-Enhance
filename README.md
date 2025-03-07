# Scanpy-Enhance
加强scanpy，对新手更加友好
# ScanPy增强分析框架

这是一个针对单细胞RNA测序数据分析的增强框架，基于ScanPy构建，提供了更加自适应的预处理、更稳健的降维、更一致的聚类以及多种可视化方式。

## 项目简介

本项目旨在增强ScanPy在单细胞数据分析中的能力，通过一系列优化和扩展解决常见分析问题：

- **自适应预处理**：动态参数选择，适应不同类型单细胞数据集的特点
- **增强降维**：改进的PCA和UMAP实现，提高结果稳定性
- **一致性聚类**：结合多次运行结果的聚类方法，减少随机性影响
- **多模式可视化**：同时支持交互式(HTML)和静态(图片)可视化

## 项目结构

```
scanpy-enhanced/
├── examples/                  # 示例脚本
│   ├── improved_analysis.py   # 完整的增强分析流程示例
│   ├── basic_visualization.py # 基本可视化示例
│   └── simple_visualization.py # 简单可视化示例
├── src/                       # 源代码
│   ├── preprocessing/         # 预处理模块
│   ├── dimensionality_reduction/ # 降维模块
│   ├── clustering/            # 聚类模块
│   ├── visualization/         # 可视化模块
│   └── utils/                 # 工具函数
├── tests/                     # 测试
├── requirements.txt           # 依赖项
└── README.md                  # 项目文档
```

## 技术亮点

1. **自适应参数选择**：基于数据分布自动计算过滤、聚类的最优参数
2. **一致性聚类**：通过多次运行不同参数设置合并聚类结果，提高结果稳定性
3. **双模式可视化**：同时支持交互式探索和发表质量的静态图表
4. **健壮的错误处理**：详细的异常处理和回退机制，减少分析中断
5. **CPU友好处理**：针对无GPU环境优化，降低资源需求

## 与传统ScanPy的对比

| 功能 | ScanPy增强框架 | 传统ScanPy |
|-----|--------------|-----------|
| 预处理 | 自适应阈值选择，多种标准化方法 | 固定参数，基本标准化 |
| 降维 | 增强的PCA/UMAP，批次校正 | 基本PCA/UMAP |
| 聚类 | 一致性聚类，分辨率优化 | 单一运行的Leiden/Louvain |
| 可视化 | 双模式(交互+静态) | 主要是静态图表 |
| 异常处理 | 缺失值处理，异常检测 | 基本错误处理 |

## 使用示例

### 基本分析

```python
import scanpy as sc
from src.preprocessing.adaptive_preprocessing import AdaptivePreprocessing
from src.clustering.adaptive_clustering import AdaptiveClustering

# 加载数据
adata = sc.datasets.pbmc68k_reduced()

# 自适应预处理
preprocessor = AdaptivePreprocessing(adata)
preprocessor.run_full_pipeline()

# 自适应聚类
clusterer = AdaptiveClustering(adata)
clusterer.run_consensus_clustering()

# 可视化
sc.pl.umap(adata, color='consensus_clusters')
```

### 静态可视化

```python
import matplotlib.pyplot as plt
import os

# 创建保存目录
output_dir = "matplotlib_plots"
os.makedirs(output_dir, exist_ok=True)

# UMAP聚类图
plt.figure(figsize=(10, 8))
sc.pl.umap(adata, color='consensus_clusters', title='UMAP - 聚类', show=False)
plt.savefig(f"{output_dir}/umap_clusters.png", dpi=300, bbox_inches='tight')
plt.close()
```

## 安装与依赖

```bash
# 克隆仓库
git clone https://github.com/nigmat-future/scanpy-enhanced.git
cd scanpy-enhanced

# 安装依赖
pip install -r requirements.txt
```

主要依赖项：
- scanpy>=1.9.3
- anndata>=0.8.0
- numpy>=1.20.0
- pandas>=1.3.0
- matplotlib>=3.5.0
- scikit-learn>=1.0.0
- umap-learn>=0.5.3
- leidenalg>=0.9.0

## 贡献指南

欢迎通过以下方式贡献：
1. Fork仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 致谢

- 感谢ScanPy团队提供的出色基础库
- 感谢单细胞数据分析社区的宝贵反馈 
