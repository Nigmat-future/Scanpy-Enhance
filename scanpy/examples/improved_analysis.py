"""
ScanPy增强分析框架 - 完整分析示例

此脚本展示如何使用增强框架进行单细胞RNA测序数据的完整分析流程。
包括预处理、降维、聚类和可视化步骤。
"""

import os
import sys
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置为CPU模式
os.environ['SCANPY_USE_CPU'] = '1'

# 添加项目根目录到路径
sys.path.append('..')

# 导入自定义模块
from src.preprocessing.adaptive_preprocessing import AdaptivePreprocessing
from src.dimensionality_reduction.advanced_reduction import AdvancedDimensionalityReduction
from src.clustering.adaptive_clustering import AdaptiveClustering
from src.visualization.interactive_plots import InteractivePlots

def main():
    """
    展示完整的增强版scanpy分析流程
    """
    # 创建输出目录
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载示例数据
    print("加载示例数据...")
    adata = sc.datasets.pbmc68k_reduced()
    print(f"数据加载完成，形状: {adata.shape}")
    
    # 1. 自适应预处理
    print("\n执行自适应预处理...")
    preprocessor = AdaptivePreprocessing(adata)
    preprocessor.run_full_pipeline(
        normalization_method='scran'
    )
    print("预处理完成")
    
    # 2. 降维
    print("\n执行改进的降维...")
    reducer = AdvancedDimensionalityReduction(adata)
    reducer.run_multiple_reductions(
        methods=['pca'],
        n_components=50
    )
    print("降维完成")
    
    # 计算UMAP用于可视化
    print("\n计算UMAP嵌入...")
    sc.pp.neighbors(adata, use_rep='X_pca')
    sc.tl.umap(adata)
    
    # 3. 自适应聚类
    print("\n执行自适应聚类...")
    clusterer = AdaptiveClustering(adata)
    optimal_res = clusterer.optimize_resolution()
    print(f"最优分辨率: {optimal_res}")
    clusterer.run_consensus_clustering(resolution=optimal_res)
    
    # 4. 差异表达分析
    print("\n执行差异表达分析...")
    sc.tl.rank_genes_groups(
        adata,
        groupby='consensus_clusters',
        method='wilcoxon',
        key_added='rank_genes_wilcoxon'
    )
    
    # 获取差异表达基因结果并保存
    result = sc.get.rank_genes_groups_df(adata, group=None, key='rank_genes_wilcoxon')
    result.to_csv(f'{output_dir}/de_genes.csv', index=False)
    print(f"已找到 {len(result)} 个差异表达基因")
    
    # 5. 可视化 - 包括交互式和静态图表
    print("\n创建可视化...")
    
    # 5.1 交互式可视化 (Plotly)
    plotter = InteractivePlots(adata)
    
    # 创建UMAP嵌入图
    umap_fig = plotter.interactive_embedding(
        basis='umap',
        color_by='consensus_clusters',
        title='UMAP - 聚类'
    )
    umap_fig.write_html(f'{output_dir}/umap_clusters.html')
    
    # 创建细胞分布图
    dist_fig = plotter.plot_trajectory(
        basis='umap',
        color_by='consensus_clusters'
    )
    dist_fig.write_html(f'{output_dir}/cell_distribution.html')
    
    # 5.2 静态可视化 (Matplotlib)
    # UMAP聚类图
    plt.figure(figsize=(10, 8))
    sc.pl.umap(adata, color='consensus_clusters', title='UMAP - 聚类', show=False)
    plt.savefig(f'{output_dir}/umap_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 获取每个聚类的前5个标记基因
    marker_genes = {}
    for group in adata.obs['consensus_clusters'].cat.categories:
        markers = sc.get.rank_genes_groups_df(adata, group=group)['names'].tolist()[:5]
        marker_genes[group] = markers
    
    # 为前3个聚类创建小提琴图
    for i, group in enumerate(list(adata.obs['consensus_clusters'].cat.categories)[:3]):
        genes = marker_genes[group]
        plt.figure(figsize=(15, 8))
        sc.pl.violin(adata, genes, groupby='consensus_clusters', title=f'{group} 标记基因表达', show=False)
        plt.savefig(f'{output_dir}/violin_cluster_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\n分析完成！结果已保存到目录:", output_dir)
    print("生成的文件：")
    print("- umap_clusters.html：交互式UMAP降维可视化")
    print("- cell_distribution.html：交互式细胞分布图")
    print("- umap_clusters.png：静态UMAP聚类图")
    print("- violin_cluster_*.png：标记基因小提琴图")
    print("- de_genes.csv：差异表达基因数据")

if __name__ == '__main__':
    main() 