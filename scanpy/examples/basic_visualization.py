import os
os.environ['SCANPY_USE_CPU'] = '1'  # 强制使用CPU版本

import scanpy as sc
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shutil
sys.path.append('..')

def main():
    """
    使用matplotlib直接可视化scanpy分析结果的简单示例
    """
    # 创建输出目录
    output_dir = "matplotlib_plots"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # 移除旧目录
    os.makedirs(output_dir)
    
    # 设置图表风格和输出图像参数
    sc.settings.set_figure_params(dpi=100, facecolor='white')
    sc.settings.figdir = output_dir  # 设置scanpy图像保存目录
    
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 加载示例数据，使用8k人类PBMC数据
    print("加载示例数据...")
    adata = sc.datasets.pbmc3k()  # 使用更稳定的3k PBMC数据集
    print(f"数据加载完成，形状: {adata.shape}")
    
    # 预处理
    print("\n执行基本预处理...")
    # 过滤细胞和基因
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # 计算基本的QC指标
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # 线粒体基因
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # 过滤线粒体和不良细胞
    adata = adata[adata.obs.n_genes_by_counts < 2500, :].copy()
    adata = adata[adata.obs.pct_counts_mt < 5, :].copy()
    
    # 标准预处理流程
    print("\n执行标准预处理...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    print("\n计算高变异基因...")
    # 使用正确的参数计算高变异基因
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    print(f"找到 {sum(adata.var.highly_variable)} 个高变异基因")
    
    # 只保留高变异基因
    adata = adata[:, adata.var.highly_variable].copy()
    
    # 缩放数据
    sc.pp.scale(adata, max_value=10)
    
    # 处理NaN值
    print("\n处理NaN值...")
    # 将NaN值替换为0
    adata.X = np.nan_to_num(adata.X, nan=0.0)
    
    # 降维
    print("\n执行降维...")
    sc.tl.pca(adata, svd_solver='arpack', n_comps=30)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
    sc.tl.umap(adata)
    
    # 聚类
    print("\n执行聚类...")
    sc.tl.leiden(adata, resolution=0.5)
    print(f"识别出的聚类数量: {len(adata.obs['leiden'].cat.categories)}")
    
    # 找到差异表达基因
    print("\n计算差异表达基因...")
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    
    # 可视化 - 保存为图片
    print("\n生成并保存可视化图像...")
    
    # 1. UMAP展示聚类
    plt.figure(figsize=(10, 10))
    sc.pl.umap(adata, color='leiden', title='UMAP - 聚类', show=False)
    plt.savefig(os.path.join(output_dir, 'umap_clusters.png'), dpi=300)
    plt.close()
    print(f"- UMAP聚类图已保存: {os.path.join(output_dir, 'umap_clusters.png')}")
    
    # 2. 提取每个聚类的前10个差异表达基因
    marker_genes = {}
    try:
        for group in adata.obs['leiden'].cat.categories:
            markers = sc.get.rank_genes_groups_df(adata, group=group)['names'].tolist()[:10]
            marker_genes[group] = markers
        print(f"提取了 {len(marker_genes)} 个聚类的标记基因")
    except Exception as e:
        print(f"提取标记基因时出错: {e}")
        # 如果出错，使用一些常见的标记基因
        marker_genes = {
            '0': ['CD3D', 'CD3E', 'IL32', 'IL7R', 'LTB'],
            '1': ['CD79A', 'CD79B', 'MS4A1', 'CD19', 'CD79A'],
            '2': ['CST3', 'LYZ', 'TYROBP', 'FCER1G', 'LST1']
        }
    
    # 单独展示部分聚类的top5基因表达
    for group in list(adata.obs['leiden'].cat.categories)[:min(3, len(adata.obs['leiden'].cat.categories))]:
        try:
            top_genes = marker_genes[group][:5]  # 取前5个基因
            
            plt.figure(figsize=(12, 10))
            sc.pl.umap(adata, color=top_genes, ncols=3, title=f'聚类 {group} 标记基因表达', show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'markers_cluster_{group}.png'), dpi=300)
            plt.close()
            print(f"- 聚类 {group} 的标记基因表达图已保存")
            
            # 为该聚类的前5个基因创建小提琴图
            plt.figure(figsize=(15, 8))
            sc.pl.violin(adata, top_genes, groupby='leiden', title=f'聚类 {group} Top5基因表达', show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'violin_cluster_{group}.png'), dpi=300)
            plt.close()
            print(f"- 聚类 {group} 的小提琴图已保存")
        except Exception as e:
            print(f"生成聚类 {group} 的可视化时出错: {e}")
    
    # 3. PCA的降维可视化
    try:
        plt.figure(figsize=(10, 8))
        sc.pl.pca(adata, color='leiden', title='PCA - 按聚类着色', show=False)
        plt.savefig(os.path.join(output_dir, 'pca_clusters.png'), dpi=300)
        plt.close()
        print(f"- PCA可视化已保存")
    except Exception as e:
        print(f"生成PCA可视化时出错: {e}")
    
    # 4. PCA方差解释比例
    try:
        plt.figure(figsize=(8, 6))
        sc.pl.pca_variance_ratio(adata, n_pcs=20, title='PCA方差解释比例', show=False)
        plt.savefig(os.path.join(output_dir, 'pca_variance.png'), dpi=300)
        plt.close()
        print(f"- PCA方差解释比例图已保存")
    except Exception as e:
        print(f"生成PCA方差解释比例图时出错: {e}")
    
    # 5. 展示部分标记基因
    try:
        # 选择每个聚类的top3标记基因
        all_markers = []
        for g in marker_genes:
            all_markers.extend(marker_genes[g][:3])
        
        # 移除重复基因
        all_markers = list(dict.fromkeys(all_markers))[:min(10, len(all_markers))]
        
        # 创建DotPlot展示基因表达模式
        plt.figure(figsize=(12, 8))
        sc.pl.dotplot(adata, all_markers, groupby='leiden', title='主要标记基因表达', show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'marker_dotplot.png'), dpi=300)
        plt.close()
        print(f"- 标记基因点图已保存")
    except Exception as e:
        print(f"生成标记基因点图时出错: {e}")
    
    print(f"\n分析完成！所有图表已保存到目录: {output_dir}")
    print(f"检查目录下的图像文件: {os.listdir(output_dir)}")

if __name__ == '__main__':
    main() 