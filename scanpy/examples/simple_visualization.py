import os
os.environ['SCANPY_USE_CPU'] = '1'  # 强制使用CPU版本

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil

def main():
    """
    简单的可视化演示，使用scanpy内置数据集
    """
    # 创建输出目录
    output_dir = "simple_plots"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # 移除旧目录
    os.makedirs(output_dir)
    
    # 设置图形参数
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 加载内置数据集
    print("加载示例数据...")
    adata = sc.datasets.pbmc68k_reduced()
    print(f"数据加载完成 - 形状: {adata.shape}")
    
    # 检查数据
    print("\n数据检查:")
    print(f"细胞数量: {adata.n_obs}")
    print(f"基因数量: {adata.n_vars}")
    
    # 应用简单处理
    print("\n应用简单处理...")
    # 过滤掉计数为0的细胞和基因
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.filter_genes(adata, min_cells=1)
    
    # 归一化
    sc.pp.normalize_total(adata)
    # 对数变换
    sc.pp.log1p(adata)
    
    # 处理NaN值 (在PCA之前)
    print("处理NaN值...")
    adata.X = np.nan_to_num(adata.X, nan=0.0)
    
    # PCA
    print("执行PCA...")
    sc.tl.pca(adata, n_comps=30, svd_solver='arpack')
    
    # 邻居图
    print("构建邻居图...")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=15)
    
    # UMAP
    print("计算UMAP嵌入...")
    sc.tl.umap(adata)
    
    # 聚类
    print("执行聚类...")
    sc.tl.leiden(adata, resolution=0.4)
    print(f"识别出 {len(adata.obs['leiden'].unique())} 个聚类")
    
    # 生成并保存图表
    print("\n生成并保存图表...")
    
    # 1. UMAP图，按聚类着色
    plt.figure(figsize=(10, 8))
    sc.pl.umap(adata, color='leiden', title='UMAP - 聚类', show=False)
    plt.savefig(os.path.join(output_dir, '1_umap_clusters.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("- UMAP聚类图已保存")
    
    # 2. 找到每个聚类的标记基因
    print("计算标记基因...")
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    
    # 3. 为每个聚类绘制前2个标记基因
    print("绘制标记基因图...")
    for cluster in adata.obs['leiden'].unique():
        try:
            # 获取该聚类的排名靠前的基因
            genes_df = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
            if cluster in genes_df.columns:
                genes = genes_df.loc[0:1, cluster].tolist()
                
                # 绘制这些基因在UMAP中的表达
                plt.figure(figsize=(15, 6))
                for i, gene in enumerate(genes):
                    plt.subplot(1, 2, i+1)
                    sc.pl.umap(adata, color=gene, title=f'聚类 {cluster} - {gene}', show=False, color_map='viridis')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'2_markers_cluster_{cluster}.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  - 聚类 {cluster} 的标记基因图已保存")
        except Exception as e:
            print(f"  - 绘制聚类 {cluster} 的标记基因图时出错: {e}")
    
    # 4. 收集所有聚类的顶级标记基因
    try:
        print("生成小提琴图...")
        marker_genes = []
        for cluster in adata.obs['leiden'].unique():
            # 获取该聚类的排名靠前的基因
            if cluster in pd.DataFrame(adata.uns['rank_genes_groups']['names']).columns:
                gene = pd.DataFrame(adata.uns['rank_genes_groups']['names']).loc[0, cluster]
                if gene not in marker_genes:
                    marker_genes.append(gene)
        
        if marker_genes:
            # 小提琴图
            plt.figure(figsize=(12, 8))
            sc.pl.violin(adata, marker_genes, groupby='leiden', rotation=90, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '3_violin_markers.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("- 小提琴图已保存")
            
            # 点图
            plt.figure(figsize=(12, 8))
            sc.pl.dotplot(adata, marker_genes, groupby='leiden', title='标记基因表达', show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '5_dotplot.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("- 点图已保存")
    except Exception as e:
        print(f"- 生成小提琴图或点图时出错: {e}")
    
    # 5. PCA图
    try:
        print("生成PCA图...")
        plt.figure(figsize=(10, 8))
        sc.pl.pca(adata, color='leiden', title='PCA - 聚类', show=False)
        plt.savefig(os.path.join(output_dir, '4_pca.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("- PCA图已保存")
    except Exception as e:
        print(f"- 生成PCA图时出错: {e}")
    
    print(f"\n分析完成! 所有图表已保存到目录: {output_dir}")
    print(f"生成的文件: {', '.join(os.listdir(output_dir))}")

if __name__ == "__main__":
    main() 