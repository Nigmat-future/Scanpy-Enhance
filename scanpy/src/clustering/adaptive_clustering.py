# adaptive_clustering.py
"""
提供自适应聚类功能的模块，包括：
- 自动优化聚类参数
- 一致性聚类
- 聚类结果验证
"""

import scanpy as sc
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

class AdaptiveClustering:
    """
    提供自适应聚类功能的类，实现了一系列增强的聚类方法：
    - 自动优化Leiden聚类的分辨率参数
    - 通过多次运行提高聚类稳定性
    - 基于标记基因验证聚类结果
    """
    
    def __init__(self, adata: sc.AnnData):
        """
        初始化聚类类
        
        @param {sc.AnnData} adata - AnnData对象，包含预处理和降维后的数据
        """
        self.adata = adata
        
    def optimize_resolution(self,
                          resolution_range: np.ndarray = np.linspace(0.1, 1.0, 20),
                          use_rep: str = 'X_pca',
                          random_state: int = 42) -> float:
        """
        优化Leiden聚类的分辨率参数
        
        @param {np.ndarray} resolution_range - 要测试的分辨率值范围
        @param {str} use_rep - 用于聚类的表示（例如'X_pca'）
        @param {int} random_state - 随机种子
        @returns {float} - 最优分辨率值
        """
        if 'neighbors' not in self.adata.uns:
            sc.pp.neighbors(self.adata, use_rep=use_rep, random_state=random_state)
            
        silhouette_scores = []
        for res in resolution_range:
            # 运行Leiden聚类，使用未来版本推荐的参数
            sc.tl.leiden(
                self.adata, 
                resolution=res, 
                random_state=random_state, 
                key_added='tmp_clusters',
                flavor='igraph',  # 使用igraph而不是leidenalg
                n_iterations=2,   # 明确指定迭代次数
                directed=False    # 未来版本需要为False
            )
            
            # 计算轮廓系数
            if use_rep in self.adata.obsm:
                score = silhouette_score(
                    self.adata.obsm[use_rep],
                    self.adata.obs['tmp_clusters']
                )
                silhouette_scores.append(score)
            
        # 删除临时聚类结果
        del self.adata.obs['tmp_clusters']
        
        # 返回得分最高的分辨率
        best_res = resolution_range[np.argmax(silhouette_scores)]
        print(f"最优分辨率: {best_res:.3f}, 轮廓系数: {max(silhouette_scores):.3f}")
        
        return best_res
        
    def run_consensus_clustering(self,
                               resolution: float,
                               n_iterations: int = 10,
                               random_state: int = 42) -> None:
        """
        运行一致性聚类，通过多次运行取得稳定的聚类结果
        
        @param {float} resolution - Leiden聚类的分辨率参数
        @param {int} n_iterations - 运行次数
        @param {int} random_state - 随机种子
        @returns {None}
        """
        # 存储每次运行的结果
        all_results = []
        
        # 多次运行聚类
        for i in range(n_iterations):
            sc.tl.leiden(
                self.adata,
                resolution=resolution,
                random_state=random_state + i,
                key_added=f'leiden_{i}',
                flavor='igraph',
                n_iterations=2,
                directed=False
            )
            all_results.append(self.adata.obs[f'leiden_{i}'])
            
        # 将所有结果合并为一个矩阵
        cluster_matrix = pd.concat(all_results, axis=1)
        
        # 使用层次聚类找到一致的聚类
        n_cells = self.adata.n_obs
        similarity_matrix = np.zeros((n_cells, n_cells))
        
        # 计算细胞对之间的共现矩阵
        for i in range(n_iterations):
            clusters = cluster_matrix.iloc[:, i]
            for cluster in clusters.unique():
                mask = (clusters == cluster)
                similarity_matrix[np.ix_(mask, mask)] += 1
                
        # 归一化相似度矩阵
        similarity_matrix /= n_iterations
        
        # 使用层次聚类得到最终聚类
        n_clusters = len(np.unique(self.adata.obs[f'leiden_0']))
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            linkage='average'
        )
        
        consensus_clusters = clustering.fit_predict(1 - similarity_matrix)
        
        # 保存结果
        self.adata.obs['consensus_clusters'] = pd.Categorical(consensus_clusters)
        
        # 清理临时结果
        for i in range(n_iterations):
            del self.adata.obs[f'leiden_{i}']
            
        print(f"一致性聚类完成，识别出 {n_clusters} 个聚类")
        
    def validate_clusters(self,
                         marker_genes: List[str],
                         cluster_key: str = 'consensus_clusters') -> Dict:
        """
        使用已知的标记基因验证聚类结果
        
        @param {List[str]} marker_genes - 已知的标记基因列表
        @param {str} cluster_key - 聚类结果的键名
        @returns {Dict} - 验证结果，包含每个聚类的评分和统计信息
        """
        if cluster_key not in self.adata.obs:
            raise ValueError(f"聚类结果 '{cluster_key}' 不存在")
            
        # 检查标记基因是否在数据中
        valid_genes = [gene for gene in marker_genes if gene in self.adata.var_names]
        if not valid_genes:
            raise ValueError("没有找到有效的标记基因")
            
        # 计算每个聚类中标记基因的表达统计
        results = {}
        for cluster in self.adata.obs[cluster_key].cat.categories:
            cluster_mask = self.adata.obs[cluster_key] == cluster
            cluster_data = self.adata[cluster_mask, valid_genes].X
            
            # 计算统计量
            results[cluster] = {
                'mean_expression': np.mean(cluster_data, axis=0),
                'median_expression': np.median(cluster_data, axis=0),
                'percent_expressed': np.mean(cluster_data > 0, axis=0) * 100,
                'genes': valid_genes
            }
            
        return results 