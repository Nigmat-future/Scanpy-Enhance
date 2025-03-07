"""
提供多种改进的降维方法，包括：
- 增强的PCA实现
- 多种降维方法比较
- 降维效果评估
"""

import scanpy as sc
import numpy as np
from typing import Optional, Union, List, Dict
import warnings

class AdvancedDimensionalityReduction:
    """
    提供多种改进的降维方法，支持PCA、UMAP等多种降维技术，
    并提供降维效果的定量评估
    """
    
    def __init__(self, adata: sc.AnnData):
        """
        初始化降维类
        
        @param {sc.AnnData} adata - AnnData对象，包含预处理后的数据
        """
        self.adata = adata
        
    def run_pca(self, 
                n_components: int = 50,
                use_highly_variable: bool = True) -> None:
        """
        运行PCA降维
        
        @param {int} n_components - 降维后的维度
        @param {bool} use_highly_variable - 是否只使用高变异基因
        @returns {None}
        """
        if use_highly_variable and 'highly_variable' not in self.adata.var:
            sc.pp.highly_variable_genes(self.adata, n_top_genes=2000)
            
        sc.pp.pca(
            self.adata,
            n_comps=n_components,
            use_highly_variable=use_highly_variable
        )
        
    def run_multiple_reductions(self,
                              methods: List[str] = ['pca'],
                              n_components: int = 50) -> None:
        """
        运行多种降维方法并比较结果
        
        @param {List[str]} methods - 要运行的降维方法列表
        @param {int} n_components - 降维后的维度
        @returns {None}
        """
        for method in methods:
            if method == 'pca':
                self.run_pca(n_components=n_components)
            else:
                warnings.warn(f"Method {method} is not supported. Skipping.")
                
    def compare_methods(self,
                       cluster_key: str,
                       methods: Optional[List[str]] = None) -> Dict[str, float]:
        """
        比较不同降维方法的效果
        
        @param {str} cluster_key - 用于评估的聚类标签
        @param {Optional[List[str]]} methods - 要比较的降维方法列表
        @returns {Dict[str, float]} - 评估结果字典，格式为 {method: score}
        """
        if methods is None:
            methods = ['pca']
                
        results = {}
        
        for method in methods:
            embedding_key = f'X_{method}'
            if embedding_key in self.adata.obsm:
                # 使用UMAP进行可视化
                sc.pp.neighbors(self.adata, use_rep=embedding_key)
                sc.tl.umap(self.adata)
                
                # 计算聚类评估指标
                if cluster_key in self.adata.obs:
                    from sklearn.metrics import silhouette_score
                    score = silhouette_score(
                        self.adata.obsm[embedding_key],
                        self.adata.obs[cluster_key]
                    )
                    results[method] = score
                    
        return results 