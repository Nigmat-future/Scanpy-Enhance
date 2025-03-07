"""
辅助函数模块，提供各种常用的辅助功能
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Dict, Tuple

def ensure_dir(path: str) -> None:
    """
    确保目录存在，如果不存在则创建
    
    @param {str} path - 目录路径
    @returns {None}
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
def replace_nan_values(X: np.ndarray, 
                     method: str = 'zeros', 
                     replace_value: Optional[float] = None) -> np.ndarray:
    """
    替换数组中的NaN值
    
    @param {np.ndarray} X - 输入数组
    @param {str} method - 替换方法，可选值：'zeros', 'mean', 'median', 'value'
    @param {Optional[float]} replace_value - 当method='value'时使用的替换值
    @returns {np.ndarray} - 处理后的数组
    """
    if not np.any(np.isnan(X)):
        return X
        
    if method == 'zeros':
        return np.nan_to_num(X, nan=0.0)
    elif method == 'mean':
        col_means = np.nanmean(X, axis=0)
        result = X.copy()
        mask = np.isnan(result)
        for i in range(X.shape[1]):
            result[mask[:, i], i] = col_means[i]
        return result
    elif method == 'median':
        col_medians = np.nanmedian(X, axis=0)
        result = X.copy()
        mask = np.isnan(result)
        for i in range(X.shape[1]):
            result[mask[:, i], i] = col_medians[i]
        return result
    elif method == 'value' and replace_value is not None:
        return np.nan_to_num(X, nan=replace_value)
    else:
        raise ValueError(f"未知的替换方法: {method} 或未提供替换值")
        
def get_marker_genes(adata: sc.AnnData, 
                   cluster_key: str, 
                   n_genes: int = 5) -> Dict[str, List[str]]:
    """
    获取每个聚类的标记基因
    
    @param {sc.AnnData} adata - AnnData对象
    @param {str} cluster_key - 聚类结果的键名
    @param {int} n_genes - 每个聚类的标记基因数量
    @returns {Dict[str, List[str]]} - 标记基因字典，格式为 {cluster_id: [gene1, gene2, ...]}
    """
    # 检查是否已经计算差异表达
    if 'rank_genes_groups' not in adata.uns:
        # 如果没有计算，则进行计算
        sc.tl.rank_genes_groups(adata, groupby=cluster_key, method='wilcoxon')
    
    # 获取每个聚类的标记基因
    marker_genes = {}
    for cluster in adata.obs[cluster_key].cat.categories:
        genes = sc.get.rank_genes_groups_df(
            adata, 
            group=cluster
        )['names'].tolist()[:n_genes]
        marker_genes[cluster] = genes
        
    return marker_genes
    
def save_anndata(adata: sc.AnnData, 
               filepath: str, 
               compress: bool = True) -> None:
    """
    保存AnnData对象，处理可能的错误
    
    @param {sc.AnnData} adata - 要保存的AnnData对象
    @param {str} filepath - 保存路径
    @param {bool} compress - 是否压缩，默认为True
    @returns {None}
    """
    try:
        if compress:
            adata.write_h5ad(filepath, compression="gzip")
        else:
            adata.write_h5ad(filepath)
        print(f"数据已成功保存到 {filepath}")
    except Exception as e:
        print(f"保存数据时出错: {str(e)}")
        # 尝试保存为csv作为备份
        try:
            obs_df = adata.obs.copy()
            obs_df.to_csv(f"{os.path.splitext(filepath)[0]}_obs.csv")
            var_df = adata.var.copy()
            var_df.to_csv(f"{os.path.splitext(filepath)[0]}_var.csv")
            print(f"元数据已保存为CSV作为备份")
        except:
            print("保存备份也失败了，请检查文件系统权限和可用空间")
            
def summarize_adata(adata: sc.AnnData) -> Dict:
    """
    生成AnnData对象的概要统计信息
    
    @param {sc.AnnData} adata - AnnData对象
    @returns {Dict} - 包含统计信息的字典
    """
    stats = {
        "shape": adata.shape,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "sparsity": "稠密" if not hasattr(adata.X, "toarray") else f"{1.0 - (adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1])):.2%}",
        "obs_keys": list(adata.obs.keys()),
        "var_keys": list(adata.var.keys()),
        "uns_keys": list(adata.uns.keys()),
        "layers": list(adata.layers.keys()) if hasattr(adata, "layers") else [],
        "obsm_keys": list(adata.obsm.keys()) if hasattr(adata, "obsm") else [],
        "varm_keys": list(adata.varm.keys()) if hasattr(adata, "varm") else []
    }
    
    return stats 