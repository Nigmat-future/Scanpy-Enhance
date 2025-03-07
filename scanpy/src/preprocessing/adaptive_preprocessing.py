"""
自适应预处理类，提供动态参数选择和优化的预处理方法

这个类实现了一系列增强的单细胞RNA测序数据预处理技术，包括：
- 自适应过滤阈值计算
- 多种标准化方法支持
- 数据质量评估
- 缺失值处理
"""

import scanpy as sc
import numpy as np
from scipy import stats
from typing import Optional, Union, Tuple, List, Dict
import warnings

class AdaptivePreprocessing:
    """
    提供自适应预处理功能的类，实现了一系列增强的预处理方法：
    - 自动优化过滤参数
    - 多种标准化方法
    - 缺失值处理
    """
    
    def __init__(self, adata: sc.AnnData):
        """
        初始化预处理类
        
        @param {sc.AnnData} adata - AnnData对象，包含单细胞数据
        """
        self.adata = adata
        
    def calculate_dynamic_thresholds(self, 
                                   percentile: float = 5.0,
                                   use_mad: bool = True) -> Tuple[float, float]:
        """
        基于数据分布计算动态阈值
        
        @param {float} percentile - 用于计算下限的百分位数
        @param {bool} use_mad - 是否使用中位数绝对偏差(MAD)而不是标准差
        @returns {Tuple[float, float]} - 基因数和细胞数的阈值元组
        """
        # 计算每个细胞的基因数
        n_genes_per_cell = np.sum(self.adata.X > 0, axis=1)
        
        # 计算每个基因在多少个细胞中表达
        n_cells_per_gene = np.sum(self.adata.X > 0, axis=0)
        
        # 计算基因过滤阈值
        gene_median = np.median(n_cells_per_gene)
        if use_mad:
            # 使用MAD更稳健，对异常值不敏感
            gene_mad = stats.median_abs_deviation(n_cells_per_gene)
            gene_thresh = max(3, gene_median - 2 * gene_mad)
        else:
            gene_thresh = max(3, np.percentile(n_cells_per_gene, percentile))
            
        # 计算细胞过滤阈值
        cell_median = np.median(n_genes_per_cell)
        if use_mad:
            cell_mad = stats.median_abs_deviation(n_genes_per_cell)
            cell_thresh = max(200, cell_median - 2 * cell_mad)
        else:
            cell_thresh = max(200, np.percentile(n_genes_per_cell, percentile))
            
        return cell_thresh, gene_thresh
        
    def filter_data(self,
                   min_genes: Optional[int] = None,
                   min_cells: Optional[int] = None,
                   max_genes: Optional[int] = None,
                   max_counts: Optional[int] = None,
                   use_dynamic_thresholds: bool = True) -> None:
        """
        过滤低质量细胞和低表达基因
        
        @param {Optional[int]} min_genes - 每个细胞至少应表达的最小基因数
        @param {Optional[int]} min_cells - 每个基因至少在多少个细胞中表达
        @param {Optional[int]} max_genes - 每个细胞最多表达的基因数 (用于过滤双细胞)
        @param {Optional[int]} max_counts - 每个细胞最大计数 (用于过滤异常细胞)
        @param {bool} use_dynamic_thresholds - 是否使用自适应阈值
        @returns {None}
        """
        if use_dynamic_thresholds:
            cell_thresh, gene_thresh = self.calculate_dynamic_thresholds()
            
            min_genes = min_genes or cell_thresh
            min_cells = min_cells or gene_thresh
            
            print(f"使用动态阈值: 最小基因数={min_genes:.1f}, 最小细胞数={min_cells:.1f}")
            
        # 过滤细胞
        sc.pp.filter_cells(self.adata, min_genes=min_genes)
        
        # 过滤基因
        sc.pp.filter_genes(self.adata, min_cells=min_cells)
        
        # 过滤可能的双细胞
        if max_genes is not None:
            self.adata = self.adata[self.adata.obs.n_genes < max_genes]
            
        # 过滤异常细胞
        if max_counts is not None:
            self.adata = self.adata[self.adata.obs.n_counts < max_counts]
            
        print(f"过滤后数据形状: {self.adata.shape}")
        
    def normalize_data(self,
                      method: str = 'standard',
                      target_sum: Optional[float] = None,
                      exclude_highly_expressed: bool = False) -> None:
        """
        标准化数据
        
        @param {str} method - 标准化方法 ('standard', 'cell_size', 'pearson')
        @param {Optional[float]} target_sum - 目标总和缩放因子
        @param {bool} exclude_highly_expressed - 是否排除高表达基因
        @returns {None}
        """
        if method == 'standard':
            # 标准的scanpy标准化
            sc.pp.normalize_total(
                self.adata,
                target_sum=target_sum or 1e4,
                exclude_highly_expressed=exclude_highly_expressed
            )
            print("使用标准总和标准化")
            
        elif method == 'cell_size':
            # 基于细胞大小的标准化
            sc.pp.normalize_per_cell(
                self.adata,
                counts_per_cell_after=target_sum or 1e4
            )
            print("使用细胞大小标准化")
                
        elif method == 'pearson':
            # Pearson残差
            from scipy.sparse import issparse
            
            # 如果是稀疏矩阵，需要先转换为密集矩阵
            if issparse(self.adata.X):
                X = self.adata.X.toarray()
            else:
                X = self.adata.X
                
            # 计算总和和均值
            cell_sums = X.sum(axis=1, keepdims=True)
            gene_means = X.mean(axis=0, keepdims=True)
            total_mean = X.mean()
            
            # 计算期望值
            expected = np.outer(cell_sums, gene_means) / total_mean
            
            # 计算Pearson残差
            variance = expected * (1 - gene_means / total_mean)
            pearson_residuals = (X - expected) / np.sqrt(variance)
            
            # 替换数据
            self.adata.X = pearson_residuals
            print("使用Pearson残差标准化")
        else:
            raise ValueError(f"未知的标准化方法: {method}")
            
    def handle_missing_values(self, method: str = 'zeros') -> None:
        """
        处理缺失值
        
        @param {str} method - 处理方法 ('zeros', 'mean', 'median', 'knn')
        @returns {None}
        """
        from scipy.sparse import issparse
        
        # 首先转换为密集矩阵以便处理
        if issparse(self.adata.X):
            X = self.adata.X.toarray()
        else:
            X = self.adata.X
            
        # 检查是否有NaN值
        if not np.any(np.isnan(X)):
            print("数据中没有缺失值")
            return
            
        # 应用不同的缺失值处理方法
        if method == 'zeros':
            print("将缺失值替换为零")
            X = np.nan_to_num(X, nan=0.0)
        elif method == 'mean':
            print("将缺失值替换为列平均值")
            col_means = np.nanmean(X, axis=0)
            mask = np.isnan(X)
            for i in range(X.shape[1]):
                X[mask[:, i], i] = col_means[i]
        elif method == 'median':
            print("将缺失值替换为列中位数")
            col_medians = np.nanmedian(X, axis=0)
            mask = np.isnan(X)
            for i in range(X.shape[1]):
                X[mask[:, i], i] = col_medians[i]
        elif method == 'knn':
            print("使用KNN插值替换缺失值")
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5)
            X = imputer.fit_transform(X)
        else:
            raise ValueError(f"未知的缺失值处理方法: {method}")
            
        # 更新数据
        self.adata.X = X
        
    def run_full_pipeline(self,
                         min_genes: Optional[int] = None,
                         min_cells: Optional[int] = None,
                         normalization_method: str = 'standard',
                         log_transform: bool = True,
                         handle_missing: bool = True) -> None:
        """
        运行完整的预处理流程
        
        @param {Optional[int]} min_genes - 每个细胞至少应表达的最小基因数
        @param {Optional[int]} min_cells - 每个基因至少在多少个细胞中表达
        @param {str} normalization_method - 标准化方法
        @param {bool} log_transform - 是否进行对数转换
        @param {bool} handle_missing - 是否处理缺失值
        @returns {None}
        """
        # 1. 过滤数据
        print("1. 过滤数据...")
        self.filter_data(min_genes=min_genes, min_cells=min_cells)
        
        # 2. 计算QC指标
        print("2. 计算质量控制指标...")
        sc.pp.calculate_qc_metrics(self.adata, inplace=True)
        
        # 3. 处理缺失值
        if handle_missing:
            print("3. 处理缺失值...")
            self.handle_missing_values()
        
        # 4. 标准化
        print("4. 数据标准化...")
        self.normalize_data(method=normalization_method)
        
        # 5. 对数转换
        if log_transform:
            print("5. 对数转换...")
            sc.pp.log1p(self.adata)
            
        # 6. 识别高变异基因
        print("6. 识别高变异基因...")
        sc.pp.highly_variable_genes(self.adata, flavor='seurat', n_top_genes=2000)
        
        print("预处理完成。")
        print(f"最终数据形状: {self.adata.shape}")
        print(f"高变异基因数量: {sum(self.adata.var.highly_variable)}")