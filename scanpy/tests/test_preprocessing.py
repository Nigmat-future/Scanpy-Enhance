"""
预处理模块单元测试
"""

import os
import sys
import unittest
import numpy as np
import scanpy as sc
from anndata import AnnData

# 添加项目根目录到路径
sys.path.append('..')

from src.preprocessing.adaptive_preprocessing import AdaptivePreprocessing

class TestAdaptivePreprocessing(unittest.TestCase):
    """
    测试自适应预处理模块
    """
    
    def setUp(self):
        """
        测试前准备，创建测试数据
        """
        # 创建一个简单的测试数据集
        np.random.seed(42)
        n_cells, n_genes = 100, 200
        X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
        
        # 添加一些噪声
        X[np.random.rand(*X.shape) > 0.8] = 0
        
        # 添加一些nan值
        mask = np.random.rand(*X.shape) < 0.05
        X = X.astype(float)
        X[mask] = np.nan
        
        # 创建AnnData对象
        self.adata = AnnData(X)
        
        # 初始化预处理器
        self.preprocessor = AdaptivePreprocessing(self.adata)
        
    def test_calculate_dynamic_thresholds(self):
        """
        测试动态阈值计算
        """
        cell_thresh, gene_thresh = self.preprocessor.calculate_dynamic_thresholds()
        
        # 检查阈值是否在合理范围内
        self.assertTrue(0 < cell_thresh < self.adata.shape[1])
        self.assertTrue(0 < gene_thresh < self.adata.shape[0])
        
    def test_filter_data(self):
        """
        测试数据过滤功能
        """
        # 记录原始形状
        original_shape = self.adata.shape
        
        # 使用动态阈值进行过滤
        self.preprocessor.filter_data(use_dynamic_thresholds=True)
        
        # 检查形状是否改变
        self.assertNotEqual(original_shape, self.adata.shape)
        
    def test_handle_missing_values(self):
        """
        测试缺失值处理
        """
        # 确保数据中有nan值
        has_nans = np.any(np.isnan(self.adata.X))
        self.assertTrue(has_nans, "测试数据中应该包含NaN值")
        
        # 测试零值替换
        self.preprocessor.handle_missing_values(method='zeros')
        
        # 确认nan值已被替换
        self.assertFalse(np.any(np.isnan(self.adata.X)))
        
    def test_run_full_pipeline(self):
        """
        测试完整预处理流程
        """
        # 运行完整流程
        self.preprocessor.run_full_pipeline(handle_missing=True)
        
        # 测试结果
        self.assertFalse(np.any(np.isnan(self.adata.X)), "数据中不应该有NaN值")
        self.assertTrue('highly_variable' in self.adata.var, "应该有高变异基因标记")
        self.assertTrue(np.sum(self.adata.var.highly_variable) > 0, "应该至少有一个高变异基因")
        
if __name__ == '__main__':
    unittest.main() 