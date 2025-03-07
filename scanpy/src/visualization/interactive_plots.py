"""
提供交互式可视化功能的模块，包括：
- 降维结果可视化
- 基因表达可视化
- 聚类结果可视化
- 轨迹分析可视化
"""

import scanpy as sc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union, Tuple
import datashader as ds
from datashader.bundling import connect_edges
import colorcet as cc

class InteractivePlots:
    """
    提供交互式可视化功能的类，实现了一系列基于Plotly的可视化方法：
    - 交互式降维结果展示
    - 基因表达热图
    - 聚类结果可视化
    - 细胞轨迹分析
    """
    
    def __init__(self, adata: sc.AnnData):
        """
        初始化可视化类
        
        @param {sc.AnnData} adata - AnnData对象，包含处理后的数据
        """
        self.adata = adata
        
    def interactive_embedding(self,
                            basis: str = 'umap',
                            color_by: Optional[str] = None,
                            category_orders: Optional[Dict[str, List[str]]] = None,
                            title: str = '') -> go.Figure:
        """
        创建交互式降维可视化
        
        @param {str} basis - 降维方法 ('umap', 'tsne', 'pca' 等)
        @param {Optional[str]} color_by - 用于着色的观测值或基因名
        @param {Optional[Dict[str, List[str]]]} category_orders - 分类变量的顺序字典
        @param {str} title - 图表标题
        @returns {go.Figure} - Plotly图形对象
        """
        # 准备数据
        embedding_key = f'X_{basis}'
        if embedding_key not in self.adata.obsm:
            raise ValueError(f"未找到嵌入 {basis}")
            
        df = pd.DataFrame(
            self.adata.obsm[embedding_key][:, :2],
            columns=[f'{basis}_1', f'{basis}_2']
        )
        
        if color_by:
            if color_by in self.adata.obs:
                df['color'] = self.adata.obs[color_by].astype(str)
            elif color_by in self.adata.var_names:
                df['color'] = self.adata[:, color_by].X.flatten()
            else:
                raise ValueError(f"未找到颜色变量 {color_by}")
                
        # 创建散点图
        if color_by in self.adata.obs and isinstance(self.adata.obs[color_by].dtype, pd.CategoricalDtype):
            # 分类数据使用离散颜色
            fig = px.scatter(
                df,
                x=f'{basis}_1',
                y=f'{basis}_2',
                color='color',
                title=title,
                category_orders={'color': category_orders.get(color_by, None)} if category_orders else None
            )
        else:
            # 连续数据使用连续颜色映射
            fig = px.scatter(
                df,
                x=f'{basis}_1',
                y=f'{basis}_2',
                color='color' if color_by else None,
                title=title,
                color_continuous_scale='Viridis'
            )
            
        # 更新布局
        fig.update_layout(
            template='plotly_white',
            width=800,
            height=600,
            showlegend=True if color_by else False
        )
        
        # 更新轴标签
        fig.update_xaxes(title=f'{basis.upper()}_1')
        fig.update_yaxes(title=f'{basis.upper()}_2')
        
        return fig
        
    def plot_gene_expression(self,
                           genes: Union[str, List[str]],
                           basis: str = 'umap',
                           ncols: int = 2,
                           title: str = '') -> go.Figure:
        """
        绘制基因表达的降维图
        
        @param {Union[str, List[str]]} genes - 要可视化的基因或基因列表
        @param {str} basis - 降维方法 ('umap', 'tsne', 'pca' 等)
        @param {int} ncols - 每行显示的图表数量
        @param {str} title - 图表标题
        @returns {go.Figure} - Plotly图形对象
        """
        if isinstance(genes, str):
            genes = [genes]
            
        # 检查基因是否存在
        valid_genes = [gene for gene in genes if gene in self.adata.var_names]
        if not valid_genes:
            raise ValueError("未找到有效的基因")
            
        # 计算行数
        nrows = (len(valid_genes) + ncols - 1) // ncols
        
        # 创建子图
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[f'{gene} 表达' for gene in valid_genes]
        )
        
        # 获取降维数据
        embedding_key = f'X_{basis}'
        if embedding_key not in self.adata.obsm:
            raise ValueError(f"未找到嵌入 {basis}")
            
        coords = self.adata.obsm[embedding_key]
        
        # 为每个基因创建散点图
        for i, gene in enumerate(valid_genes):
            row = i // ncols + 1
            col = i % ncols + 1
            
            expression = self.adata[:, gene].X.flatten()
            
            scatter = go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode='markers',
                marker=dict(
                    size=5,
                    color=expression,
                    colorscale='Viridis',
                    showscale=True if i == 0 else False
                ),
                name=gene
            )
            
            fig.add_trace(scatter, row=row, col=col)
            
            # 更新轴标签
            fig.update_xaxes(title=f'{basis.upper()}_1', row=row, col=col)
            fig.update_yaxes(title=f'{basis.upper()}_2', row=row, col=col)
            
        # 更新布局
        fig.update_layout(
            template='plotly_white',
            title=title,
            height=400 * nrows,
            width=600 * ncols,
            showlegend=False
        )
        
        return fig
        
    def plot_trajectory(self,
                       basis: str = 'umap',
                       color_by: Optional[str] = None,
                       n_neighbors: int = 10,
                       min_mass: int = 5) -> go.Figure:
        """
        创建细胞轨迹图
        
        @param {str} basis - 降维方法 ('umap', 'tsne', 'pca' 等)
        @param {Optional[str]} color_by - 用于着色的观测值
        @param {int} n_neighbors - 用于构建轨迹的邻居数
        @param {int} min_mass - 最小点密度
        @returns {go.Figure} - Plotly图形对象
        """
        # 获取降维数据
        embedding_key = f'X_{basis}'
        if embedding_key not in self.adata.obsm:
            raise ValueError(f"未找到嵌入 {basis}")
            
        coords = self.adata.obsm[embedding_key]
        
        # 创建数据框
        df = pd.DataFrame(
            coords[:, :2],
            columns=[f'{basis}_1', f'{basis}_2']
        )
        
        if color_by:
            if color_by in self.adata.obs:
                df['color'] = self.adata.obs[color_by]
            else:
                raise ValueError(f"未找到颜色变量 {color_by}")
                
        # 使用datashader创建密度图
        cvs = ds.Canvas(plot_width=400, plot_height=400)
        agg = cvs.points(df, f'{basis}_1', f'{basis}_2')
        
        # 创建轨迹
        graph = connect_edges(
            agg,
            n_neighbors=n_neighbors,
            min_mass=min_mass
        )
        
        # 创建图形
        fig = go.Figure()
        
        # 添加密度图
        fig.add_trace(
            go.Heatmap(
                z=agg.values,
                colorscale='Viridis',
                showscale=False
            )
        )
        
        # 添加轨迹
        for line in graph:
            fig.add_trace(
                go.Scatter(
                    x=line.coords[0],
                    y=line.coords[1],
                    mode='lines',
                    line=dict(color='white', width=1),
                    showlegend=False
                )
            )
            
        # 更新布局
        fig.update_layout(
            template='plotly_dark',
            title=f'{basis.upper()} 轨迹分析',
            xaxis_title=f'{basis.upper()}_1',
            yaxis_title=f'{basis.upper()}_2',
            width=800,
            height=800
        )
        
        return fig
        
    def plot_cluster_tree(self,
                         cluster_key: str = 'consensus_clusters') -> go.Figure:
        """
        创建聚类树的交互式可视化
        
        @param cluster_key: 聚类结果的键名
        @returns: Plotly图形对象
        """
        if cluster_key not in self.adata.obs:
            raise ValueError(f"Cluster key {cluster_key} not found in adata.obs")
            
        # 创建简单的树状图
        fig = go.Figure()
        
        # 使用现有的聚类作为颜色
        clusters = self.adata.obs[cluster_key].astype('category').cat.categories
        
        # 创建简化的树形结构
        x_positions = list(range(len(clusters)))
        y_positions = [0] * len(clusters)
        
        # 添加每个聚类的标记
        for i, cluster in enumerate(clusters):
            fig.add_trace(
                go.Scatter(
                    x=[x_positions[i]],
                    y=[y_positions[i]],
                    mode='markers+text',
                    marker=dict(size=15, color=f'hsl({i*360/len(clusters)}, 80%, 50%)'),
                    text=[cluster],
                    textposition="bottom center",
                    name=cluster
                )
            )
        
        # 优化布局
        fig.update_layout(
            template='plotly_white',
            width=800,
            height=400,
            title="Cluster Distribution",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        
        return fig 