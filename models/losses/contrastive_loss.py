"""
对比学习损失函数 (Contrastive Learning Losses)
用于分子毒性预测的表示学习，基于毒性相似性构造正负样本对

作者: AI助手
日期: 2024-06-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class InfoNCELoss(nn.Module):
    """InfoNCE对比学习损失函数"""
    
    def __init__(self, temperature: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(self, 
                features: torch.Tensor, 
                labels: torch.Tensor,
                positive_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算InfoNCE损失
        
        Args:
            features: 分子特征表示 [batch_size, feature_dim]
            labels: 毒性标签 [batch_size, num_tasks]
            positive_mask: 正样本对掩码 [batch_size, batch_size]
            
        Returns:
            InfoNCE损失
        """
        batch_size = features.shape[0]
        device = features.device
        
        # 标准化特征
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 如果没有提供正样本掩码，则基于标签相似性构造
        if positive_mask is None:
            positive_mask = self._create_positive_mask(labels)
        
        # 移除对角线元素 (自身相似性)
        mask = torch.eye(batch_size, device=device).bool()
        similarity_matrix.masked_fill_(mask, float('-inf'))
        positive_mask.masked_fill_(mask, False)
        
        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)
        
        # 正样本的log概率
        pos_sim = similarity_matrix * positive_mask.float()
        pos_exp_sim = torch.exp(pos_sim) * positive_mask.float()
        
        # 每个样本的正样本数量
        num_positives = positive_mask.sum(dim=1)
        
        # 计算损失
        log_prob = pos_sim - torch.log(sum_exp_sim + 1e-8)
        loss = -(log_prob * positive_mask.float()).sum(dim=1) / (num_positives + 1e-8)
        
        # 过滤掉没有正样本的样本
        valid_samples = num_positives > 0
        if valid_samples.sum() > 0:
            loss = loss[valid_samples]
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def _create_positive_mask(self, labels: torch.Tensor, threshold: float = 0.8) -> torch.Tensor:
        """基于毒性标签相似性创建正样本掩码"""
        batch_size = labels.shape[0]
        device = labels.device
        
        # 计算标签相似性 (余弦相似度)
        labels_norm = F.normalize(labels.float(), dim=1)
        label_similarity = torch.matmul(labels_norm, labels_norm.T)
        
        # 基于阈值创建正样本掩码
        positive_mask = label_similarity > threshold
        
        return positive_mask


class TripletLoss(nn.Module):
    """三元组损失 - 用于分子表示学习"""
    
    def __init__(self, margin: float = 1.0, p: int = 2, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.p = p
        self.reduction = reduction
        
    def forward(self,
                anchor: torch.Tensor,
                positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        计算三元组损失
        
        Args:
            anchor: 锚点样本特征 [batch_size, feature_dim]
            positive: 正样本特征 [batch_size, feature_dim]
            negative: 负样本特征 [batch_size, feature_dim]
            
        Returns:
            三元组损失
        """
        # 计算距离
        pos_dist = F.pairwise_distance(anchor, positive, p=self.p)
        neg_dist = F.pairwise_distance(anchor, negative, p=self.p)
        
        # 三元组损失
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SupConLoss(nn.Module):
    """监督对比学习损失 (Supervised Contrastive Learning)"""
    
    def __init__(self, temperature: float = 0.1, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算监督对比学习损失
        
        Args:
            features: 特征表示 [batch_size, feature_dim]
            labels: 标签 [batch_size, num_tasks]
            
        Returns:
            监督对比学习损失
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 标准化特征
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature
        )
        
        # 数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # 创建标签掩码
        labels_normalized = F.normalize(labels.float(), dim=1)
        mask = torch.matmul(labels_normalized, labels_normalized.T) > 0.8
        
        # 移除对角线
        mask = mask.fill_diagonal_(False)
        
        # 计算log概率
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # 计算均值 log-likelihood
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        # 损失
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        
        # 过滤掉没有正样本的样本
        valid_samples = mask.sum(1) > 0
        if valid_samples.sum() > 0:
            loss = loss[valid_samples].mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            
        return loss


class HardNegativeMiner(nn.Module):
    """硬负样本挖掘器"""
    
    def __init__(self, negative_ratio: float = 0.5, distance_threshold: float = 0.1):
        super().__init__()
        self.negative_ratio = negative_ratio
        self.distance_threshold = distance_threshold
        
    def mine_hard_negatives(self,
                           features: torch.Tensor,
                           labels: torch.Tensor,
                           num_negatives: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        挖掘硬负样本
        
        Args:
            features: 特征表示 [batch_size, feature_dim]
            labels: 标签 [batch_size, num_tasks]
            num_negatives: 需要的负样本数量
            
        Returns:
            hard_negative_indices: 硬负样本索引
            hard_negative_scores: 硬负样本难度分数
        """
        batch_size = features.shape[0]
        device = features.device
        
        # 计算特征距离矩阵
        features_norm = F.normalize(features, dim=1)
        distance_matrix = torch.cdist(features_norm, features_norm, p=2)
        
        # 计算标签相似性
        labels_norm = F.normalize(labels.float(), dim=1)
        label_similarity = torch.matmul(labels_norm, labels_norm.T)
        
        # 定义负样本：标签不相似但特征相近的样本
        negative_mask = label_similarity < 0.3  # 标签不相似
        hard_mask = distance_matrix < self.distance_threshold  # 特征相近
        
        # 硬负样本掩码
        hard_negative_mask = negative_mask & hard_mask
        
        # 移除对角线
        hard_negative_mask.fill_diagonal_(False)
        
        # 计算硬负样本分数 (距离越近，标签越不相似，分数越高)
        hard_scores = (1.0 / (distance_matrix + 1e-8)) * (1.0 - label_similarity) * hard_negative_mask.float()
        
        # 为每个样本选择top-k硬负样本
        hard_negative_indices = []
        hard_negative_scores = []
        
        for i in range(batch_size):
            scores_i = hard_scores[i]
            if scores_i.sum() > 0:
                _, top_indices = torch.topk(scores_i, min(num_negatives, (scores_i > 0).sum().item()))
                hard_negative_indices.append(top_indices)
                hard_negative_scores.append(scores_i[top_indices])
            else:
                # 如果没有硬负样本，随机选择
                random_indices = torch.randperm(batch_size, device=device)[:num_negatives]
                random_indices = random_indices[random_indices != i]
                hard_negative_indices.append(random_indices)
                hard_negative_scores.append(torch.zeros(len(random_indices), device=device))
        
        return hard_negative_indices, hard_negative_scores


class ContrastiveToxicityLoss(nn.Module):
    """
    基于毒性的对比学习损失
    结合InfoNCE、三元组损失和硬负样本挖掘
    """
    
    def __init__(self,
                 temperature: float = 0.1,
                 margin: float = 1.0,
                 alpha: float = 0.5,  # InfoNCE权重
                 beta: float = 0.3,   # 三元组损失权重
                 gamma: float = 0.2): # 监督对比学习权重
        super().__init__()
        self.alpha = alpha
        self.beta = beta  
        self.gamma = gamma
        
        # 各种损失函数
        self.infonce_loss = InfoNCELoss(temperature=temperature)
        self.triplet_loss = TripletLoss(margin=margin)
        self.supcon_loss = SupConLoss(temperature=temperature)
        
        # 硬负样本挖掘器
        self.hard_negative_miner = HardNegativeMiner()
        
    def forward(self,
                features: torch.Tensor,
                labels: torch.Tensor,
                return_components: bool = False) -> torch.Tensor:
        """
        计算综合对比学习损失
        
        Args:
            features: 分子特征表示 [batch_size, feature_dim]
            labels: 毒性标签 [batch_size, num_tasks]
            return_components: 是否返回各组件损失
            
        Returns:
            总损失或损失组件字典
        """
        batch_size = features.shape[0]
        device = features.device
        
        # 1. InfoNCE损失
        infonce_loss = self.infonce_loss(features, labels)
        
        # 2. 监督对比学习损失
        supcon_loss = self.supcon_loss(features, labels)
        
        # 3. 三元组损失 (使用硬负样本挖掘)
        triplet_loss = torch.tensor(0.0, device=device)
        if batch_size >= 3:
            # 挖掘硬负样本
            hard_neg_indices, _ = self.hard_negative_miner.mine_hard_negatives(
                features, labels, num_negatives=1
            )
            
            # 构造三元组
            anchors = []
            positives = []
            negatives = []
            
            # 基于标签相似性构造正样本对
            labels_norm = F.normalize(labels.float(), dim=1)
            similarity_matrix = torch.matmul(labels_norm, labels_norm.T)
            
            for i in range(batch_size):
                # 找到最相似的正样本 (除了自己)
                sim_scores = similarity_matrix[i]
                sim_scores[i] = -1  # 排除自己
                pos_idx = sim_scores.argmax()
                
                if sim_scores[pos_idx] > 0.5:  # 相似度阈值
                    anchors.append(features[i])
                    positives.append(features[pos_idx])
                    
                    # 使用挖掘的硬负样本
                    if len(hard_neg_indices[i]) > 0:
                        neg_idx = hard_neg_indices[i][0]
                        negatives.append(features[neg_idx])
                    else:
                        # 随机负样本
                        neg_idx = torch.randint(0, batch_size, (1,), device=device)[0]
                        while neg_idx == i or neg_idx == pos_idx:
                            neg_idx = torch.randint(0, batch_size, (1,), device=device)[0]
                        negatives.append(features[neg_idx])
            
            if len(anchors) > 0:
                anchor_tensor = torch.stack(anchors)
                positive_tensor = torch.stack(positives)
                negative_tensor = torch.stack(negatives)
                triplet_loss = self.triplet_loss(anchor_tensor, positive_tensor, negative_tensor)
        
        # 总损失
        total_loss = (self.alpha * infonce_loss + 
                     self.beta * triplet_loss + 
                     self.gamma * supcon_loss)
        
        if return_components:
            return {
                'total_loss': total_loss,
                'infonce_loss': infonce_loss,
                'triplet_loss': triplet_loss,
                'supcon_loss': supcon_loss
            }
        
        return total_loss


# 对比学习训练工具函数
class ContrastiveDataAugmentation:
    """对比学习数据增强"""
    
    @staticmethod
    def molecular_augmentation(features: torch.Tensor, 
                             noise_level: float = 0.1) -> torch.Tensor:
        """分子特征增强"""
        # 添加高斯噪声
        noise = torch.randn_like(features) * noise_level
        augmented = features + noise
        return augmented
    
    @staticmethod
    def create_positive_pairs(batch_features: torch.Tensor,
                            batch_labels: torch.Tensor,
                            similarity_threshold: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建正样本对"""
        batch_size = batch_features.shape[0]
        
        # 计算标签相似性
        labels_norm = F.normalize(batch_labels.float(), dim=1)
        similarity_matrix = torch.matmul(labels_norm, labels_norm.T)
        
        # 找到相似的样本对
        positive_pairs = []
        pair_indices = []
        
        for i in range(batch_size):
            similar_indices = torch.where(similarity_matrix[i] > similarity_threshold)[0]
            similar_indices = similar_indices[similar_indices != i]  # 排除自身
            
            if len(similar_indices) > 0:
                for j in similar_indices:
                    positive_pairs.append((batch_features[i], batch_features[j]))
                    pair_indices.append((i, j.item()))
        
        if len(positive_pairs) > 0:
            pairs_tensor = torch.stack([torch.stack(pair) for pair in positive_pairs])
            return pairs_tensor, pair_indices
        else:
            return torch.empty(0, 2, batch_features.shape[1]), []


# 使用示例
if __name__ == "__main__":
    # 创建对比学习损失
    contrastive_loss = ContrastiveToxicityLoss(
        temperature=0.1,
        margin=1.0,
        alpha=0.5,
        beta=0.3,
        gamma=0.2
    )
    
    # 模拟数据
    batch_size = 8
    feature_dim = 512
    num_tasks = 26
    
    features = torch.randn(batch_size, feature_dim)
    labels = torch.rand(batch_size, num_tasks)  # 毒性标签
    
    # 计算损失
    try:
        loss_dict = contrastive_loss(features, labels, return_components=True)
        
        print("对比学习损失组件:")
        for key, value in loss_dict.items():
            print(f"{key}: {value.item():.4f}")
        
        print(f"\n对比学习损失函数创建成功!")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc() 