from torch import nn
import pdb
import torch.nn as nn
import pdb
import os
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

__all__ = ['Conv4','resnet10', 'resnet18', 'classifier']


def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=4, stride=2)
    )


class Conv4(nn.Module):
    def __init__(self, num_classes, remove_linear=False):
        super(Conv4, self).__init__()
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        
        if remove_linear:
            self.logits = None
        else:
            self.logits = nn.Linear(64, num_classes)

    def forward(self, x, feature=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
#         
        if self.logits is None:
            if feature:
                return x, None
            else:
                return x
        if feature:
            x1 = self.logits(x)
            return x, x1

        return self.logits(x)


'''============================================= SimpleShot ResNet ========================================'''



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, remove_linear=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if remove_linear:
            self.fc = None
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, feature=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.fc is None:
            if feature:
                return x, None
            else:
                return x
        if feature:
            x1 = self.fc(x)
            return x, x1
        x = self.fc(x)
        return x


def resnet10(**kwargs):
    """Constructs a ResNet-10 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


class Classifier_Triu(nn.Module):
    def __init__(self, reg_param=0.3, fea_dim = 64):
        super(Classifier_Triu, self).__init__()
        self.reg_param = reg_param
        _num_images=600
        self.feature_dim = fea_dim
        gamma_m = float(1 / _num_images)
        gamma_S = 1 
        self.m = torch.nn.Parameter(torch.tensor( np.zeros((self.feature_dim,),)*gamma_m, dtype=torch.float32))
        self.S = torch.nn.Parameter(torch.tensor(np.eye(self.feature_dim)*gamma_S, dtype=torch.float32))
        self.nu = torch.nn.Parameter(torch.tensor(float(self.feature_dim)*gamma_S, dtype=torch.float32))
        self.kappa = torch.nn.Parameter(torch.tensor(1*gamma_m, dtype=torch.float32))
        
        self.triu_mask = torch.triu(torch.ones(self.feature_dim, self.feature_dim), diagonal=0).cuda().t()
        
        
    def fit(self, X, y):
        X = F.normalize(X, p=2, dim=1)
        self.classes_ = np.unique(y.cpu())
        self.mu = []
        self.sigma = []
        sigma = torch.zeros_like(self.S)
        self.support_N_j_list=[]
        ### common code
        lower_triu_matrix=self.S * self.triu_mask
        N_j = X[y==0].shape[0]
        head_mu_eq = (self.kappa / (self.kappa + N_j)) * self.m + (N_j / (self.kappa + N_j))
        left_eq = (self.kappa + N_j + 1)/((self.kappa+N_j)*(self.nu+N_j-self.feature_dim+1))
        middle_eq = torch.mm(lower_triu_matrix,(lower_triu_matrix).t())
        for j in self.classes_:
            X_j = X[y==j]
            N_j = X_j.shape[0]
            self.support_N_j_list.append(N_j)
            d = X_j.shape[1]
            mu_j = head_mu_eq * torch.mean(X_j, dim=0)
            S_j = torch.zeros_like(self.S)
        
            for i in range(X_j.shape[0]):
                S_j += torch.mm(X_j[i, :].unsqueeze(1), X_j[i, :].unsqueeze(0))         
            left_eq = 1.0 / (self.nu + N_j + d + 2)
            right_eq = middle_eq + S_j + self.kappa * torch.mm(self.m.unsqueeze(1), self.m.unsqueeze(0))-(self.kappa + N_j) * torch.mm(mu_j.unsqueeze(1), mu_j.unsqueeze(0))
            sigma_j = left_eq * right_eq

            sigma += sigma_j
            self.mu.append(mu_j)
            self.sigma.append(sigma_j)
            
        sigma_inv = self.inv_matrix(self.sigma)
        self.sigma_inv = self.reg_matrix(sigma_inv)
    def predict(self, X):
        X = F.normalize(X, p=2, dim=1)
        # common code 
        N_j = self.support_N_j_list[0]
        common_term = self.nu+N_j+1
        left_t_eq = torch.lgamma(0.5*common_term)-torch.lgamma(0.5*(common_term-self.feature_dim))-0.5*self.feature_dim*torch.log(common_term-self.feature_dim)
        predicts_matrix=[]
        
        for i in range(X.shape[0]):
            neg_distrances=[]
            for j in range(len(self.classes_)):
                diff = X[i, :] - self.mu[j]
                gaussian_dist = torch.mm(torch.mm(diff.unsqueeze(0), self.sigma_inv[j]), diff.unsqueeze(1))
                neg_distrances.append(-1*gaussian_dist)
            predicts_matrix.append(torch.cat(neg_distrances,dim=1))
        predicts_matrix = torch.cat(predicts_matrix,dim=0)
        
        return predicts_matrix
    
    def inv_matrix(self, matrix):
        sigma_inv=[]
        for sigma_j in matrix:
            sigma_inv.append(torch.inverse(sigma_j))
        return sigma_inv
    
    def reg_matrix(self, matrix):
        sigm_reg=[]
        for item in matrix:
            sigm_reg.append((1-self.reg_param) * item + self.reg_param * torch.eye(item.shape[0]).cuda())
        return sigm_reg
    
    def parameters(self):
        parameters = [self.m, self.S, self.nu, self.kappa]
        for p in parameters:
            yield p



class Classifier(nn.Module):
    def __init__(self, reg_param=0.3, fea_dim = 64):
        super(Classifier, self).__init__()
        self.reg_param = reg_param
        _num_images=600
        self.feature_dim = fea_dim
        gamma_m = float(1 / _num_images)
        gamma_S = 1 
        self.m = torch.nn.Parameter(torch.tensor( np.zeros((self.feature_dim,),)*gamma_m, dtype=torch.float32))
        self.S = torch.nn.Parameter(torch.tensor(np.eye(self.feature_dim)*gamma_S, dtype=torch.float32))
        self.nu = torch.nn.Parameter(torch.tensor(float(self.feature_dim)*gamma_S, dtype=torch.float32))
        self.kappa = torch.nn.Parameter(torch.tensor(1*gamma_m, dtype=torch.float32))
        
    def fit(self, X, y):
        X = F.normalize(X, p=2, dim=1)
        self.classes_ = np.unique(y.cpu())
        self.mu = []
        self.sigma = []
        sigma = torch.zeros_like(self.S)
        self.support_N_j_list=[]
        ### common code
        N_j = X[y==0].shape[0]
        head_mu_eq = (self.kappa / (self.kappa + N_j)) * self.m + (N_j / (self.kappa + N_j))
        left_eq = (self.kappa + N_j + 1)/((self.kappa+N_j)*(self.nu+N_j-self.feature_dim+1))
        middle_eq = torch.mm(self.S,(self.S).t())
        for j in self.classes_:
            X_j = X[y==j]
            N_j = X_j.shape[0]
            self.support_N_j_list.append(N_j)
            d = X_j.shape[1]
            mu_j = head_mu_eq * torch.mean(X_j, dim=0)
            S_j = torch.zeros_like(self.S)
        
            for i in range(X_j.shape[0]):
                S_j += torch.mm(X_j[i, :].unsqueeze(1), X_j[i, :].unsqueeze(0))         
            left_eq = 1.0 / (self.nu + N_j + d + 2)
            right_eq = middle_eq + S_j + self.kappa * torch.mm(self.m.unsqueeze(1), self.m.unsqueeze(0))-(self.kappa + N_j) * torch.mm(mu_j.unsqueeze(1), mu_j.unsqueeze(0))
            sigma_j = left_eq * right_eq

            sigma += sigma_j
            self.mu.append(mu_j)

        
#         if self.lda:
        sigma *= 1.0 / (len(self.classes_))
        sigma_inv = np.linalg.inv(sigma)
        self.sigma_inv = (1-self.reg_param) * sigma_inv + self.reg_param * np.eye(sigma_inv.shape[0])
        
    def predict(self, X):
        X = F.normalize(X, p=2, dim=1)
        # common code 
        N_j = self.support_N_j_list[0]
        common_term = self.nu+N_j+1
        left_t_eq = torch.lgamma(0.5*common_term)-torch.lgamma(0.5*(common_term-self.feature_dim))-0.5*self.feature_dim*torch.log(common_term-self.feature_dim)
        predicts_matrix=[]
        
        for i in range(X.shape[0]):
            neg_distrances=[]
            for j in range(len(self.classes_)):
                diff = X[i, :] - self.mu[j]
                gaussian_dist = torch.mm(torch.mm(diff.unsqueeze(0), self.sigma_inv), diff.unsqueeze(1))
                neg_distrances.append(-1*gaussian_dist)
            predicts_matrix.append(torch.cat(neg_distrances,dim=1))
        predicts_matrix = torch.cat(predicts_matrix,dim=0)
        
        return predicts_matrix
    
    def inv_matrix(self, matrix):
        sigma_inv=[]
        for sigma_j in matrix:
            sigma_inv.append(torch.inverse(sigma_j))
        return sigma_inv
    
    def reg_matrix(self, matrix):
        sigm_reg=[]
        for item in matrix:
            sigm_reg.append((1-self.reg_param) * item + self.reg_param * torch.eye(item.shape[0]).cuda())
        return sigm_reg
    
    def parameters(self):
        parameters = [self.m, self.S, self.nu, self.kappa]
        for p in parameters:
            yield p

