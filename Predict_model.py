# -*- coding: utf-8 -*-
"""
Author :ZhangX
Date   :2025 07 16


"""
from __future__ import print_function
import torch 
import numpy as np
from math import *
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import pandas as pd  #use pandas Read the xls file
from torch.autograd import Variable
import random
from scipy.spatial import KDTree
from torchinfo import summary
from torchsummary import summary
import os


torch.set_default_tensor_type(torch.DoubleTensor)

Model_number = 71

# Read point cloud data
file_path = 'D:\\PYTHON\\Artery\\Data\\Artery_650_processed.xls'
sheet = pd.read_excel(file_path, sheet_name=None)
#print(sheet)

# Read damage data
damage_sheet = pd.read_excel(file_path, sheet_name='damage')
damage_data = {}
for i in range(Model_number):
    model_name = f'artery{i+1}'
    row = damage_sheet[damage_sheet['index'] == model_name]
    if not row.empty:
        damage_data[i+1] = {
            'avg_damage': row['average damage in the media layer'].values[0],
            'max_damage': row['maximum damage in the media layer'].values[0]
        }


#Process point cloud data
for sheet_name, df in sheet.items():
    if sheet_name.startswith('Artery'):
        df = df.astype(float)
        number = sheet_name[6:]
        globals()[f'data{number}'] = df
        
# df_list = [globals()[f'data{i}'] for i in range(1, 91)]
# row_counts = [len(df) for df in df_list]


# The number of sampling points
num_sample_point = 650


def sample_point_cloud(data, num_sample_point, num_z_bins=10, num_angular_bins=8):
    """
 
    """
    points = data.values
    n = len(points)
    
    if n == num_sample_point:
        return points
    
    # 
    initial_points = points[:, :3]  
    deformed_points = points[:, 3:]  
    
    # Calculate the Z-axis range and stratify
    z_min, z_max = np.min(initial_points[:, 2]), np.max(initial_points[:, 2])
    z_range = z_max - z_min
    z_bins = np.linspace(z_min, z_max, num_z_bins + 1)

    # Define the weights of the two end regions    
    end_zone_ratio = 0.15  # 
    end_zone_width = z_range * end_zone_ratio
    
    # Store sampling points for each layer
    sampled_points = []
    layer_weights = []  # 
    layer_point_counts = [] 

    for i in range(num_z_bins):
        z_low = z_bins[i]
        z_high = z_bins[i+1]
        layer_mask = (initial_points[:, 2] >= z_low) & (initial_points[:, 2] < z_high)
        layer_points = initial_points[layer_mask]
        
        if len(layer_points) == 0:
            layer_weights.append(0)
            layer_point_counts.append(0)
            continue
            
        # 
        layer_center = (z_low + z_high) / 2
        
        # 
        dist_to_end = min(
            abs(layer_center - z_min),
            abs(layer_center - z_max)
        )
        
        # 
        weight = 3.0 if dist_to_end < end_zone_width else 1.0
        layer_weights.append(weight)
        layer_point_counts.append(len(layer_points))
    
    # 
    total_weight = sum(weight * count for weight, count in zip(layer_weights, layer_point_counts))
    
    # 
    for i in range(num_z_bins):
        z_low = z_bins[i]
        z_high = z_bins[i+1]
        layer_mask = (initial_points[:, 2] >= z_low) & (initial_points[:, 2] < z_high)
        layer_initial = initial_points[layer_mask]  # 当
        layer_deformed = deformed_points[layer_mask]  # 
        
        if len(layer_initial) == 0:
            continue
            
        # Calculate the number of points that should be sampled in the current layer (considering weights)
        weighted_fraction = (layer_weights[i] * len(layer_initial)) / total_weight
        layer_target = max(1, int(round(num_sample_point * weighted_fraction)))
        
        # 在XY平面计算角度并分区
        angles = np.arctan2(layer_initial[:, 1], layer_initial[:, 0])  # [-π, π]
        angles = np.where(angles < 0, angles + 2 * np.pi, angles)  # [0, 2π]
        angle_bins = np.linspace(0, 2 * np.pi, num_angular_bins + 1)
        
        # Uniform sampling within the angle partition
        sector_sampled_list = []  # 存储当前层各扇区的采样点
        
        for j in range(num_angular_bins):
            angle_low = angle_bins[j]
            angle_high = angle_bins[j+1]
            
            # 
            angle_mask = (angles >= angle_low) & (angles < angle_high)
            sector_initial = layer_initial[angle_mask]  # 
            sector_deformed = layer_deformed[angle_mask]  # 
            
            if len(sector_initial) == 0:
                continue
                
            # 
            sector_target = max(1, int(round(layer_target * len(sector_initial) / len(layer_initial))))
            
            # 
            if len(sector_initial) >= sector_target:
                indices = np.random.choice(len(sector_initial), sector_target, replace=False)
                selected_initial = sector_initial[indices]
                selected_deformed = sector_deformed[indices]
            # 
            else:
                selected_initial = sector_initial
                selected_deformed = sector_deformed
            
            # 
            sector_sampled = np.hstack((selected_initial, selected_deformed))
            sector_sampled_list.append(sector_sampled)
        
        # 
        if sector_sampled_list:
            layer_sampled = np.vstack(sector_sampled_list)
            sampled_points.append(layer_sampled)
    
    # 
    if sampled_points:
        sampled_points = np.vstack(sampled_points)
    else:
        # 
        return sample_point_cloud_fallback(data, num_sample_point)
    
    # 
    total_sampled = len(sampled_points)
    if total_sampled > num_sample_point:
        # 
        indices = np.random.choice(total_sampled, num_sample_point, replace=False)
        sampled_points = sampled_points[indices]
    elif total_sampled < num_sample_point:
        # 
        num_needed = num_sample_point - total_sampled
        indices = np.random.choice(total_sampled, num_needed, replace=True)
        sampled_points = np.vstack([sampled_points, sampled_points[indices]])
    
    return sampled_points

def sample_point_cloud_fallback(data, num_sample_point):
    """"""
    n = len(data)
    if n >= num_sample_point:
        indices = np.random.choice(n, num_sample_point, replace=False)
        return data.iloc[indices].values
    else:
        # 
        indices = np.random.choice(n, num_sample_point, replace=True)
        return data.iloc[indices].values


for i in range(1, Model_number + 1):
    data_var_name = f'data{i}'
    if data_var_name in globals():
        # 
        globals()[data_var_name] = sample_point_cloud(
            globals()[data_var_name], 
            num_sample_point,
            num_z_bins=10,      # 
            num_angular_bins=8   # 
            )

#Initial point coordinates, deformed point coordinates
pointcoordinate_initial = ['x0', 'y0', 'z0']
pointcoordinate_deformed = ['x1', 'y1', 'z1']
#1
data_deformed = {}
data_initial = {}

for i in range(1, Model_number+1): 
    data_var_name = f'data{i}'
    if data_var_name in globals():
        data = globals()[data_var_name]
        globals()[f'{data_var_name}_deformed'] = data[:, 3:6]
        globals()[f'{data_var_name}_initial'] = data[:, 0:3]






###############       Normalize the point cloud    ################### 

def normalize_point_cloud(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc_normalized = pc / m
    # 确保输出为float64类型
    return pc_normalized.astype(np.float64), centroid.astype(np.float64), float(m)


# 初始化存储结果的列表
data_initial_normalized_list = []
data_deformed_normalized_list = []
centroid_initial_list = []
centroid_deformed_list = []
m_initial_list = []
m_deformed_list = []
data_list = []
damage_avg_list = []
damage_max_list = []

# 处理数据集
for i in range(1, Model_number+1):
    
    data_var = globals().get(f'data{i}')
    if data_var is not None:
        data_list.append(data_var)
    
    # 处理变形数据
    deformed_data = globals()[f'data{i}_deformed']
    deformed_normalized, centroid_deformed, m_deformed = normalize_point_cloud(deformed_data)
    data_deformed_normalized_list.append(deformed_normalized)
    centroid_deformed_list.append(centroid_deformed)
    m_deformed_list.append(m_deformed)
    # 可选保存: np.savetxt(f"data{i}_deformed_normalized.txt", deformed_normalized, fmt='%f', delimiter=',')
    
    # 处理初始数据
    initial_data = globals()[f'data{i}_initial']
    initial_normalized, centroid_initial, m_initial = normalize_point_cloud(initial_data)
    data_initial_normalized_list.append(initial_normalized)
    centroid_initial_list.append(centroid_initial)
    m_initial_list.append(m_initial)
    
    if i in damage_data:
        damage_avg_list.append(damage_data[i]['avg_damage'])
        damage_max_list.append(damage_data[i]['max_damage'])
    else:
        # 如果找不到损伤数据，使用默认值
        damage_avg_list.append(0.0)
        damage_max_list.append(0.0)
        


 

# 转换为与原始代码完全一致的NumPy数组和命名
data = np.array(data_list)
data_initial = np.array(data_initial_normalized_list, dtype=np.float64)
data_deformed = np.array(data_deformed_normalized_list, dtype=np.float64)
centroid_initial = np.array(centroid_initial_list)
centroid_deformed = np.array(centroid_deformed_list)
m_initial = np.array(m_initial_list)
m_deformed = np.array(m_deformed_list)
damage_avg = np.array(damage_avg_list, dtype=np.float64)
damage_max = np.array(damage_max_list, dtype=np.float64)
############### Generate normalized coordinates before and after deformation


np.save("data_random_sample_point.npy", data)
np.save("data_initial.npy", data_initial)
np.save("data_deformed.npy", data_deformed)
np.save("centroid_initial.npy", centroid_initial)
np.save("centroid_deformed.npy", centroid_deformed)
np.save("m_initial.npy", m_initial)
np.save("m_deformed.npy", m_deformed)
np.save("damage_avg.npy", damage_avg)
np.save("damage_max.npy", damage_max)

'''
Read and store float64 data, transfer it to CUDA
'''
data = np.load("data_random_sample_point.npy", allow_pickle = True)
data_initial = np.load("data_initial.npy", allow_pickle = True)
data_deformed = np.load("data_deformed.npy", allow_pickle = True)

data_initial = torch.tensor(data_initial, dtype=torch.float64).cuda().transpose(2, 1)
data_deformed = torch.tensor(data_deformed, dtype=torch.float64).cuda().transpose(2, 1)
damage_avg = torch.tensor(np.load("damage_avg.npy", allow_pickle=True), dtype=torch.float64).cuda()
damage_max = torch.tensor(np.load("damage_max.npy", allow_pickle=True), dtype=torch.float64).cuda()

# 创建损伤目标张量 [batch_size, 2]
damage_targets = torch.stack([damage_avg, damage_max], dim=1)



#'''
#
##########     Increase data by rotating around the origin point  #########
#
#
#''''
#
def rotate_pointcloud_batch(data_initial_batch, data_deformed_batch):
    """"""
    B, C, N = data_initial_batch.shape
    device = data_initial_batch.device
    
    # Generate random rotation angle (0-360 degrees)
    angles = torch.rand(B, 3) * 2 * np.pi  # [B, 3]
    
    rotated_initial = []
    rotated_deformed = []
    
    for i in range(B):
        alpha, beta, gamma = angles[i]
        
        # 
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(alpha), -torch.sin(alpha)],
            [0, torch.sin(alpha), torch.cos(alpha)]
        ], dtype=torch.float64, device=device)
        
        Ry = torch.tensor([
            [torch.cos(beta), 0, torch.sin(beta)],
            [0, 1, 0],
            [-torch.sin(beta), 0, torch.cos(beta)]
        ], dtype=torch.float64, device=device)
        
        Rz = torch.tensor([
            [torch.cos(gamma), -torch.sin(gamma), 0],
            [torch.sin(gamma), torch.cos(gamma), 0],
            [0, 0, 1]
        ], dtype=torch.float64, device=device)
        
        R = torch.mm(torch.mm(Rz, Ry), Rx)  # 
        
        # 
        initial = data_initial_batch[i]  # [3, N]
        rotated_initial_i = torch.mm(R, initial)
        
        # 
        deformed = data_deformed_batch[i]  # [3, N]
        rotated_deformed_i = torch.mm(R, deformed)
        
        rotated_initial.append(rotated_initial_i)
        rotated_deformed.append(rotated_deformed_i)
    
    return torch.stack(rotated_initial), torch.stack(rotated_deformed)
#        
#        
#Rotate_pointcloud(data_tensor,30,30,30, pointcloud_initial, pointcloud_deformed)


'''

                             ########   Regression PointNet model   #########


''' 
# T-Net: is a pointnet itself.Obtain a 3x3 transformation matrix to correct the pose of the point cloud; the effect is average, and PointNet2 did not incorporate this part
# Map to 9 data points through a fully connected layer, and finally adjust to a 3x3 matrix
#torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode=‘zeros’)
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)


        iden = torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32)).view(1,9).repeat(batchsize,1)

        #Send the identity matrix to the GPU
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
         # view: reconstruct the tensor dimensions, where -1 indicates that the missing parameter is automatically calculated by the system (which is the size of the batch size)
        # The returned result is of size batchsize x 3 x 3
        x = x.view(-1, 3, 3)
        return x

# The data is k-dimensional, used for high-dimensional features after the MLP (Multilayer Perceptron)
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        #iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(batchsize, 1)
        iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1, self.k * self.k).repeat(batchsize, 1)

        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


     
class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=False, feature_transform=True, channel=3):
        '''
        global_feat=True indicates that it is for classification ,False  for segmentation
        feature_transform=True  This indicates that a transformation of the feature space is required, equivalent to the switch for the STNkd module
        '''
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        
        B, D, N = x.size()#Return the scale of each dimension of the tensor, where the second dimension is called dimension (channel)
        trans = self.stn(x)  #Obtain a 3x3 coordinate transformation matrix
        x = x.transpose(2, 1)  #Adjust the dimension of the points, convert the point cloud data into the nx3 format, which facilitates computation with the rotation matrix. The transformation is from #B C N to B N C, placing the channel dimension last
        
        if D > 3:  #If the input is greater than three-dimensional, only the first three dimensions are processed: XYZ (STN transforms the coordinates)
            feature = x[:, :, 3:]  #Features start from the fourth dimension
            x = x[:, :, :3]
        x = torch.bmm(x, trans)  #tensor*matrix  The purpose of the STN (Spatial Transformer Network) is to estimate a spatial transformation and then apply this spatial transformation to x
        if D > 3:
            x = torch.cat([x, feature], dim=2)  #After processing, the features are concatenated back together
        x = x.transpose(2, 1)  #the channels are swapped back to the B C N
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:  
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)  
            x = torch.bmm(x, trans_feat)  
            x = x.transpose(2, 1)  
        else:  
            trans_feat = None

        pointfeat = x #Preserve the features after the first MLP for subsequent segmentation, facilitating feature concatenation and fusion
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)  #the global_feature after maxpooling in the architecture diagram
        if self.global_feat:  #If it is for classification, directly return x, which is the global_feature
            return x, trans, trans_feat  #trans:The estimated transformation matrix for 3D space  trans_feat:The estimated transformation matrix for the feature space
        else:  #If it is for segmentation, x needs to be replicated N times along the N-point dimension (and then concatenated with the features obtained after the second MLP, which serves as the input to the segmentation network)
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):   #When aligning features, since the feature space has a higher dimension, the optimization is more challenging. Therefore, a regularization term is added to make the solved affine transformation matrix closer to orthogonal
    d = trans.size()[1]  #channels number
    # I is a 3-dimensional tensor with ones on the diagonal and no data in dimension 0
    I = torch.eye(d)[None, :, :]  # increase channels
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
          #torch.bmm performs 3D tensor multiplication, where the first dimension is indexed, and the last two dimensions are matrix multiplied

# 添加残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)
    
# 
class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256)
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256)
        )
        
    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        return torch.cat([f1, f2, f3], dim=1)
    
# 
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, N = x.size()
        
        # 
        avg_out = self.avg_pool(x).view(B, C)
        avg_out = self.fc(avg_out).view(B, C, 1)
        
        # 
        max_out = self.max_pool(x).view(B, C)
        max_out = self.fc(max_out).view(B, C, 1)
        
        # 
        attention_weights = avg_out + max_out
        return x * attention_weights.expand_as(x)

class MultiTaskPointNet(nn.Module):
    def __init__(self, channel=3, point_out_num=3, damage_out_num=2):
        super(MultiTaskPointNet, self).__init__()
        #self.out_num = out_num
        self.stn = STN3d(channel)
        
        # 
        self.res1 = ResidualBlock(channel, 64)
        self.res2 = ResidualBlock(64, 128)
        self.res3 = ResidualBlock(128, 256)
        self.attention = ChannelAttention(256)  # 添加通道注意力机制
        self.multi_scale = MultiScaleFeatureExtractor(256)
        self.conv4 = torch.nn.Conv1d(768, 512, 1)  # 768 = 256*3
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)        
        self.fstn = STNkd(k=256)
        
        # 
        self.point_branch = nn.Sequential(
            nn.Conv1d(5056, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, point_out_num, 1)
        )
        
        # 
        self.damage_branch = nn.Sequential(
            nn.Linear(2048, 1024),  # 
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, damage_out_num)
        )
         # 
        self.aux_damage_branch = nn.Sequential(
            nn.Conv1d(2048, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, damage_out_num)
        )
            

    def forward(self, point_cloud):
        B, C, N = point_cloud.size()
        # 共
        z_coords = point_cloud[:, 2, :].clone()
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        
        if C > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud = torch.bmm(point_cloud, trans)
        if C > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)

        point_cloud = point_cloud.transpose(2, 1)

        out1 = self.res1(point_cloud)
        out2 = self.res2(out1)
        out3 = self.res3(out2)
        out3 = self.attention(out3)
        multi_features = self.multi_scale(out3)
        
        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4(multi_features)))
        out5 = self.bn5(self.conv5(out4))
        
        # 全
        global_feature = torch.max(out5, 2)[0]  # [B, 2048]
        
        # 
        out_max = torch.max(out5, 2, keepdim=True)[0]
        expand = out_max.repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        point_output = self.point_branch(concat)
        point_output = point_output.transpose(2, 1).contiguous()
        
        # 
        damage_output = self.damage_branch(global_feature)
        aux_damage_output = self.aux_damage_branch(out5)
        
        return point_output, damage_output, aux_damage_output, trans_feat, z_coords


    
class MultiTaskLoss(nn.Module):
    def __init__(self, point_loss_scale=1.0, damage_loss_scale=1.0, 
                 aux_damage_loss_scale=0.5, mat_diff_loss_scale=0.001,
                 end_weight_factor=3.0, end_zone_ratio=0.15):
        super(MultiTaskLoss, self).__init__()
        self.point_loss = nn.SmoothL1Loss()
        self.damage_loss = nn.SmoothL1Loss()
        self.aux_damage_loss = nn.SmoothL1Loss()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.point_loss_scale = point_loss_scale
        self.damage_loss_scale = damage_loss_scale
        self.aux_damage_loss_scale = aux_damage_loss_scale
        self.end_weight_factor = end_weight_factor
        self.end_zone_ratio = end_zone_ratio

    def forward(self, point_pred, point_target, damage_pred, damage_target, 
                aux_damage_pred, trans_feat, z_coords):
        
        device = point_pred.device
        
        
        print(f"[Loss] point_pred shape: {point_pred.shape}")
        print(f"[Loss] point_target shape: {point_target.shape}")
        print(f"[Loss] z_coords shape: {z_coords.shape}")
        
        # 
        if point_pred.dim() == 3:
            if point_pred.size(1) == 3:  # [B, 3, N] 格式
                B, _, N = point_pred.shape
                point_pred = point_pred.transpose(1, 2)  # 转为 [B, N, 3]
                point_target = point_target.transpose(1, 2)  # 转为 [B, N, 3]
            elif point_pred.size(2) == 3:  # [B, N, 3] 格式
                B, N, _ = point_pred.shape
            else:
                raise ValueError(f"Unrecognized point prediction shape: {point_pred.shape}")
        else:
            raise ValueError(f"Point prediction tensor dimension error: {point_pred.dim()}")
        
        # 
        point_loss_unreduced = F.smooth_l1_loss(
            point_pred, point_target, reduction='none'
        )  # [B, N, 3]
        
        # 
        if z_coords.dim() == 3:
            if z_coords.size(1) == 1:  # [B, 1, N]
                z_coords = z_coords.squeeze(1)  # [B, N]
            elif z_coords.size(2) == 1:  # [B, N, 1]
                z_coords = z_coords.squeeze(2)  # [B, N]
            elif z_coords.size(1) == 3:  # [B, 3, N]
                z_coords = z_coords[:, 2, :]  # 提取 z 坐标 [B, N]
            elif z_coords.size(2) == 3:  # [B, N, 3]
                z_coords = z_coords[:, :, 2]  # 提取 z 坐标 [B, N]
            else:
                # 尝
                z_coords = z_coords.view(B, N)
        elif z_coords.dim() == 2:
            # 
            pass
        else:
            # 
            z_coords = z_coords.view(B, N)
        
        print(f"[Loss] z_coords reshaped: {z_coords.shape}")
        
        # 
        if z_coords.shape != (B, N):
            # 
            if z_coords.numel() == B * N:
                z_coords = z_coords.view(B, N)
            else:
                raise ValueError(
                    f"z_coords 形状 {z_coords.shape} 与期望形状 [{B}, {N}] 不匹配"
                )
        
        # 
        z_min = z_coords.min(dim=1, keepdim=True)[0]  # 保持维度用于广播
        z_max = z_coords.max(dim=1, keepdim=True)[0]
        z_range = z_max - z_min
        end_zone = z_range * self.end_zone_ratio
        
        lower_end_mask = z_coords < (z_min + end_zone)
        upper_end_mask = z_coords > (z_max - end_zone)
        end_mask = lower_end_mask | upper_end_mask
        

        weights = torch.ones(B, N, device=device)
        

        weights = torch.where(
            end_mask, 
            torch.full_like(weights, self.end_weight_factor), 
            weights
        )
        

        weights_expanded = weights.unsqueeze(-1).expand_as(point_loss_unreduced)
        weighted_point_loss = weights_expanded * point_loss_unreduced
        point_loss = torch.mean(weighted_point_loss)

        
        damage_loss = self.damage_loss(damage_pred, damage_target)
        
        
        aux_damage_loss = self.aux_damage_loss(aux_damage_pred, damage_target)
        
        
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        
        
        total_loss = (self.point_loss_scale * point_loss + 
                     self.damage_loss_scale * damage_loss + 
                     self.aux_damage_loss_scale * aux_damage_loss + 
                     self.mat_diff_loss_scale * mat_diff_loss)
        
        return total_loss, point_loss, damage_loss, aux_damage_loss 




print("Input data dimensions：  " , data_initial.size())
print("Target value dimensions：  " , data_deformed.size())
batch, featuresize, pointnumber = data_initial.size()



















#####################   #Train Model       ####################################
def train_multi_task_model(model, input_data, point_target, damage_target, epochs, learning_rate, optimizer, criterion, scheduler=None):
    print("########### Train Begin #############")
    best_loss = float('inf')
    patience = 500
    patience_counter = 0
    
    # 
    log_file = open("training_log.csv", "w")
    log_file.write("Epoch,TotalLoss,PointLoss,DamageLoss,LearningRate\n")
    
    # 
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 
        rotated_input, rotated_point_target = rotate_pointcloud_batch(input_data, point_target)
        
        # 
        point_pred, damage_pred, aux_damage_pred, trans_feat, z_coords = model(rotated_input)
        
        # 
        if rotated_point_target.dim() == 3 and rotated_point_target.size(1) == 3:
            rotated_point_target = rotated_point_target.transpose(1, 2)  # [B, N, 3]
        
        # 
        total_loss, point_loss, damage_loss, aux_loss = criterion(
            point_pred, 
            rotated_point_target,  # 
            damage_pred, 
            damage_target,
            aux_damage_pred,
            trans_feat,
            z_coords  
        )
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        # 
        if scheduler:
            scheduler.step(total_loss.item())
            
        # 
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_point_pred, val_damage_pred, val_aux_damage_pred, val_trans_feat, val_z_coords = model(input_data)
                
                # 
                val_point_target = data_deformed
                if val_point_target.dim() == 3 and val_point_target.size(1) == 3:
                    val_point_target = val_point_target.transpose(1, 2)  # [B, N, 3]
                
                val_total_loss, _, _, _ = criterion(
                    val_point_pred, 
                    val_point_target,  # 
                    val_damage_pred, 
                    damage_targets,
                    val_aux_damage_pred,
                    val_trans_feat,
                    val_z_coords
                )
            val_losses.append(val_total_loss.item())
            model.train()
        
        # 
        current_lr = optimizer.param_groups[0]['lr']
        log_file.write(f"{epoch},{total_loss.item()},{point_loss.item()},{damage_loss.item()},{current_lr}\n")
        log_file.flush()
        
        # 
        if epoch % 1 == 0:
            if val_total_loss.item() < best_loss:
                best_loss = val_total_loss.item()
                patience_counter = 0
                torch.save(model.state_dict(), 'best_multi_task_model.pkl')
                print(f"Epoch {epoch}: New best validation loss {best_loss:.6f}")
            else:
                patience_counter += 1
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}, "
                  f"Val Loss: {val_total_loss.item():.6f}, "
                  f"Point Loss: {point_loss.item():.6f}, Damage Loss: {damage_loss.item():.6f}, "
                  f"LR: {current_lr:.6f}")
        
        if patience_counter >= patience:  # 
            print(f"Early stopping at epoch {epoch}")
            break
    
    log_file.close()
    print("Training complete. Best model saved as 'best_multi_task_model.pkl'")





# 
if __name__ == "__main__":
    RPN = MultiTaskPointNet().cuda()
    
    # 
    def init_weights(m):
        if type(m) in [nn.Conv1d, nn.Linear]:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)# 避免零初始化偏置
    
    RPN.apply(init_weights)
    
    epochs = 5000
    learning_rate = 0.001
    # Learning Rate Scheduler
    optimizer = torch.optim.AdamW(RPN.parameters(), lr=learning_rate, betas=(0.9, 0.999),eps=1e-8,weight_decay=1e-2)
    
    # 
    criterion = MultiTaskLoss(
        point_loss_scale=1.0,      # 
        damage_loss_scale=1.0,      # 
        aux_damage_loss_scale=0.5,  # 
        mat_diff_loss_scale=0.001
    )
    
    # 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=100, 
        verbose=True
    )
    
    # 
    train_multi_task_model(
        model=RPN,
        input_data=data_initial,
        point_target=data_deformed,
        damage_target=damage_targets,
        epochs=epochs,
        learning_rate=learning_rate,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler
    )

torch.save(RPN.state_dict(), 'RPN_model.pkl')
print("Saved as 'RPN_model.pkl'")
