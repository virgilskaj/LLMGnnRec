import os
import numpy as np
from time import time
import pickle
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from utility.parser import parse_args
from utility.norm import build_sim, build_knn_normalized_graph
args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmerGAdvancedEnhancer(nn.Module):
    """
    Advanced EmerG enhancer specifically designed to improve R@10 and N@10/N@20
    Focus on early-stage ranking improvements
    """
    def __init__(self, embed_size, n_items):
        super(EmerGAdvancedEnhancer, self).__init__()
        self.embed_size = embed_size
        self.n_items = n_items
        
        # Enhanced item-specific modeling (more sophisticated than Lite version)
        self.item_specific_weights = nn.Embedding(n_items, 8)  # 8 interaction patterns
        
        # Multi-scale feature refinement (for better top-k performance)
        self.multi_scale_refiner = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_size, embed_size),
                nn.ReLU(),
                nn.Linear(embed_size, embed_size),
                nn.Dropout(args.drop_rate * 0.3)
            ) for _ in range(3)  # 3 different scales
        ])
        
        # Cross-modal attention for better ranking
        self.cross_attention = nn.MultiheadAttention(embed_size, num_heads=4, dropout=args.drop_rate * 0.5)
        
        # Ranking-focused enhancement
        self.ranking_enhancer = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
            nn.Sigmoid()  # For gating
        )
        
        # Initialize conservatively
        nn.init.xavier_uniform_(self.item_specific_weights.weight, gain=0.1)

    def forward(self, image_feats, text_feats, item_indices=None):
        """
        Advanced enhancement focusing on ranking performance
        """
        if item_indices is None:
            item_indices = torch.arange(min(self.n_items, image_feats.shape[0])).cuda()
        
        batch_size = image_feats.shape[0]
        if len(item_indices) != batch_size:
            item_indices = torch.arange(batch_size).cuda()
        
        # Get item-specific interaction patterns
        interaction_patterns = self.item_specific_weights(item_indices)  # [batch, 8]
        interaction_patterns = F.softmax(interaction_patterns, dim=-1)
        
        # Multi-scale feature refinement
        refined_features = []
        for refiner in self.multi_scale_refiner:
            refined_img = refiner(image_feats)
            refined_txt = refiner(text_feats)
            refined_features.extend([refined_img, refined_txt])
        
        # Cross-modal attention (for better feature interaction)
        stacked_feats = torch.stack([image_feats, text_feats], dim=1)  # [batch, 2, embed]
        attended_feats, _ = self.cross_attention(stacked_feats, stacked_feats, stacked_feats)
        attended_image, attended_text = attended_feats[:, 0], attended_feats[:, 1]
        
        # Item-specific feature combination (enhanced version)
        weights = interaction_patterns.unsqueeze(-1)  # [batch, 8, 1]
        
        # Combine multiple refined versions with learned weights
        enhanced_image = (weights[:, 0] * image_feats + 
                         weights[:, 1] * refined_features[0] +
                         weights[:, 2] * refined_features[2] +
                         weights[:, 3] * attended_image +
                         weights[:, 4] * text_feats * 0.1 +  # Cross-modal
                         weights[:, 5] * refined_features[1] * 0.1 +
                         weights[:, 6] * refined_features[3] * 0.1 +
                         weights[:, 7] * attended_text * 0.1)
        
        enhanced_text = (weights[:, 0] * text_feats +
                        weights[:, 1] * refined_features[1] +
                        weights[:, 2] * refined_features[3] +
                        weights[:, 3] * attended_text +
                        weights[:, 4] * image_feats * 0.1 +  # Cross-modal
                        weights[:, 5] * refined_features[0] * 0.1 +
                        weights[:, 6] * refined_features[2] * 0.1 +
                        weights[:, 7] * attended_image * 0.1)
        
        # Ranking-focused enhancement
        combined_feats = torch.cat([enhanced_image, enhanced_text], dim=-1)
        ranking_gates = self.ranking_enhancer(combined_feats)
        
        # Apply ranking enhancement (especially important for top-k metrics)
        enhanced_image = enhanced_image * (1.0 + 0.1 * ranking_gates)
        enhanced_text = enhanced_text * (1.0 + 0.1 * ranking_gates)
        
        return enhanced_image, enhanced_text


class MM_Model_Final_Optimized(nn.Module):
    """
    Final optimized version targeting all metrics improvement
    Special focus on R@10 and NDCG improvements
    """
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats, user_init_embedding, item_attribute_dict):

        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size

        # EXACTLY the same as original MM_Model
        self.image_trans = nn.Linear(image_feats.shape[1], args.embed_size)
        self.text_trans = nn.Linear(text_feats.shape[1], args.embed_size)
        self.user_trans = nn.Linear(user_init_embedding.shape[1], args.embed_size)  
        self.item_trans = nn.Linear(item_attribute_dict['title'].shape[1], args.embed_size)  
        nn.init.xavier_uniform_(self.image_trans.weight)
        nn.init.xavier_uniform_(self.text_trans.weight)   
        nn.init.xavier_uniform_(self.user_trans.weight)
        nn.init.xavier_uniform_(self.item_trans.weight)

        self.user_id_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        self.image_feats = torch.tensor(image_feats).float().cuda()
        self.text_feats = torch.tensor(text_feats).float().cuda()
        self.user_feats = torch.tensor(user_init_embedding).float().cuda()
        self.item_feats = {}
        for key in item_attribute_dict.keys():                                   
            self.item_feats[key] = torch.tensor(item_attribute_dict[key]).float().cuda() 

        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.Sigmoid()  
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.batch_norm = nn.BatchNorm1d(args.embed_size)
        self.tau = 0.5

        # Advanced EmerG enhancer for all metrics improvement
        self.use_enhancement = getattr(args, 'use_enhanced_gnn', False)
        if self.use_enhancement:
            self.advanced_enhancer = EmerGAdvancedEnhancer(args.embed_size, n_items)
            
            # Additional ranking-focused components
            self.top_k_booster = nn.Sequential(
                nn.Linear(args.embed_size, args.embed_size),
                nn.ReLU(),
                nn.Linear(args.embed_size, args.embed_size),
                nn.Sigmoid()
            )

    def mm(self, x, y):
        if args.sparse:
            return torch.sparse.mm(x, y)
        else:
            return torch.mm(x, y)
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def batched_contrastive_loss(self, z1, z2, batch_size=4096):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  
            between_sim = f(self.sim(z1[mask], z2))  

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
                   
        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum+1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)

        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum+1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)

        if mean_flag == False:
            return rowsum_diag*csr_mat*colsum_diag
        else:
            return rowsum_diag*csr_mat

    def matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  #
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  #
        values = torch.from_numpy(cur_matrix.data)  #
        shape = torch.Size(cur_matrix.shape)

        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()  #

    def para_dict_to_tenser(self, para_dict):  
        """
        :param para_dict: nn.ParameterDict()
        :return: tensor
        """
        tensors = []

        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        tensors = torch.stack(tensors, dim=0)

        return tensors


    def forward(self, ui_graph, iu_graph, image_ui_graph, image_iu_graph, text_ui_graph, text_iu_graph):

        # EXACTLY as original - feature mask 
        i_mask_nodes, u_mask_nodes = None, None
        if args.mask:
            i_perm = torch.randperm(self.n_items)
            i_num_mask_nodes = int(args.mask_rate * self.n_items)
            i_mask_nodes = i_perm[: i_num_mask_nodes]
            for key in self.item_feats.keys():
                self.item_feats[key][i_mask_nodes] = self.item_feats[key].mean(0)

        u_perm = torch.randperm(self.n_users)
        u_num_mask_nodes = int(args.mask_rate * self.n_users)
        u_mask_nodes = u_perm[: u_num_mask_nodes]
        self.user_feats[u_mask_nodes] = self.user_feats.mean(0) 

        # EXACTLY as original - feature transformation
        image_feats = self.dropout(self.image_trans(self.image_feats))
        text_feats = self.dropout(self.text_trans(self.text_feats))
        user_feats = self.dropout(self.user_trans(self.user_feats.to(torch.float32)))
        item_feats = {}
        for key in self.item_feats.keys():
            item_feats[key] = self.dropout(self.item_trans(self.item_feats[key]))

        # Advanced EmerG Enhancement targeting weak metrics
        if self.use_enhancement:
            try:
                enhanced_image_feats, enhanced_text_feats = self.advanced_enhancer(image_feats, text_feats)
                
                # Additional top-k focused enhancement
                image_boost = self.top_k_booster(enhanced_image_feats)
                text_boost = self.top_k_booster(enhanced_text_feats)
                
                # Blend original and enhanced features with learned weights
                alpha = 0.15  # Slightly more aggressive than Lite version
                image_feats = (1 - alpha) * image_feats + alpha * enhanced_image_feats * image_boost
                text_feats = (1 - alpha) * text_feats + alpha * enhanced_text_feats * text_boost
                
            except Exception as e:
                print(f"Advanced enhancement failed: {e}")

        # EXACTLY as original - graph propagation  
        for i in range(args.layers):
            image_user_feats = self.mm(ui_graph, image_feats)
            image_item_feats = self.mm(iu_graph, image_user_feats)

            text_user_feats = self.mm(ui_graph, text_feats)
            text_item_feats = self.mm(iu_graph, text_user_feats)

        # EXACTLY as original - aug item attribute
        user_feat_from_item = {}
        for key in self.item_feats.keys():
            user_feat_from_item[key] = self.mm(ui_graph, item_feats[key])
            item_feats[key] = self.mm(iu_graph, user_feat_from_item[key])

        # EXACTLY as original - aug user profile
        item_prof_feat = self.mm(iu_graph, user_feats)
        user_prof_feat = self.mm(ui_graph,  item_prof_feat)

        # EXACTLY as original - ID embeddings
        u_g_embeddings = self.user_id_embedding.weight
        i_g_embeddings = self.item_id_embedding.weight             

        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]
        for i in range(self.n_ui_layers):    
            if i == (self.n_ui_layers-1):
                u_g_embeddings = self.softmax( torch.mm(ui_graph, i_g_embeddings) ) 
                i_g_embeddings = self.softmax( torch.mm(iu_graph, u_g_embeddings) )
            else:
                u_g_embeddings = torch.mm(ui_graph, i_g_embeddings) 
                i_g_embeddings = torch.mm(iu_graph, u_g_embeddings) 

            user_emb_list.append(u_g_embeddings)
            item_emb_list.append(i_g_embeddings)

        u_g_embeddings = torch.mean(torch.stack(user_emb_list), dim=0)
        i_g_embeddings = torch.mean(torch.stack(item_emb_list), dim=0) 

        # EXACTLY as original - final fusion
        u_g_embeddings = u_g_embeddings + args.model_cat_rate*F.normalize(image_user_feats, p=2, dim=1) + args.model_cat_rate*F.normalize(text_user_feats, p=2, dim=1)
        i_g_embeddings = i_g_embeddings + args.model_cat_rate*F.normalize(image_item_feats, p=2, dim=1) + args.model_cat_rate*F.normalize(text_item_feats, p=2, dim=1)
        # profile
        u_g_embeddings += args.user_cat_rate*F.normalize(user_prof_feat, p=2, dim=1)
        i_g_embeddings += args.user_cat_rate*F.normalize(item_prof_feat, p=2, dim=1)

        # attribute 
        for key in self.item_feats.keys():
            u_g_embeddings += args.item_cat_rate*F.normalize(user_feat_from_item[key], p=2, dim=1)
            i_g_embeddings += args.item_cat_rate*F.normalize(item_feats[key], p=2, dim=1) 

        return u_g_embeddings, i_g_embeddings, image_item_feats, text_item_feats, image_user_feats, text_user_feats, user_feats, item_feats, user_prof_feat, item_prof_feat, user_feat_from_item, item_feats, i_mask_nodes, u_mask_nodes


class Decoder(nn.Module):
    def __init__(self, feat_size):
        super(Decoder, self).__init__()
        self.feat_size=feat_size

        self.u_net = nn.Sequential(
            nn.Linear(args.embed_size, int(self.feat_size)),
            nn.LeakyReLU(True),
        )

        self.i_net = nn.Sequential(
            nn.Linear(args.embed_size, int(self.feat_size)),
            nn.LeakyReLU(True),
        )

    def forward(self, u, i):
        u_output = self.u_net(u.float())  
        tensor_list = []
        for index,value in enumerate(i.keys()):  
            tensor_list.append(i[value]) 
        i_tensor = torch.stack(tensor_list)
        i_output = self.i_net(i_tensor.float())  
        return u_output, i_output