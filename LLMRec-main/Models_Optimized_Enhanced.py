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

# Use try-except to handle args parsing in case of import issues
try:
    args = parse_args()
except:
    # Create a mock args object for testing
    class MockArgs:
        def __init__(self):
            self.embed_size = 64
            self.drop_rate = 0.1
            self.layers = 1
            self.sparse = 1
            self.mask = False
            self.mask_rate = 0.0
            self.model_cat_rate = 0.02
            self.user_cat_rate = 2.8
            self.item_cat_rate = 0.005
    args = MockArgs()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdaptiveFeatureEnhancer(nn.Module):
    """
    Adaptive feature enhancer inspired by EmerG's item-specific modeling
    Focus on learning adaptive feature weights rather than complex graph generation
    """
    def __init__(self, embedding_dim, num_features=2):
        super(AdaptiveFeatureEnhancer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        
        # Lightweight feature adaptation (inspired by EmerG's hypernetworks)
        self.feature_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 4),
                nn.ReLU(),
                nn.Linear(embedding_dim // 4, embedding_dim),
                nn.Sigmoid()  # Use sigmoid to ensure positive scaling
            ) for _ in range(num_features)
        ])
        
        # Cross-feature interaction (simplified from EmerG's graph interaction)
        self.cross_interaction = nn.Sequential(
            nn.Linear(embedding_dim * num_features, embedding_dim),
            nn.ReLU(),
            nn.Dropout(args.drop_rate * 0.5),  # Lighter dropout
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Gating mechanism to control enhancement strength
        self.enhancement_gate = nn.Sequential(
            nn.Linear(embedding_dim * num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, modal_features):
        """
        Adaptively enhance modal features
        Args:
            modal_features: list of [n_items, embedding_dim] tensors
        Returns:
            enhanced_features: list of enhanced modal features
        """
        enhanced_features = []
        
        # Step 1: Apply adaptive scaling to each modality
        for i, modal_feat in enumerate(modal_features):
            # Learn adaptive weights for this modality
            adaptive_weights = self.feature_adapters[i](modal_feat)
            # Apply adaptive scaling (multiplicative enhancement)
            enhanced_feat = modal_feat * (1.0 + 0.1 * adaptive_weights)  # Small enhancement
            enhanced_features.append(enhanced_feat)
        
        # Step 2: Cross-modal interaction (inspired by EmerG's feature interaction)
        if len(enhanced_features) >= 2:
            # Concatenate features for interaction learning
            concat_features = torch.cat(enhanced_features, dim=-1)
            
            # Learn cross-modal interactions
            interaction_output = self.cross_interaction(concat_features)
            
            # Compute gating weight to control enhancement strength
            gate_weight = self.enhancement_gate(concat_features)
            
            # Apply gated enhancement to all modalities
            for i in range(len(enhanced_features)):
                # Add controlled interaction enhancement
                enhanced_features[i] = enhanced_features[i] + gate_weight * 0.05 * interaction_output
        
        return enhanced_features


class HighOrderFeatureInteraction(nn.Module):
    """
    High-order feature interaction module inspired by EmerG's graph-based approach
    Uses efficient tensor operations instead of explicit graph generation
    """
    def __init__(self, embedding_dim, order=2):
        super(HighOrderFeatureInteraction, self).__init__()
        self.embedding_dim = embedding_dim
        self.order = order
        
        # Feature interaction layers for different orders
        self.interaction_layers = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(order)
        ])
        
        # Combination weights for different interaction orders
        self.combination_weights = nn.Parameter(torch.ones(order + 1))  # +1 for original features
        
    def forward(self, features):
        """
        Compute high-order feature interactions
        Args:
            features: [n_items, embedding_dim]
        Returns:
            enhanced_features: [n_items, embedding_dim] with high-order interactions
        """
        interaction_results = [features]  # 0th order (original)
        
        current_feat = features
        for i in range(self.order):
            # Compute i+1 order interaction
            current_feat = current_feat * self.interaction_layers[i](features)
            interaction_results.append(current_feat)
        
        # Weighted combination of different orders
        weights = F.softmax(self.combination_weights, dim=0)
        enhanced_features = sum(w * feat for w, feat in zip(weights, interaction_results))
        
        return enhanced_features


class MM_Model_Optimized_Enhanced(nn.Module):
    """
    Optimized enhanced MM_Model that carefully integrates EmerG ideas
    while preserving LLMRec's proven performance
    """
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, 
                 image_feats, text_feats, user_init_embedding, item_attribute_dict):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size

        # Original LLMRec components (EXACTLY as original)
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
        
        # Store features
        self.image_feats = torch.tensor(image_feats).float().cuda()
        self.text_feats = torch.tensor(text_feats).float().cuda()
        self.user_feats = torch.tensor(user_init_embedding).float().cuda()
        self.item_feats = {}
        for key in item_attribute_dict.keys():                                   
            self.item_feats[key] = torch.tensor(item_attribute_dict[key]).float().cuda() 

        # Enhanced components (carefully designed to improve, not replace)
        if getattr(args, 'use_enhanced_gnn', True):
            # Adaptive feature enhancer (lightweight)
            self.adaptive_enhancer = AdaptiveFeatureEnhancer(args.embed_size, num_features=2)
            
            # High-order feature interaction (inspired by EmerG's graph interactions)
            self.high_order_interaction = HighOrderFeatureInteraction(args.embed_size, order=2)
            
            # Enhancement control parameters
            self.enhancement_strength = nn.Parameter(torch.tensor(0.1))  # Learnable enhancement strength
            
        else:
            self.adaptive_enhancer = None
            self.high_order_interaction = None
            self.enhancement_strength = None
        
        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.Sigmoid()  
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.batch_norm = nn.BatchNorm1d(args.embed_size)
        self.tau = 0.5

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
            cur_matrix = cur_matrix.tocoo()
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))
        values = torch.from_numpy(cur_matrix.data)
        shape = torch.Size(cur_matrix.shape)
        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()

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
        # Feature mask (EXACTLY as original LLMRec)
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

        # Transform features (EXACTLY as original LLMRec)
        image_feats = self.dropout(self.image_trans(self.image_feats))
        text_feats = self.dropout(self.text_trans(self.text_feats))
        user_feats = self.dropout(self.user_trans(self.user_feats.to(torch.float32)))
        item_feats = {}
        for key in self.item_feats.keys():
            item_feats[key] = self.dropout(self.item_trans(self.item_feats[key]))

        # ORIGINAL LLMRec graph propagation (PRESERVED EXACTLY)
        for i in range(args.layers):
            image_user_feats = self.mm(ui_graph, image_feats)
            image_item_feats = self.mm(iu_graph, image_user_feats)

            text_user_feats = self.mm(ui_graph, text_feats)
            text_item_feats = self.mm(iu_graph, text_user_feats)

        # Apply EmerG-inspired enhancements AFTER original propagation
        if self.adaptive_enhancer is not None:
            try:
                # Enhance the propagated features (not the original ones)
                propagated_features = [image_item_feats, text_item_feats]
                enhanced_propagated = self.adaptive_enhancer(propagated_features)
                
                # Apply controlled enhancement
                strength = torch.sigmoid(self.enhancement_strength)  # Ensure 0-1 range
                image_item_feats = image_item_feats + strength * 0.1 * (enhanced_propagated[0] - image_item_feats)
                text_item_feats = text_item_feats + strength * 0.1 * (enhanced_propagated[1] - text_item_feats)
                
                # Also enhance user features
                user_propagated_features = [image_user_feats, text_user_feats]
                enhanced_user_propagated = self.adaptive_enhancer(user_propagated_features)
                
                image_user_feats = image_user_feats + strength * 0.1 * (enhanced_user_propagated[0] - image_user_feats)
                text_user_feats = text_user_feats + strength * 0.1 * (enhanced_user_propagated[1] - text_user_feats)
                
            except Exception as e:
                print(f"Enhancement failed, using original features: {e}")
        
        # Apply high-order interactions to item features
        if self.high_order_interaction is not None:
            try:
                image_item_feats = self.high_order_interaction(image_item_feats)
                text_item_feats = self.high_order_interaction(text_item_feats)
            except Exception as e:
                print(f"High-order interaction failed: {e}")

        # Aug item attribute (EXACTLY as original LLMRec)
        user_feat_from_item = {}
        for key in self.item_feats.keys():
            user_feat_from_item[key] = self.mm(ui_graph, item_feats[key])
            item_feats[key] = self.mm(iu_graph, user_feat_from_item[key])

        # Aug user profile (EXACTLY as original LLMRec)
        item_prof_feat = self.mm(iu_graph, user_feats)
        user_prof_feat = self.mm(ui_graph, item_prof_feat)

        # ID embeddings and propagation (EXACTLY as original LLMRec)
        u_g_embeddings = self.user_id_embedding.weight
        i_g_embeddings = self.item_id_embedding.weight             

        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]
        for i in range(self.n_ui_layers):    
            if i == (self.n_ui_layers-1):
                u_g_embeddings = self.softmax(torch.mm(ui_graph, i_g_embeddings)) 
                i_g_embeddings = self.softmax(torch.mm(iu_graph, u_g_embeddings))
            else:
                u_g_embeddings = torch.mm(ui_graph, i_g_embeddings) 
                i_g_embeddings = torch.mm(iu_graph, u_g_embeddings) 

            user_emb_list.append(u_g_embeddings)
            item_emb_list.append(i_g_embeddings)

        u_g_embeddings = torch.mean(torch.stack(user_emb_list), dim=0)
        i_g_embeddings = torch.mean(torch.stack(item_emb_list), dim=0) 

        # Enhanced feature fusion (SAME as original but with enhanced features)
        u_g_embeddings = u_g_embeddings + args.model_cat_rate*F.normalize(image_user_feats, p=2, dim=1) + args.model_cat_rate*F.normalize(text_user_feats, p=2, dim=1)
        i_g_embeddings = i_g_embeddings + args.model_cat_rate*F.normalize(image_item_feats, p=2, dim=1) + args.model_cat_rate*F.normalize(text_item_feats, p=2, dim=1)
        
        # Profile features (EXACTLY as original)
        u_g_embeddings += args.user_cat_rate*F.normalize(user_prof_feat, p=2, dim=1)
        i_g_embeddings += args.user_cat_rate*F.normalize(item_prof_feat, p=2, dim=1)

        # Attribute features (EXACTLY as original)
        for key in self.item_feats.keys():
            u_g_embeddings += args.item_cat_rate*F.normalize(user_feat_from_item[key], p=2, dim=1)
            i_g_embeddings += args.item_cat_rate*F.normalize(item_feats[key], p=2, dim=1) 

        return u_g_embeddings, i_g_embeddings, image_item_feats, text_item_feats, image_user_feats, text_user_feats, user_feats, item_feats, user_prof_feat, item_prof_feat, user_feat_from_item, item_feats, i_mask_nodes, u_mask_nodes


class Decoder(nn.Module):
    """
    Original Decoder from LLMRec (preserved exactly)
    """
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