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


class ItemFeatureAttention(nn.Module):
    """
    Simplified item-specific attention mechanism inspired by EmerG
    Focuses on learning item-specific feature importance without complex graph generation
    """
    def __init__(self, embedding_dim, num_modalities=2):
        super(ItemFeatureAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_modalities = num_modalities
        
        # Attention weights for each modality
        self.modal_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(embedding_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_modalities)
        ])
        
        # Feature interaction layers (simplified from EmerG's graph approach)
        self.feature_interaction = nn.Sequential(
            nn.Linear(embedding_dim * num_modalities, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Dropout(args.drop_rate)
        )

    def forward(self, modal_features):
        """
        Apply item-specific attention to modal features
        Args:
            modal_features: list of [n_items, embedding_dim] tensors
        Returns:
            enhanced_features: list of enhanced modal features
        """
        enhanced_features = []
        attention_weights = []
        
        for i, modal_feat in enumerate(modal_features):
            # Compute attention weights for this modality
            att_weight = self.modal_attention[i](modal_feat)
            attention_weights.append(att_weight)
            
            # Apply attention
            enhanced_feat = modal_feat * att_weight
            enhanced_features.append(enhanced_feat)
        
        # Cross-modal feature interaction (inspired by EmerG's graph interaction)
        if len(enhanced_features) >= 2:
            # Concatenate all modal features
            concat_features = torch.cat(enhanced_features, dim=-1)
            
            # Apply feature interaction transformation
            interaction_output = self.feature_interaction(concat_features)
            
            # Distribute interaction output back to individual modalities
            for i in range(len(enhanced_features)):
                enhanced_features[i] = enhanced_features[i] + 0.1 * interaction_output
        
        return enhanced_features, attention_weights


class MM_Model_Simple_Enhanced(nn.Module):
    """
    Simplified enhanced MM_Model that integrates key ideas from EmerG
    without complex graph generation to avoid dimension issues
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

        # Original LLMRec components
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

        # Enhanced components (simplified from EmerG)
        if getattr(args, 'use_enhanced_gnn', True):
            self.item_attention = ItemFeatureAttention(args.embed_size, num_modalities=2)
            
            # Enhanced feature fusion inspired by EmerG's feature interaction
            self.enhanced_fusion = nn.Sequential(
                nn.Linear(args.embed_size * 2, args.embed_size),
                nn.ReLU(),
                nn.Linear(args.embed_size, args.embed_size),
                nn.Dropout(args.drop_rate)
            )
        else:
            self.item_attention = None
            self.enhanced_fusion = None
        
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
        # Feature mask (from original LLMRec)
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

        # Transform features (original LLMRec)
        image_feats = self.dropout(self.image_trans(self.image_feats))
        text_feats = self.dropout(self.text_trans(self.text_feats))
        user_feats = self.dropout(self.user_trans(self.user_feats.to(torch.float32)))
        item_feats = {}
        for key in self.item_feats.keys():
            item_feats[key] = self.dropout(self.item_trans(self.item_feats[key]))

        # Apply enhanced item-specific attention (simplified EmerG approach)
        if self.item_attention is not None:
            try:
                modal_features = [image_feats, text_feats]
                enhanced_modal_features, attention_weights = self.item_attention(modal_features)
                image_feats, text_feats = enhanced_modal_features[0], enhanced_modal_features[1]
                
                # Apply enhanced fusion
                if self.enhanced_fusion is not None:
                    fused_features = torch.cat([image_feats, text_feats], dim=-1)
                    fusion_output = self.enhanced_fusion(fused_features)
                    
                    # Add fusion output to both modalities (residual-like connection)
                    image_feats = image_feats + 0.1 * fusion_output
                    text_feats = text_feats + 0.1 * fusion_output
                    
            except Exception as e:
                print(f"Enhanced attention processing failed, using original features: {e}")

        # Original LLMRec graph propagation (preserved)
        for i in range(args.layers):
            image_user_feats = self.mm(ui_graph, image_feats)
            image_item_feats = self.mm(iu_graph, image_user_feats)

            text_user_feats = self.mm(ui_graph, text_feats)
            text_item_feats = self.mm(iu_graph, text_user_feats)

        # Aug item attribute (original LLMRec)
        user_feat_from_item = {}
        for key in self.item_feats.keys():
            user_feat_from_item[key] = self.mm(ui_graph, item_feats[key])
            item_feats[key] = self.mm(iu_graph, user_feat_from_item[key])

        # Aug user profile (original LLMRec)
        item_prof_feat = self.mm(iu_graph, user_feats)
        user_prof_feat = self.mm(ui_graph, item_prof_feat)

        # ID embeddings and propagation (original LLMRec)
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

        # Enhanced feature fusion (combining original and enhanced features)
        u_g_embeddings = u_g_embeddings + args.model_cat_rate*F.normalize(image_user_feats, p=2, dim=1) + args.model_cat_rate*F.normalize(text_user_feats, p=2, dim=1)
        i_g_embeddings = i_g_embeddings + args.model_cat_rate*F.normalize(image_item_feats, p=2, dim=1) + args.model_cat_rate*F.normalize(text_item_feats, p=2, dim=1)
        
        # Profile features
        u_g_embeddings += args.user_cat_rate*F.normalize(user_prof_feat, p=2, dim=1)
        i_g_embeddings += args.user_cat_rate*F.normalize(item_prof_feat, p=2, dim=1)

        # Attribute features
        for key in self.item_feats.keys():
            u_g_embeddings += args.item_cat_rate*F.normalize(user_feat_from_item[key], p=2, dim=1)
            i_g_embeddings += args.item_cat_rate*F.normalize(item_feats[key], p=2, dim=1) 

        return u_g_embeddings, i_g_embeddings, image_item_feats, text_item_feats, image_user_feats, text_user_feats, user_feats, item_feats, user_prof_feat, item_prof_feat, user_feat_from_item, item_feats, i_mask_nodes, u_mask_nodes


class Decoder(nn.Module):
    """
    Original Decoder from LLMRec (preserved)
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