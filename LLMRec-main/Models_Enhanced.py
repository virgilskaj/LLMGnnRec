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


class ItemSpecificGraphGenerator(nn.Module):
    """
    Adapted from EmerG's GraphGenerator to generate item-specific feature interaction graphs
    This module generates dynamic graphs based on item features for enhanced recommendation
    """
    def __init__(self, num_item_features, num_total_features, embedding_dim, device):
        super(ItemSpecificGraphGenerator, self).__init__()
        self.num_total_features = num_total_features
        self.num_item_features = num_item_features
        self.device = device
        
        # Generate separate graph generators for different feature types
        self.generators = nn.ModuleList([nn.Sequential(
            nn.Linear(num_item_features * embedding_dim + num_total_features, num_total_features),
            nn.ReLU(),
            nn.Linear(num_total_features, num_total_features),
            nn.ReLU(),
            nn.Linear(num_total_features, num_total_features)
        ) for _ in range(num_total_features)])

    def forward(self, feature_emb):
        """
        Generate item-specific feature interaction graphs
        Args:
            feature_emb: concatenated item features [batch_size, num_item_features * embedding_dim]
        Returns:
            graph: item-specific feature interaction matrix [batch_size, num_total_features, num_total_features]
        """
        graph_fields = []
        for i in range(self.num_total_features):
            field_index = torch.tensor([i]).to(self.device)
            field_onehot = F.one_hot(field_index, num_classes=self.num_total_features).repeat(feature_emb.shape[0], 1)
            graph_field = self.generators[i](torch.cat([feature_emb, field_onehot], dim=1))
            graph_fields.append(graph_field)
        
        graph = torch.cat([graph_field.unsqueeze(1) for graph_field in graph_fields], dim=1)
        # Average across batch for shared graph structure
        task_size = graph.shape[0]
        graph = torch.mean(graph, dim=0).squeeze()
        graph = graph.unsqueeze(0).repeat(task_size, 1, 1)
        
        return graph


class EnhancedGNNLayer(nn.Module):
    """
    Enhanced GNN layer combining EmerG's graph-based feature interaction 
    with LLMRec's multi-modal propagation
    """
    def __init__(self, num_features, embedding_dim, gnn_layers=3, use_residual=True, device=None):
        super(EnhancedGNNLayer, self).__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.gnn_layers = gnn_layers
        self.use_residual = use_residual
        self.device = device
        
        # Feature interaction layers (from EmerG)
        self.feature_interaction_layers = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(gnn_layers)
        ])
        
        # Multi-modal fusion layers
        self.modal_fusion = nn.ModuleList([
            nn.Linear(embedding_dim * 3, embedding_dim) for _ in range(gnn_layers)  # ID + Image + Text
        ])
        
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=args.drop_rate)

    def forward(self, graphs, feature_emb, modal_features=None):
        """
        Enhanced forward pass with item-specific graphs and multi-modal features
        Args:
            graphs: item-specific feature interaction graphs
            feature_emb: base feature embeddings
            modal_features: dict containing 'image' and 'text' features
        """
        h = feature_emb
        final_h = feature_emb
        
        for i in range(self.gnn_layers):
            # Apply item-specific feature interaction (from EmerG)
            if graphs is not None:
                # Use the generated graphs for feature interaction
                graph = graphs[i] if isinstance(graphs, list) else graphs
                a = torch.bmm(graph, h)
            else:
                # Fallback to linear transformation
                a = self.feature_interaction_layers[i](h)
            
            if i != self.gnn_layers - 1:
                a = self.relu(a)
            
            # Multi-modal feature fusion (enhanced from LLMRec)
            if modal_features is not None:
                # Concatenate ID, image, and text features
                if len(modal_features) >= 2:  # image and text available
                    fused_features = torch.cat([
                        a, 
                        modal_features.get('image', torch.zeros_like(a)),
                        modal_features.get('text', torch.zeros_like(a))
                    ], dim=-1)
                    a = self.modal_fusion[i](fused_features)
                    a = self.dropout(a)
            
            # Residual connection
            if self.use_residual:
                h = a + h
                final_h = final_h + h
            else:
                h = a
                final_h = a
        
        return final_h


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism adapted from EmerG for LLMRec
    """
    def __init__(self, embedding_dim, num_heads=4, dropout_rate=0.0):
        super(MultiHeadSelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.W_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_o = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = self.head_dim ** 0.5

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embedding_dim
        )
        
        # Final linear projection
        output = self.W_o(attended_values)
        return output


class MM_Model_Enhanced(nn.Module):
    """
    Enhanced MM_Model that integrates EmerG's item-specific GNN with LLMRec's multi-modal approach
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

        # Enhanced GNN components (from EmerG)
        num_item_features = len(item_attribute_dict) + 2  # +2 for image and text
        num_total_features = num_item_features + 1  # +1 for user features
        
        if getattr(args, 'use_enhanced_gnn', True):
            self.graph_generator = ItemSpecificGraphGenerator(
                num_item_features, num_total_features, args.embed_size, device
            )
            
            self.enhanced_gnn = EnhancedGNNLayer(
                num_total_features, args.embed_size, 
                getattr(args, 'gnn_layers', 3), use_residual=True, device=device
            )
            
            # Multi-head attention for feature enhancement
            if getattr(args, 'use_attention', True):
                self.multi_head_attention = MultiHeadSelfAttention(
                    args.embed_size, num_heads=getattr(args, 'attention_heads', 4), 
                    dropout_rate=args.drop_rate
                )
            else:
                self.multi_head_attention = None
        else:
            self.graph_generator = None
            self.enhanced_gnn = None
            self.multi_head_attention = None
        
        # Feature fusion layers
        self.feature_fusion = nn.Linear(args.embed_size * num_total_features, args.embed_size)
        
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

    def generate_item_specific_features(self, batch_items):
        """
        Generate item-specific feature representations for graph generation
        """
        # Get item features
        item_image_feats = self.dropout(self.image_trans(self.image_feats[batch_items]))
        item_text_feats = self.dropout(self.text_trans(self.text_feats[batch_items]))
        
        # Get item attribute features
        item_attr_feats = []
        for key in self.item_feats.keys():
            attr_feat = self.dropout(self.item_trans(self.item_feats[key][batch_items]))
            item_attr_feats.append(attr_feat)
        
        # Concatenate all item features
        all_item_feats = [item_image_feats, item_text_feats] + item_attr_feats
        concatenated_feats = torch.cat(all_item_feats, dim=1)
        
        return concatenated_feats

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

        # Enhanced GNN propagation with item-specific graphs (if enabled)
        if self.graph_generator is not None and self.enhanced_gnn is not None:
            enhanced_features = []
            
            # Generate item-specific graphs for a sample of items (to manage computational cost)
            sample_items = torch.randint(0, self.n_items, (min(512, self.n_items),)).cuda()
            item_specific_feats = self.generate_item_specific_features(sample_items)
            item_graphs = self.graph_generator(item_specific_feats)
            
            # Apply enhanced GNN to multi-modal features
            modal_dict = {'image': image_feats, 'text': text_feats}
            
            # Process each modality through enhanced GNN
            for modality in ['image', 'text']:
                modal_feat = modal_dict[modality]
                
                # Prepare features for GNN (add batch dimension if needed)
                if len(modal_feat.shape) == 2:
                    modal_feat = modal_feat.unsqueeze(1)  # Add feature dimension
                
                # Apply enhanced GNN with item-specific graphs
                try:
                    enhanced_modal_feat = self.enhanced_gnn(
                        item_graphs[:modal_feat.shape[0]] if item_graphs.shape[0] >= modal_feat.shape[0] else None,
                        modal_feat,
                        modal_dict
                    )
                    
                    # Apply multi-head attention for further enhancement
                    if self.multi_head_attention is not None and len(enhanced_modal_feat.shape) == 3:
                        enhanced_modal_feat = self.multi_head_attention(enhanced_modal_feat)
                        enhanced_modal_feat = enhanced_modal_feat.squeeze(1)
                    
                    enhanced_features.append(enhanced_modal_feat)
                except Exception as e:
                    # Fallback to original features if enhanced GNN fails
                    print(f"Enhanced GNN failed for {modality}, using original features: {e}")
                    enhanced_features.append(modal_feat)
        else:
            # Use original features if enhanced GNN is disabled
            enhanced_features = [image_feats, text_feats]

        # Original LLMRec graph propagation (preserved)
        for i in range(args.layers):
            image_user_feats = self.mm(ui_graph, enhanced_features[0])  # Use enhanced image features
            image_item_feats = self.mm(iu_graph, image_user_feats)

            text_user_feats = self.mm(ui_graph, enhanced_features[1])   # Use enhanced text features
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