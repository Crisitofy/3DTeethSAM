import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.sam2.utils.transforms import SAM2Transforms
from model.sam2.build_sam import build_sam2 
from model.sam2.modeling.sam2_utils import LayerNorm2d

from model.Mask_Refiner import ResUNet

class MultiLevelFeatureFusion(nn.Module):
    """
    Multi-level feature fusion
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.s0_processor = nn.Sequential(
            nn.Conv2d(d_model // 8, d_model // 4, kernel_size=2, stride=2),
            LayerNorm2d(d_model // 4),
            nn.GELU(),
            nn.Conv2d(d_model // 4, d_model, kernel_size=2, stride=2),
            LayerNorm2d(d_model),
            nn.GELU()
        )
        self.s1_processor = nn.Sequential(
            nn.Conv2d(d_model // 4, d_model, kernel_size=2, stride=2),
            LayerNorm2d(d_model),
            nn.GELU()
        )
        self.low_res_processor = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1),
            LayerNorm2d(d_model),
            nn.GELU()
        )
        self.concat_fusion_conv = nn.Conv2d(d_model * 3, d_model, kernel_size=1)

        # channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d_model, d_model//16, 1),
            nn.GELU(),
            nn.Conv2d(d_model//16, d_model, 1),
            nn.Sigmoid()
        )
        
        # spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # feature processing and residual connection
        self.refine = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            LayerNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            LayerNorm2d(d_model),
            nn.GELU()
        )
        
        # final projection to embedding space
        self.final_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, low_res_feat, high_res_feats):
        """
        Args:
            low_res_feat: [B, 256, 64, 64] - low-resolution features
            high_res_feats: List[[B, 32, 256, 256], [B, 64, 128, 128]] - high-resolution features

        Returns:
            fused_feat: [B, 4096, 256] - fused features, ready for transformer input, [B, HW, C]
            fused_features: [B, C, H, W] - fused features, ready for refinement
        """
        feat_s0, feat_s1 = high_res_feats  # feat_s0 is highest resolution
        
        s0_processed = self.s0_processor(feat_s0)
        s1_processed = self.s1_processor(feat_s1)
        low_res_processed = self.low_res_processor(low_res_feat)
        concatenated = torch.cat([s0_processed, s1_processed, low_res_processed], dim=1)
        fused = self.concat_fusion_conv(concatenated)

        # CBAM
        channel_attn = self.channel_attention(fused)
        fused = fused * channel_attn
        avg_feat = torch.mean(fused, dim=1, keepdim=True)
        max_feat, _ = torch.max(fused, dim=1, keepdim=True)
        spatial_feat = torch.cat([avg_feat, max_feat], dim=1)
        spatial_attn = self.spatial_attention(spatial_feat)
        fused = fused * spatial_attn

        # refine fused features
        fused_features = self.refine(fused)  # [B, C, H, W]

        # reshape to sequence format, suitable for transformer processing
        seq_feat = fused_features.flatten(2).transpose(1, 2)  # [B, 4096, 256]

        # final projection and normalization
        out = self.final_proj(seq_feat)
        out = self.dropout(out)
        out = self.norm(out + seq_feat)  # residual connection

        return out, fused_features

class Mask_Classifier(nn.Module):
    """classification head based on fused image features and Transformer decoder"""
    def __init__(self, img_feature_dim, num_classification_queries, num_classes_to_predict, 
                 num_decoder_layers, nhead, dim_feedforward, dropout):
        super().__init__()
        self.num_classification_queries = num_classification_queries
        self.img_feature_dim = img_feature_dim

        self.classification_queries = nn.Parameter(torch.empty(num_classification_queries, img_feature_dim))
        nn.init.kaiming_uniform_(self.classification_queries, a=math.sqrt(5))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=img_feature_dim, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.output_mlp = nn.Sequential(
            nn.Linear(img_feature_dim, img_feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(img_feature_dim // 2, num_classes_to_predict)
        )

    def forward(self, fused_features):
        """
        Args:
            fused_features: [B, HW, D_img] - fused features
        Returns:
            class_logits: [B, num_classification_queries, num_classes_to_predict]
        """
        B = fused_features.shape[0]
        
        # prepare target sequence for Transformer decoder (learnable queries), batch_first=True: tgt.shape [B, N, E] 
        tgt_queries = self.classification_queries.unsqueeze(0).expand(B, -1, -1)
        
        # Transformer decoder, decoder_output: [B, num_classification_queries, img_feature_dim]
        decoder_output = self.transformer_decoder(tgt_queries, fused_features)
        
        # final classification via MLP, class_logits: [B, num_classification_queries, num_classes_to_predict] 
        class_logits = self.output_mlp(decoder_output)
        
        return class_logits

class PEG(nn.Module):
    """PEG, based on Transformer decoder"""
    def __init__(self, d_model=256, num_queries=16, num_layers=5, 
                 dropout=0.15, nhead=8):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        
        self.query_embed = nn.Parameter(torch.empty(num_queries, d_model))
        nn.init.kaiming_uniform_(self.query_embed, a=math.sqrt(5))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True,
            dropout=dropout
        )

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.label_head = nn.Linear(d_model, 1)

    def forward(self, image_embed):
        """
        Args:
            image_embed: [B, H*W, C] - fused image embeddings
        Returns:
            prompt_embed: [B, num_queries, d_model] - generated prompt embeddings
            labels: [B, num_queries] - confidence for each prompt embedding, range [0, 1]
        """
        B, _, _ = image_embed.shape
        device = image_embed.device
        
        query_embed = self.query_embed.to(device)
        query_embed = query_embed.unsqueeze(0).expand(B, -1, -1)

        prompt_embed = self.transformer_decoder(query_embed, image_embed)

        labels = self.label_head(prompt_embed).squeeze(-1)  # [B, num_queries]
        
        return prompt_embed, labels

class DentalSegmentationSystem(nn.Module):
    """
    End-to-end dental segmentation system, using SAM2 and Transformer-based prompt generation network
    """
    def __init__(self, config):
        super().__init__()
        
        # model parameters
        self.d_model = config.get('embed_dim', 256)
        self.num_queries = config.get('num_teeth', 16) 

        self.num_classes_for_head = 16
        self.num_layers = config.get('num_layers', 6) 
        self.dropout_rate = config.get('dropout_rate', 0.1)

        self.feature_fusion = MultiLevelFeatureFusion(
            d_model=self.d_model,
            dropout=self.dropout_rate
        )

        self.auto_prompt_generator = PEG(
            d_model=self.d_model,
            num_queries=self.num_queries,
            num_layers=self.num_layers,
            dropout=self.dropout_rate, 
            nhead=config.get('nhead', 8),
        )

        self.class_head = Mask_Classifier(
            img_feature_dim=self.d_model,
            num_classification_queries=self.num_queries,
            num_classes_to_predict=self.num_classes_for_head,
            num_decoder_layers=config.get('classification_head_num_decoder_layers', 2),
            nhead=config.get('classification_head_nhead', 4),
            dim_feedforward=config.get('classification_head_decoder_dim_feedforward', self.d_model * 4),
            dropout=config.get('classification_head_dropout', self.dropout_rate)
        )

        self.sam_model = build_sam2(
            config_file=config.get('sam_config', 'configs/sam2.1/sam2.1_hiera_l.yaml'),
            ckpt_path=config.get('sam_checkpoint', 'checkpoints/sam2.1_hiera_large.pt')
        )
        
        self.bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        
        self._transforms = SAM2Transforms(
            resolution=self.sam_model.image_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )
        
        self.refine_net = ResUNet(
            num_classes=config.get('num_classes', 17),
            dropout_rate=self.dropout_rate
        )

        if not config.get('finetune_sam', False):
            for param in self.sam_model.parameters():
                param.requires_grad = False
    
    def forward(self, images):
        """
        Args:
            images: input images [B, 3, H, W]
        Returns:
            sam_masks: SAM generated masks [B, num_classes, H, W]
            refined_masks: refined masks [B, num_classes, H, W]
            confidece: existence probabilities [B, num_queries]
            class_logits: class logits [B, num_queries, 16]
        """
        image_embed, high_res_feats, orig_hw = self.process_images(images)

        seq_features, fused_features = self.feature_fusion(image_embed, high_res_feats) # [B, HW, C], [B, C, H, W]

        prompt_embed, confidence = self.auto_prompt_generator(seq_features)  

        class_logits = self.class_head(seq_features)  # [B, num_queries, 16]

        sam_masks = self.generate_masks(prompt_embed, image_embed, high_res_feats, orig_hw)

        sam_probs = torch.sigmoid(sam_masks)  # [B, num_classes, H, W]

        refined_masks = self.refine_net(images, sam_probs, fused_features)
        
        return sam_masks, refined_masks, confidence, class_logits
        
    def process_images(self, images):
        """
        Args:
            images: image tensor [B, 3, H, W]
            
        Returns:
            image_embed: image embedding
            high_res_feats: high-resolution features
            orig_hw: original image size
        """
        B, _, H, W = images.shape
        orig_hw = [(H, W) for _ in range(B)]
        
        img_batch= self._transforms.transforms(images)
        
        backbone_out = self.sam_model.forward_image(img_batch)
        _, vision_feats, _, _ = self.sam_model._prepare_backbone_features(backbone_out)
        
        if self.sam_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam_model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(B, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self.bb_feat_sizes[::-1])
        ][::-1]

        image_embed = feats[-1]
        high_res_feats = feats[:-1]
        return image_embed, high_res_feats, orig_hw

    def generate_masks(self, prompt_embed, image_embed, high_res_feats, orig_hw):
        """
        Args:
            prompt_embed: prompt embeddings [B, num_queries, d_model]
            image_embed: image embedding [B, H'*W', C]
            high_res_feats: high-resolution features list
            orig_hw: original image size
            
        Returns:
            sam_masks: SAM generated masks [B, num_classes, H, W]
        """
        B = prompt_embed.shape[0]

        image_embeddings = torch.repeat_interleave(image_embed, self.num_queries, dim=0) # (B*n_teeth, C, H', W')
        high_res_features = [torch.repeat_interleave(feats, self.num_queries, dim=0) for feats in high_res_feats]
        
        # flatten prompt_embed to match repeated image_embeddings
        sparse_embeddings = prompt_embed.reshape(-1, prompt_embed.shape[-1]).unsqueeze(1)  # [B*n_teeth, 1, embed_dim]
        
        # get dense_embeddings
        dense_embeddings = self.sam_model.sam_prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            B * self.num_queries, 
            -1, 
            self.sam_model.sam_prompt_encoder.image_embedding_size[0], 
            self.sam_model.sam_prompt_encoder.image_embedding_size[1]
        )

        low_res_masks, _, _, _ = self.sam_model.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,  
            repeat_image=False,  
            high_res_features=high_res_features,
        )

        masks = F.interpolate(low_res_masks, orig_hw[0], mode="bilinear", align_corners=False)
        masks = masks.view(B, self.num_queries, masks.shape[2], masks.shape[3])

        fg_probs = torch.sigmoid(masks)
        bg_prob = 1.0 - torch.clamp(fg_probs.sum(dim=1, keepdim=True), 0, 1)
        eps = 1e-6
        bg_masks = torch.log(bg_prob / (1 - bg_prob + eps) + eps)
        sam_masks = torch.cat([bg_masks, masks], dim=1)  # [B, 17, H, W]

        return sam_masks
        
  