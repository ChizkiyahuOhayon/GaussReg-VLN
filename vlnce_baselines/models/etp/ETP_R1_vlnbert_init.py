import torch


def get_tokenizer(args):
    from transformers import AutoTokenizer
    if args.dataset == 'rxr' or args.tokenizer == 'xlm':
        cfg_name = 'bert_config/xlm-roberta-base'
    else:
        cfg_name = 'bert_config/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def get_vlnbert_models(config=None, dropout_rate=0.1):
    
    from transformers import PretrainedConfig
    from vlnce_baselines.models.etp.ETP_R1_vilmodel_cmt import GlocalTextPathNavCMT

    model_class = GlocalTextPathNavCMT

    model_name_or_path = config.pretrained_path
    new_ckpt_weights = {}
    keywords = ['graph_query_text', 'graph_attentioned_txt_embeds_transform', 'global_sap_head']
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path, map_location='cpu')
        for k, v in ckpt_weights.items():
            if k.startswith('module'):
                new_ckpt_weights[k[7:]] = v
            if any(key in k for key in keywords):
                new_ckpt_weights['bert.' + k] = v
            else:
                new_ckpt_weights[k] = v
    
    cfg_name = 'bert_config/xlm-roberta-base'
    vis_config = PretrainedConfig.from_pretrained(cfg_name)

    vis_config.type_vocab_size = 2

    vis_config.max_action_steps = 100
    vis_config.image_feat_size = 512
    vis_config.use_depth_embedding = config.use_depth_embedding
    vis_config.depth_feat_size = 128
    vis_config.angle_feat_size = 4
    vis_config.gauss_feat_size = getattr(config, 'gauss_feat_size', 0)
    vis_config.gauss_residual_scale = getattr(
        config, 'gauss_residual_scale', 1.0
    )
    vis_config.candidate_scorer_hidden_size = getattr(
        config, 'candidate_scorer_hidden_size', 0
    )
    vis_config.candidate_scorer_scale = getattr(
        config, 'candidate_scorer_scale', 1.0
    )
    vis_config.gaussian_bev_hidden_size = getattr(
        config, 'gaussian_bev_hidden_size', 0
    )
    vis_config.anchor_repair_hidden_size = getattr(
        config, 'anchor_repair_hidden_size', 0
    )
    vis_config.hindsight_stop_hidden_size = getattr(
        config, 'hindsight_stop_hidden_size', 0
    )
    vis_config.terminal_commit_hidden_size = getattr(
        config, 'terminal_commit_hidden_size', 0
    )
    vis_config.geo_token_hidden_size = getattr(
        config, 'geo_token_hidden_size', 0
    )
    vis_config.successor_hidden_size = getattr(config, 'successor_hidden_size', 0)

    vis_config.num_l_layers = 12
    vis_config.num_pano_layers = 2
    vis_config.num_x_layers = 4
    vis_config.graph_sprels = config.use_sprels
    vis_config.glocal_fuse = 'global'

    vis_config.fix_lang_embedding = config.fix_lang_embedding
    vis_config.fix_pano_embedding = config.fix_pano_embedding

    vis_config.update_lang_bert = not vis_config.fix_lang_embedding
    vis_config.output_attentions = True

    vis_config.pred_head_dropout_prob = dropout_rate
    vis_config.hidden_dropout_prob = dropout_rate
    vis_config.attention_probs_dropout_prob = dropout_rate

    vis_config.use_lang2visn_attn = True

    vis_config.max_txt_task_embeddings = 4
    vis_config.max_gmap_task_embeddings = 3

    visual_model = model_class.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=vis_config, 
        state_dict=new_ckpt_weights)
    has_gaussian_weight = any(
        key.endswith('global_encoder.gmap_gauss_embedding.weight')
        for key in new_ckpt_weights
    )
    if (visual_model.global_encoder.gmap_gauss_embedding is not None and
            not has_gaussian_weight):
        # from_pretrained reinitializes missing keys after model.__init__.
        torch.nn.init.zeros_(
            visual_model.global_encoder.gmap_gauss_embedding.weight
        )
    has_candidate_scorer = any(
        'candidate_scorer.output.' in key for key in new_ckpt_weights
    )
    if visual_model.candidate_scorer is not None and not has_candidate_scorer:
        # from_pretrained reinitializes missing keys after model.__init__.
        visual_model.candidate_scorer.reset_output()
    has_gaussian_bev = any(
        'gaussian_bev.' in key for key in new_ckpt_weights
    )
    if visual_model.gaussian_bev is not None and not has_gaussian_bev:
        # from_pretrained reinitializes missing keys after model.__init__.
        visual_model.gaussian_bev.reset_output()
    has_anchor_repair = any(
        'anchor_repair.' in key for key in new_ckpt_weights
    )
    if visual_model.anchor_repair is not None and not has_anchor_repair:
        # from_pretrained reinitializes missing keys after model.__init__.
        visual_model.anchor_repair.reset_output()
    has_hindsight_stop = any(
        'hindsight_stop.' in key for key in new_ckpt_weights
    )
    if visual_model.hindsight_stop is not None and not has_hindsight_stop:
        # from_pretrained reinitializes missing keys after model.__init__.
        visual_model.hindsight_stop.reset_output()
    has_terminal_commit = any(
        'terminal_commit.' in key for key in new_ckpt_weights
    )
    if visual_model.terminal_commit is not None and not has_terminal_commit:
        # from_pretrained reinitializes missing keys after model.__init__.
        visual_model.terminal_commit.reset_output()
    has_geo_token = any(
        'geo_token.' in key for key in new_ckpt_weights
    )
    if visual_model.geo_token is not None and not has_geo_token:
        # from_pretrained reinitializes missing keys after model.__init__.
        visual_model.geo_token.reset_output()
    return visual_model
