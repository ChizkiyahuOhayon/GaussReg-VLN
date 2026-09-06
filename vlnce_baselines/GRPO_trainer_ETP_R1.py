import gc
import os
import sys
import random
import warnings
from collections import defaultdict
from typing import Dict, List
import jsonlines

import lmdb
import msgpack_numpy
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm
from gym import Space
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_for_rl, is_slurm_batch_job
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.models.graph_utils import GraphMap, MAX_DIST
from vlnce_baselines.utils import reduce_loss

from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele,
)
from vlnce_baselines.common.utils import dis_to_con, gather_list_and_concat
from vlnce_baselines.anchor_relative import anchor_relative_advantages
from vlnce_baselines.hindsight_stop import (
    counterfactual_stop_returns,
    hindsight_stop_targets,
    success_set_nll,
)
from vlnce_baselines.setwise_policy import (
    recoverable_outcome,
    setwise_group_advantages,
)
from vlnce_baselines.frontier_advantage import (
    action_set_advantages,
    annealed_weight,
    mix_advantages,
)
from vlnce_baselines.geo_token import align_candidate_tokens
from habitat_extensions.measures import NDTW, StepsTaken
from fastdtw import fastdtw

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401

import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from vlnce_baselines.common.ops import pad_tensors_wgrad, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence
import cv2
import copy
from collections import OrderedDict
import hashlib
import pickle

@baseline_registry.register_trainer(name="GRPO-R1")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.GRPO.max_traj_len) #  * 0.97 transfered gt path got 0.96 spl
        self.illegal_episodes_count = 0

        self.grpo_epsilon = config.GRPO.grpo_epsilon  
        self.grpo_beta = config.GRPO.grpo_beta        
        if self.grpo_beta < 1e-6:
            self.need_ref_policy = False
        else:
            self.need_ref_policy = True
        self.max_grad_norm = config.GRPO.max_grad_norm
        self.initial_num_envs = config.NUM_ENVIRONMENTS 
        self.grpo_update_epochs = config.GRPO.update_epochs
        self.enable_amp = config.GRPO.enable_amp
        self.enable_all_dropouts = config.GRPO.enable_all_dropouts
        self.dropout_in_sampling = config.GRPO.dropout_in_sampling
        self.dropout_rate = config.GRPO.dropout_rate
        self.frontier_advantage = getattr(
            config.GRPO, 'frontier_advantage', False
        )
        self.scaler = GradScaler(enabled=self.enable_amp)
        print("config.GRPO:\n", config.GRPO)
        print(f"GRPO params: grpo_epsilon {self.grpo_epsilon}, grpo_beta {self.grpo_beta}, max_grad_norm {self.max_grad_norm}, grpo_update_epochs {self.grpo_update_epochs} \
              enable_amp {self.enable_amp}, need_ref_policy {self.need_ref_policy}, enable_all_dropouts {self.enable_all_dropouts}, dropout_rate {self.dropout_rate}, dropout_in_sampling {self.dropout_in_sampling}")
    def _make_dirs(self):
        if self.config.local_rank == 0:
            self._make_ckpt_dir()
            if self.config.EVAL.SAVE_RESULTS:
                self._make_results_dir()

    def save_checkpoint(self, iteration: int):
        if self.config.ONLY_LAST_SAVEALL and (not iteration == self.config.GRPO.iters):
            torch.save(
                        obj={
                            "state_dict": self.policy.state_dict(), 
                            "config": self.config, 
                            "iteration": iteration
                        },
                        f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
                    )
        else:
            torch.save(
                obj={
                    "state_dict": self.policy.state_dict(), 
                    "config": self.config, 
                    "optim_state": self.optimizer.state_dict(), 
                    "scheduler_state": self.scheduler.state_dict(), 
                    "iteration": iteration, 
                },
                f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
            )

    def _set_config(self):
        self.split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.split
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS[self.config.local_rank]
        self.config.use_pbar = not is_slurm_batch_job()
        ''' if choosing image '''
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        print(f"init camera information: resize_config:{resize_config}, crop_config:{crop_config}, new_camera_heading:{camera_orientations}")

        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        if self.config.MODEL.task_type == 'rxr':
            # self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION_TRAIN', 'STEPS_TAKEN', 'COLLISIONS', 'NDTW', 'SDTW']
            self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION_TRAIN', 'STEPS_TAKEN', 'COLLISIONS']
        elif self.config.MODEL.task_type == 'r2r':
            self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION_TRAIN', 'STEPS_TAKEN', 'COLLISIONS']
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.GRPO.batch_size
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            torch.cuda.set_device(self.device)

    def _init_envs(self):
        # for DDP to load different data
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank * 10 
        self.config.freeze()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        env_num = self.envs.num_envs
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(f'LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}')
        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        return observation_space, action_space

    def setup_training_parts(self):
        self.policy.eval()
        for param in self.policy.parameters():
            param.requires_grad = False

        if not self.trainable_parts:
            print("Warning: No specific parts designated as trainable.")
            return

        print("Setting up specific parts for training...")
        for module_to_train in self.trainable_parts:
            module_to_train.train()
            print(f"  Module {type(module_to_train).__name__} set to TRAIN mode.")
            for param in module_to_train.parameters():
                param.requires_grad = True
            print(f"    Parameters for {type(module_to_train).__name__} UNFROZEN.")

    def set_policy_mode(self, mode):
        if not self.enable_all_dropouts:
            if not self.trainable_parts:
                return
            if mode == 'train':
                for module_part in self.trainable_parts:
                    module_part.train()
            elif mode == 'eval':
                for module_part in self.trainable_parts:
                    module_part.eval()
            else:
                raise ValueError("Mode must be 'train' or 'eval'.")
        else:
            if mode == 'train':
                if self.world_size > 1:
                    self.policy.net.module.vln_bert.train()
                else:
                    self.policy.net.vln_bert.train()
            elif mode == 'eval':
                self.policy.eval()
            else:
                raise ValueError("Mode must be 'train' or 'eval'.")
            
    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ):
        start_iter = 0
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
            dropout_rate=self.dropout_rate,
        )
        ''' initialize the waypoint predictor here '''
        from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        cwp_fn = 'data/wp_pred/check_cwp_bestdist_hfov63' if self.config.MODEL.task_type == 'rxr' else 'data/wp_pred/check_cwp_bestdist_hfov90'
        self.waypoint_predictor.load_state_dict(torch.load(cwp_fn, map_location = torch.device('cpu'))['predictor']['state_dict']) 
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)

        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers 

        try:
            vln_bert_module = self.policy.net.vln_bert
            gauss_only = getattr(self.config.GRPO, 'gauss_only', False)
            scorer_only = getattr(
                self.config.GRPO, 'candidate_scorer_only', False
            )
            gaussian_bev_only = getattr(
                self.config.GRPO, 'gaussian_bev_only', False
            )
            anchor_repair_only = getattr(
                self.config.GRPO, 'anchor_repair_only', False
            )
            hindsight_stop_only = getattr(
                self.config.GRPO, 'hindsight_stop_only', False
            )
            terminal_commit_only = getattr(
                self.config.GRPO, 'terminal_commit_only', False
            )
            success_set_commit = getattr(
                self.config.GRPO, 'success_set_commit', False
            )
            setwise_group_policy = getattr(
                self.config.GRPO, 'setwise_group_policy', False
            )
            frontier_advantage = getattr(
                self.config.GRPO, 'frontier_advantage', False
            )
            geo_token_only = getattr(
                self.config.GRPO, 'geo_token_only', False
            )
            if success_set_commit and not terminal_commit_only:
                raise ValueError(
                    'success_set_commit requires terminal_commit_only'
                )
            if setwise_group_policy:
                if any([
                        gauss_only, scorer_only, gaussian_bev_only,
                        anchor_repair_only, hindsight_stop_only,
                        terminal_commit_only, success_set_commit,
                        frontier_advantage, geo_token_only]):
                    raise ValueError(
                        'E9 cannot use E2-E8 training modes'
                    )
                if (self.config.GRPO.sample_num != 8 or
                        self.config.MODEL.task_type != 'r2r' or
                        self.config.MODEL.gauss_feat_size != 0 or
                        self.config.MODEL.candidate_scorer_hidden_size != 0 or
                        self.config.MODEL.gaussian_bev_hidden_size != 0 or
                        self.config.MODEL.anchor_repair_hidden_size != 0 or
                        self.config.MODEL.hindsight_stop_hidden_size != 0 or
                        self.config.MODEL.terminal_commit_hidden_size != 0 or
                        self.config.MODEL.geo_token_hidden_size != 0):
                    raise ValueError(
                        'E9 requires R2R, sample_num=8, and no E2-E8 module'
                    )
            if frontier_advantage:
                if any([
                        gauss_only, scorer_only, gaussian_bev_only,
                        anchor_repair_only, hindsight_stop_only,
                        terminal_commit_only, success_set_commit,
                        setwise_group_policy, geo_token_only]):
                    raise ValueError('E10 cannot use E2-E9 training modes')
                if (self.config.GRPO.sample_num != 8 or
                        self.config.MODEL.task_type != 'r2r' or
                        self.config.MODEL.gauss_feat_size != 0 or
                        self.config.MODEL.candidate_scorer_hidden_size != 0 or
                        self.config.MODEL.gaussian_bev_hidden_size != 0 or
                        self.config.MODEL.anchor_repair_hidden_size != 0 or
                        self.config.MODEL.hindsight_stop_hidden_size != 0 or
                        self.config.MODEL.terminal_commit_hidden_size != 0 or
                        self.config.MODEL.geo_token_hidden_size != 0):
                    raise ValueError(
                        'E10 requires R2R, sample_num=8, and no E2-E8 module'
                    )
            if sum([
                    gauss_only, scorer_only, gaussian_bev_only,
                    anchor_repair_only, hindsight_stop_only,
                    terminal_commit_only, geo_token_only]) > 1:
                raise ValueError(
                    'GRPO lightweight-only modes are mutually exclusive'
                )
            if geo_token_only:
                geo_token = vln_bert_module.geo_token
                if geo_token is None:
                    raise ValueError(
                        'geo_token_only requires '
                        'MODEL.geo_token_hidden_size > 0'
                    )
                if (self.config.GRPO.sample_num != 8 or
                        self.config.MODEL.task_type != 'r2r' or
                        self.config.MODEL.gauss_feat_size != 0 or
                        self.config.MODEL.candidate_scorer_hidden_size != 0 or
                        self.config.MODEL.gaussian_bev_hidden_size != 0 or
                        self.config.MODEL.anchor_repair_hidden_size != 0 or
                        self.config.MODEL.hindsight_stop_hidden_size != 0 or
                        self.config.MODEL.terminal_commit_hidden_size != 0 or
                        setwise_group_policy or frontier_advantage):
                    raise ValueError(
                        'E11 requires strict E0, R2R, sample_num=8, and '
                        'no E2-E10 module'
                    )
                self.trainable_parts = [geo_token]
            elif terminal_commit_only:
                terminal_commit = vln_bert_module.terminal_commit
                if terminal_commit is None:
                    raise ValueError(
                        'terminal_commit_only requires '
                        'MODEL.terminal_commit_hidden_size > 0'
                    )
                if (self.config.GRPO.sample_num != 1 or
                        self.config.MODEL.task_type != 'r2r' or
                        self.config.MODEL.gauss_feat_size != 0 or
                        self.config.MODEL.candidate_scorer_hidden_size != 0 or
                        self.config.MODEL.gaussian_bev_hidden_size != 0 or
                        self.config.MODEL.anchor_repair_hidden_size != 0 or
                        self.config.MODEL.hindsight_stop_hidden_size != 0):
                    experiment = 'E8-1' if success_set_commit else 'E7-1'
                    raise ValueError(
                        '%s requires R2R, sample_num=1, '
                        'gauss_feat_size=0, and no E3-E6 module' % experiment
                    )
                self.trainable_parts = [terminal_commit]
            elif hindsight_stop_only:
                hindsight_stop = vln_bert_module.hindsight_stop
                if hindsight_stop is None:
                    raise ValueError(
                        'hindsight_stop_only requires '
                        'MODEL.hindsight_stop_hidden_size > 0'
                    )
                if (self.config.GRPO.sample_num != 1 or
                        self.config.MODEL.task_type != 'r2r' or
                        self.config.MODEL.gauss_feat_size != 0 or
                        self.config.MODEL.candidate_scorer_hidden_size != 0 or
                        self.config.MODEL.gaussian_bev_hidden_size != 0 or
                        self.config.MODEL.anchor_repair_hidden_size != 0):
                    raise ValueError(
                        'E6-1 requires R2R, sample_num=1, '
                        'gauss_feat_size=0, and no E3/E4/E5 module'
                    )
                self.trainable_parts = [hindsight_stop]
            elif anchor_repair_only:
                anchor_repair = vln_bert_module.anchor_repair
                if anchor_repair is None:
                    raise ValueError(
                        'anchor_repair_only requires '
                        'MODEL.anchor_repair_hidden_size > 0'
                    )
                if (self.config.GRPO.sample_num != 8 or
                        self.config.MODEL.gauss_feat_size != 5 or
                        self.config.MODEL.gauss_residual_scale != 0.0 or
                        self.config.MODEL.candidate_scorer_hidden_size != 0 or
                        self.config.MODEL.gaussian_bev_hidden_size != 0):
                    raise ValueError(
                        'E5 requires sample_num=8, gauss_feat_size=5, '
                        'gauss_residual_scale=0.0, and no E3/E4 module'
                    )
                self.trainable_parts = [anchor_repair]
            elif gaussian_bev_only:
                gaussian_bev = vln_bert_module.gaussian_bev
                if gaussian_bev is None:
                    raise ValueError(
                        'gaussian_bev_only requires '
                        'MODEL.gaussian_bev_hidden_size > 0'
                    )
                if (self.config.MODEL.gauss_feat_size != 5 or
                        self.config.MODEL.gauss_residual_scale != 0.0 or
                        self.config.MODEL.candidate_scorer_hidden_size != 0):
                    raise ValueError(
                        'E4 requires gauss_feat_size=5, '
                        'gauss_residual_scale=0.0, and no candidate scorer'
                    )
                self.trainable_parts = [gaussian_bev]
            elif scorer_only:
                scorer = vln_bert_module.candidate_scorer
                if scorer is None:
                    raise ValueError(
                        'candidate_scorer_only requires '
                        'MODEL.candidate_scorer_hidden_size > 0'
                    )
                if (self.config.MODEL.gauss_feat_size != 5 or
                        self.config.MODEL.gauss_residual_scale != 0.0):
                    raise ValueError(
                        'E3 requires gauss_feat_size=5 and '
                        'gauss_residual_scale=0.0'
                    )
                self.trainable_parts = [scorer]
            elif gauss_only:
                gauss_embedding = vln_bert_module.global_encoder.gmap_gauss_embedding
                if gauss_embedding is None:
                    raise ValueError(
                        'GRPO.gauss_only requires MODEL.gauss_feat_size=5'
                    )
                self.trainable_parts = [gauss_embedding]
            else:
                self.trainable_parts = [
                    vln_bert_module.global_encoder,
                    vln_bert_module.graph_query_text,
                    vln_bert_module.graph_attentioned_txt_embeds_transform,
                    vln_bert_module.global_sap_head
                ]
            for part in self.trainable_parts:
                if not isinstance(part, torch.nn.Module):
                    raise TypeError(f"Part {part} is not an nn.Module")
        except AttributeError as e:
            print(f"Error accessing specified submodules: {e}")
            print("Please ensure the paths to trainable submodules are correct.")
            self.trainable_parts = []
        self.setup_training_parts()

        if self.config.GPU_NUMBERS > 1:
            print('Using', self.config.GPU_NUMBERS,'GPU!')
            # find_unused_parameters=False fix ddp bug
            self.policy.net = DDP(self.policy.net.to(self.device), device_ids=[self.device],
                output_device=self.device, find_unused_parameters=False, broadcast_buffers=False)

        not_trainable_parameters = [p for p in self.policy.parameters() if not p.requires_grad]
        trainable_parameters = [(n, p) for n, p in self.policy.named_parameters() if p.requires_grad]
        if getattr(self.config.GRPO, 'gauss_only', False):
            trainable_count = sum(p.numel() for _, p in trainable_parameters)
            if len(trainable_parameters) != 1 or trainable_count != 3840:
                raise RuntimeError(
                    'Gaussian-only GRPO expected one 3,840-parameter tensor, got '
                    '%d tensors and %d parameters' %
                    (len(trainable_parameters), trainable_count)
                )
        if getattr(self.config.GRPO, 'candidate_scorer_only', False):
            trainable_count = sum(p.numel() for _, p in trainable_parameters)
            invalid_names = [
                name for name, _ in trainable_parameters
                if '.candidate_scorer.' not in name
            ]
            if invalid_names or trainable_count >= 500000:
                raise RuntimeError(
                    'Candidate-scorer GRPO expected only a sub-0.5M scorer, '
                    'got invalid=%s and %d parameters' %
                    (invalid_names, trainable_count)
                )
        if getattr(self.config.GRPO, 'gaussian_bev_only', False):
            trainable_count = sum(p.numel() for _, p in trainable_parameters)
            invalid_names = [
                name for name, _ in trainable_parameters
                if '.gaussian_bev.' not in name
            ]
            if invalid_names or trainable_count >= 100000:
                raise RuntimeError(
                    'Gaussian-BEV GRPO expected only a sub-0.1M field, '
                    'got invalid=%s and %d parameters' %
                    (invalid_names, trainable_count)
                )
        if getattr(self.config.GRPO, 'anchor_repair_only', False):
            trainable_count = sum(p.numel() for _, p in trainable_parameters)
            invalid_names = [
                name for name, _ in trainable_parameters
                if '.anchor_repair.' not in name
            ]
            if invalid_names or trainable_count >= 200000:
                raise RuntimeError(
                    'Anchor-repair GRPO expected only a sub-0.2M repair '
                    'module, got invalid=%s and %d parameters' %
                    (invalid_names, trainable_count)
                )
        if getattr(self.config.GRPO, 'hindsight_stop_only', False):
            trainable_count = sum(p.numel() for _, p in trainable_parameters)
            invalid_names = [
                name for name, _ in trainable_parameters
                if '.hindsight_stop.' not in name
            ]
            if invalid_names or trainable_count >= 250000:
                raise RuntimeError(
                    'Hindsight-stop training expected only a sub-0.25M head, '
                    'got invalid=%s and %d parameters' %
                    (invalid_names, trainable_count)
                )
        if getattr(self.config.GRPO, 'terminal_commit_only', False):
            trainable_count = sum(p.numel() for _, p in trainable_parameters)
            invalid_names = [
                name for name, _ in trainable_parameters
                if '.terminal_commit.' not in name
            ]
            if invalid_names or trainable_count >= 250000:
                raise RuntimeError(
                    'Terminal-commit training expected only a sub-0.25M head, '
                    'got invalid=%s and %d parameters' %
                    (invalid_names, trainable_count)
                )
        if getattr(self.config.GRPO, 'geo_token_only', False):
            trainable_count = sum(p.numel() for _, p in trainable_parameters)
            invalid_names = [
                name for name, _ in trainable_parameters
                if '.geo_token.' not in name
            ]
            if invalid_names or trainable_count != 1857:
                raise RuntimeError(
                    'GeoToken training expected exactly 1,857 GeoToken '
                    'parameters, got invalid=%s and %d parameters' %
                    (invalid_names, trainable_count)
                )
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in trainable_parameters
                        if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01},
            {'params': [p for n, p in trainable_parameters
                        if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
        ]
        if trainable_parameters:
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.GRPO.lr)
            print(f"Optimizer configured with {len(trainable_parameters)} trainable parameters.")
            print(f"Remaining {len(not_trainable_parameters)} untrainable parameters.")
        else:
            self.optimizer = None
            print("Warning: No parameters were set to trainable. Optimizer not configured.")
        
        num_warmup_steps = self.config.GRPO.warmup_iters
        num_training_steps = self.config.GRPO.iters
        min_lr_ratio = self.config.GRPO.min_lr_ratio
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            progress = min(1.0, progress)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            decayed_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
            return decayed_lr_multiplier
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        if load_from_ckpt: 
            if config.GRPO.is_requeue: 
                import glob
                search_pattern = os.path.join(config.CHECKPOINT_FOLDER, "*.pth")
                ckpt_list = glob.glob(search_pattern)
                ckpt_list.sort(key=os.path.getmtime)
                ckpt_path = ckpt_list[-1] 
            else:
                ckpt_path = config.GRPO.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            if config.GRPO.is_requeue:
                start_iter = ckpt_dict["iteration"]
            else:
                start_iter = 0

            if 'module' in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                incompatible_keys = self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
                self.policy.net = self.policy.net.module 
                self.waypoint_predictor = torch.nn.DataParallel(self.waypoint_predictor.to(self.device),
                    device_ids=[self.device], output_device=self.device)
            elif 'module' not in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS > 1:
                new_state_dict = OrderedDict()
                for k, v in ckpt_dict['state_dict'].items():
                    if k.startswith("net."):
                        name = k.replace("net.", "net.module.", 1)
                        new_state_dict[name] = v
                    else:
                        new_state_dict[k] = v
                incompatible_keys = self.policy.load_state_dict(new_state_dict, strict=False)
            else:
                incompatible_keys = self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)

            print("\n" + "="*25 + " Weight loading mismatch report " + "="*25)
            if incompatible_keys.missing_keys:
                print("The following network layers exist in the model but are missing in the weight file (initial values will be used):")
                for key in sorted(incompatible_keys.missing_keys):
                    print(f"  - {key}")
            else:
                print("All network layers present in the model were found in the weight file.")
            if incompatible_keys.unexpected_keys:
                print("\nThe following network layers exist in the weight file but are missing in the model (will be ignored):")
                for key in sorted(incompatible_keys.unexpected_keys):
                    print(f"  - {key}")
            else:
                print("\nThere are no extra network layers in the weight file.")
            print("="*75 + "\n")

            if getattr(self.config.GRPO, 'gauss_only', False):
                invalid_missing = [
                    key for key in incompatible_keys.missing_keys
                    if not key.endswith('gmap_gauss_embedding.weight')
                ]
                if invalid_missing or incompatible_keys.unexpected_keys:
                    raise RuntimeError(
                        'E2 checkpoint mismatch: missing=%s, unexpected=%s' %
                        (invalid_missing, incompatible_keys.unexpected_keys)
                    )
                gauss_weight = vln_bert_module.global_encoder.gmap_gauss_embedding.weight
                if (incompatible_keys.missing_keys and
                        torch.count_nonzero(gauss_weight).item() != 0):
                    raise RuntimeError(
                        'Missing Gaussian checkpoint weight must retain zero initialization'
                    )
            if getattr(self.config.GRPO, 'candidate_scorer_only', False):
                invalid_missing = [
                    key for key in incompatible_keys.missing_keys
                    if (not key.endswith('gmap_gauss_embedding.weight') and
                        '.candidate_scorer.' not in key)
                ]
                if invalid_missing or incompatible_keys.unexpected_keys:
                    raise RuntimeError(
                        'E3 checkpoint mismatch: missing=%s, unexpected=%s' %
                        (invalid_missing, incompatible_keys.unexpected_keys)
                    )
                scorer_missing = [
                    key for key in incompatible_keys.missing_keys
                    if '.candidate_scorer.' in key
                ]
                if scorer_missing:
                    missing_names = {
                        key.split('.candidate_scorer.', 1)[1]
                        for key in scorer_missing
                    }
                    expected_names = {
                        name for name, _ in
                        vln_bert_module.candidate_scorer.named_parameters()
                    }
                    if missing_names != expected_names:
                        raise RuntimeError(
                            'E3 checkpoint has a partial candidate scorer: %s' %
                            sorted(missing_names)
                        )
                    vln_bert_module.candidate_scorer.reset_output()
            if getattr(self.config.GRPO, 'gaussian_bev_only', False):
                invalid_missing = [
                    key for key in incompatible_keys.missing_keys
                    if (not key.endswith('gmap_gauss_embedding.weight') and
                        '.gaussian_bev.' not in key)
                ]
                if invalid_missing or incompatible_keys.unexpected_keys:
                    raise RuntimeError(
                        'E4 checkpoint mismatch: missing=%s, unexpected=%s' %
                        (invalid_missing, incompatible_keys.unexpected_keys)
                    )
                bev_missing = [
                    key for key in incompatible_keys.missing_keys
                    if '.gaussian_bev.' in key
                ]
                if bev_missing:
                    missing_names = {
                        key.split('.gaussian_bev.', 1)[1]
                        for key in bev_missing
                    }
                    expected_names = {
                        name for name, _ in
                        vln_bert_module.gaussian_bev.named_parameters()
                    }
                    if missing_names != expected_names:
                        raise RuntimeError(
                            'E4 checkpoint has a partial Gaussian BEV: %s' %
                            sorted(missing_names)
                        )
                    vln_bert_module.gaussian_bev.reset_output()
            if getattr(self.config.GRPO, 'anchor_repair_only', False):
                invalid_missing = [
                    key for key in incompatible_keys.missing_keys
                    if (not key.endswith('gmap_gauss_embedding.weight') and
                        '.anchor_repair.' not in key)
                ]
                if invalid_missing or incompatible_keys.unexpected_keys:
                    raise RuntimeError(
                        'E5 checkpoint mismatch: missing=%s, unexpected=%s' %
                        (invalid_missing, incompatible_keys.unexpected_keys)
                    )
                repair_missing = [
                    key for key in incompatible_keys.missing_keys
                    if '.anchor_repair.' in key
                ]
                if repair_missing:
                    missing_names = {
                        key.split('.anchor_repair.', 1)[1]
                        for key in repair_missing
                    }
                    expected_names = {
                        name for name, _ in
                        vln_bert_module.anchor_repair.named_parameters()
                    }
                    if missing_names != expected_names:
                        raise RuntimeError(
                            'E5 checkpoint has a partial anchor repair: %s' %
                            sorted(missing_names)
                        )
                    vln_bert_module.anchor_repair.reset_output()
            if getattr(self.config.GRPO, 'hindsight_stop_only', False):
                invalid_missing = [
                    key for key in incompatible_keys.missing_keys
                    if '.hindsight_stop.' not in key
                ]
                if invalid_missing or incompatible_keys.unexpected_keys:
                    raise RuntimeError(
                        'E6 checkpoint mismatch: missing=%s, unexpected=%s' %
                        (invalid_missing, incompatible_keys.unexpected_keys)
                    )
                stop_missing = [
                    key for key in incompatible_keys.missing_keys
                    if '.hindsight_stop.' in key
                ]
                if stop_missing:
                    missing_names = {
                        key.split('.hindsight_stop.', 1)[1]
                        for key in stop_missing
                    }
                    expected_names = {
                        name for name, _ in
                        vln_bert_module.hindsight_stop.named_parameters()
                    }
                    if missing_names != expected_names:
                        raise RuntimeError(
                            'E6 checkpoint has a partial hindsight-stop head: %s' %
                            sorted(missing_names)
                        )
                    vln_bert_module.hindsight_stop.reset_output()
            if getattr(self.config.GRPO, 'terminal_commit_only', False):
                invalid_missing = [
                    key for key in incompatible_keys.missing_keys
                    if '.terminal_commit.' not in key
                ]
                if invalid_missing or incompatible_keys.unexpected_keys:
                    raise RuntimeError(
                        'E7 checkpoint mismatch: missing=%s, unexpected=%s' %
                        (invalid_missing, incompatible_keys.unexpected_keys)
                    )
                commit_missing = [
                    key for key in incompatible_keys.missing_keys
                    if '.terminal_commit.' in key
                ]
                if commit_missing:
                    missing_names = {
                        key.split('.terminal_commit.', 1)[1]
                        for key in commit_missing
                    }
                    expected_names = {
                        name for name, _ in
                        vln_bert_module.terminal_commit.named_parameters()
                    }
                    if missing_names != expected_names:
                        raise RuntimeError(
                            'E7 checkpoint has a partial terminal-commit '
                            'head: %s' % sorted(missing_names)
                        )
                    vln_bert_module.terminal_commit.reset_output()
            if getattr(self.config.GRPO, 'geo_token_only', False):
                invalid_missing = [
                    key for key in incompatible_keys.missing_keys
                    if '.geo_token.' not in key
                ]
                if invalid_missing or incompatible_keys.unexpected_keys:
                    raise RuntimeError(
                        'E11 checkpoint mismatch: missing=%s, unexpected=%s' %
                        (invalid_missing, incompatible_keys.unexpected_keys)
                    )
                token_missing = [
                    key for key in incompatible_keys.missing_keys
                    if '.geo_token.' in key
                ]
                if token_missing:
                    missing_names = {
                        key.split('.geo_token.', 1)[1]
                        for key in token_missing
                    }
                    expected_names = {
                        name for name, _ in
                        vln_bert_module.geo_token.named_parameters()
                    }
                    if missing_names != expected_names:
                        raise RuntimeError(
                            'E11 checkpoint has a partial GeoToken module: %s' %
                            sorted(missing_names)
                        )
                    vln_bert_module.geo_token.reset_output()
            if frontier_advantage and (
                    incompatible_keys.missing_keys or
                    incompatible_keys.unexpected_keys):
                raise RuntimeError(
                    'E10 requires an exact E0 checkpoint: missing=%s, '
                    'unexpected=%s' % (
                        incompatible_keys.missing_keys,
                        incompatible_keys.unexpected_keys,
                    )
                )

            if config.GRPO.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                print("optimizer is load from checkpoint")
                if "scheduler_state" in ckpt_dict:
                    self.scheduler.load_state_dict(ckpt_dict["scheduler_state"])
            else:
                print("optimizer is initialized")
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}, iteration: {start_iter}")
		
        if self.need_ref_policy:
            self.ref_policy = policy.from_config(
                config=config,
                observation_space=observation_space,
                action_space=action_space,
            )
            self.ref_policy.to(self.device)
            self.ref_policy.eval() 
            for param in self.ref_policy.parameters():
                param.requires_grad = False
            if self.config.GPU_NUMBERS > 1:
                policy_state_dict = self.policy.state_dict()
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in policy_state_dict.items():
                    if k.startswith("net.module."):
                        name = k.replace("net.module.", "net.", 1)
                        new_state_dict[name] = v
                    else:
                        new_state_dict[k] = v
                self.ref_policy.load_state_dict(new_state_dict)
            else:
                self.ref_policy.load_state_dict(self.policy.state_dict())
        else:
            logger.info("BETA == 0, Skip create ref_policy!")

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params/1e6:.2f} MB. Trainable: {params_t/1e6:.2f} MB.")
        logger.info("Finished setting up policy.")
        return start_iter

    def _vp_feature_variable(self, obs):
        batch_rgb_fts, batch_dep_fts, batch_loc_fts = [], [], []
        batch_nav_types, batch_view_lens = [], []
        
        for i in range(self.envs.num_envs):
            rgb_fts, dep_fts, loc_fts , nav_types = [], [], [], []
            cand_idxes = np.zeros(12, dtype=np.bool)
            cand_idxes[obs['cand_img_idxes'][i]] = True

            rgb_fts.append(obs['cand_rgb'][i])
            dep_fts.append(obs['cand_depth'][i])
            loc_fts.append(obs['cand_angle_fts'][i])
            nav_types += [1] * len(obs['cand_angles'][i])

            rgb_fts.append(obs['pano_rgb'][i][~cand_idxes])
            dep_fts.append(obs['pano_depth'][i][~cand_idxes])
            loc_fts.append(obs['pano_angle_fts'][~cand_idxes])
            nav_types += [0] * (12-np.sum(cand_idxes))
            
            batch_rgb_fts.append(torch.cat(rgb_fts, dim=0))
            batch_dep_fts.append(torch.cat(dep_fts, dim=0))
            batch_loc_fts.append(torch.cat(loc_fts, dim=0))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_view_lens.append(len(nav_types))

        batch_rgb_fts = pad_tensors_wgrad(batch_rgb_fts)
        batch_dep_fts = pad_tensors_wgrad(batch_dep_fts)
        batch_loc_fts = pad_tensors_wgrad(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'rgb_fts': batch_rgb_fts, 'dep_fts': batch_dep_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
        }
        
    def _nav_gmap_variable(self, cur_vp, cur_pos, cur_ori, task_type):
        batch_gmap_vp_ids, batch_gmap_step_ids, batch_gmap_lens = [], [], []
        batch_gmap_img_fts, batch_gmap_pos_fts = [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        batch_gmap_task_embeddings = []

        for i, gmap in enumerate(self.gmaps):
            node_vp_ids = list(gmap.node_pos.keys())
            ghost_vp_ids = list(gmap.ghost_pos.keys())
            if len(ghost_vp_ids) == 0:
                batch_no_vp_left.append(True)
            else:
                batch_no_vp_left.append(False)

            gmap_vp_ids = [None] + node_vp_ids + ghost_vp_ids
            gmap_step_ids = [0] + [gmap.node_stepId[vp] for vp in node_vp_ids] + [0]*len(ghost_vp_ids)
            gmap_visited_masks = [0] + [1] * len(node_vp_ids) + [0] * len(ghost_vp_ids)

            gmap_img_fts = [gmap.get_node_embeds(vp) for vp in node_vp_ids] + \
                           [gmap.get_node_embeds(vp) for vp in ghost_vp_ids]
            gmap_img_fts = torch.stack(
                [torch.zeros_like(gmap_img_fts[0])] + gmap_img_fts, dim=0
            )

            gmap_pos_fts = gmap.get_pos_fts(
                cur_vp[i], cur_pos[i], cur_ori[i], gmap_vp_ids
            )
            gmap_pair_dists = np.zeros((len(gmap_vp_ids), len(gmap_vp_ids)), dtype=np.float32)
            for j in range(1, len(gmap_vp_ids)):
                for k in range(j+1, len(gmap_vp_ids)):
                    vp1 = gmap_vp_ids[j]
                    vp2 = gmap_vp_ids[k]
                    if not vp1.startswith('g') and not vp2.startswith('g'):
                        dist = gmap.shortest_dist[vp1][vp2]
                    elif not vp1.startswith('g') and vp2.startswith('g'):
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = gmap.shortest_dist[vp1][front_vp2] + front_dis2
                    elif vp1.startswith('g') and vp2.startswith('g'):
                        front_dis1, front_vp1 = gmap.front_to_ghost_dist(vp1)
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = front_dis1 + gmap.shortest_dist[front_vp1][front_vp2] + front_dis2
                    else:
                        raise NotImplementedError
                    gmap_pair_dists[j, k] = gmap_pair_dists[k, j] = dist / MAX_DIST
            
            batch_gmap_vp_ids.append(gmap_vp_ids)
            gmap_step_ids_tensor = torch.LongTensor(gmap_step_ids)
            batch_gmap_step_ids.append(gmap_step_ids_tensor)
            batch_gmap_task_embeddings.append(torch.full_like(gmap_step_ids_tensor, task_type))
            batch_gmap_lens.append(len(gmap_vp_ids))
            batch_gmap_img_fts.append(gmap_img_fts)
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
        
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_task_embeddings = pad_sequence(batch_gmap_task_embeddings, batch_first=True).cuda()
        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        batch_gmap_pos_fts = pad_tensors_wgrad(batch_gmap_pos_fts).cuda()
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        bs = self.envs.num_envs
        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(bs, max_gmap_len, max_gmap_len).float()
        for i in range(bs):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vp_ids': batch_gmap_vp_ids, 'gmap_step_ids': batch_gmap_step_ids,
            'gmap_img_fts': batch_gmap_img_fts, 'gmap_pos_fts': batch_gmap_pos_fts, 
            'gmap_masks': batch_gmap_masks, 'gmap_visited_masks': batch_gmap_visited_masks, 'gmap_pair_dists': gmap_pair_dists,
            'no_vp_left': batch_no_vp_left, 'gmap_task_embeddings': batch_gmap_task_embeddings
        }

    def train(self):
        self._set_config()
        if self.config.MODEL.task_type == 'rxr':
            self.gt_data = {}
            for role in self.config.TASK_CONFIG.DATASET.ROLES:
                with gzip.open(
                    self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                        split=self.split, role=role
                    ), "rt") as f:
                    self.gt_data.update(json.load(f))

        observation_space, action_space = self._init_envs()

        start_iter = self._initialize_policy(
            self.config,
            self.config.GRPO.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        total_iter = self.config.GRPO.iters 
        log_every  = self.config.GRPO.log_every 
        writer     = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        if self.config.local_rank < 1:
            config_path = os.path.join(self.config.CHECKPOINT_FOLDER, "config.yaml")
            with open(config_path, "w") as f:
                f.write(self.config.dump())
            logger.info(f"Configuration saved to {config_path}")

        logger.info('Traning Starts... GOOD LUCK!')
        
        self.data_buffer = []
        self.training_iteration = start_iter
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter-idx, 0)) 
            cur_iter = idx + interval 

            logs = self._train_interval(interval)

            final_logs = {}
            if self.world_size > 1:
                for k, v_list in logs.items():
                    if not v_list: 
                        local_sum = 0.0
                        local_count = 0
                    else:
                        local_sum = sum(v_list)
                        local_count = len(v_list)
                    metric_tensor = torch.tensor([local_sum, local_count], dtype=torch.float64, device=self.device)
                    distr.all_reduce(metric_tensor, op=distr.ReduceOp.SUM)
                    if self.local_rank < 1:
                        global_sum, global_count = metric_tensor[0].item(), metric_tensor[1].item()
                        if global_count > 0:
                            final_logs[k] = global_sum / global_count
                        else:
                            final_logs[k] = 0.0 
            else: 
                for k, v_list in logs.items():
                    if not v_list: continue
                    final_logs[k] = np.mean(v_list)

            if self.local_rank < 1:
                loss_str = f'iter {cur_iter}: '
                for k, avg_val in final_logs.items():
                    loss_str += f'{k}: {avg_val:.3f}, '
                    writer.add_scalar(f'grpo/{k}', avg_val, cur_iter)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                writer.add_scalar('train/lr', current_lr, cur_iter)
                logger.info(loss_str)
                logger.info(f"lr: {current_lr}")
                self.save_checkpoint(cur_iter)
        
    def _train_interval(self, interval):
        if self.world_size > 1:
            self.policy.net.module.rgb_encoder.eval()
            self.policy.net.module.depth_encoder.eval()
        else:
            self.policy.net.rgb_encoder.eval()
            self.policy.net.depth_encoder.eval()
        self.waypoint_predictor.eval()

        if self.local_rank < 1:
            pbar = tqdm.trange(interval, leave=False, dynamic_ncols=True)
        else:
            pbar = range(interval)
        self.logs = defaultdict(list)

        for idx in pbar:
            if self.frontier_advantage:
                self.frontier_advantage_weight = annealed_weight(
                    self.config.GRPO.frontier_advantage_start,
                    self.config.GRPO.frontier_advantage_end,
                    self.training_iteration,
                    self.config.GRPO.iters,
                )
            with torch.no_grad():
                with autocast(enabled=self.enable_amp):
                    self.sample_data(self.config.GRPO.sample_num)
            self.update()
            self.training_iteration += 1

            if self.local_rank < 1:
                pbar.set_postfix({'iter': f'{idx+1}/{interval}'})

        return deepcopy(self.logs)
    

    def update(self):
        if not self.data_buffer:
            logger.info("Data buffer is empty. Skipping update.")
            return

        if getattr(self.config.GRPO, 'terminal_commit_only', False):
            self._update_terminal_commit()
            return

        if getattr(self.config.GRPO, 'hindsight_stop_only', False):
            self._update_hindsight_stop()
            return

        self.set_policy_mode("train")

        # --- 1. Pre-calculate advantages for all trajectories (done once for the entire buffer) ---
        anchor_repair_only = getattr(
            self.config.GRPO, 'anchor_repair_only', False
        )
        setwise_group_policy = getattr(
            self.config.GRPO, 'setwise_group_policy', False
        )
        frontier_advantage = getattr(
            self.config.GRPO, 'frontier_advantage', False
        )
        advantages_all_ep_all_samples = [[0.0] * self.initial_num_envs for _ in range(self.config.GRPO.sample_num)]
        setwise_active_groups = None
        if setwise_group_policy:
            outcomes = [
                sample['setwise_outcome'] for sample in self.data_buffer
            ]
            advantages_all_ep_all_samples, setwise_active_groups = (
                setwise_group_advantages(outcomes)
            )
            valid_outcomes = [
                outcome for sample in outcomes for outcome in sample
                if outcome is not None
            ]
            self.logs['setwise_active_group_rate'].append(
                np.mean(setwise_active_groups)
            )
            self.logs['setwise_oracle_success'].append(
                np.mean([outcome[0] for outcome in valid_outcomes])
            )
            self.logs['setwise_best_spl'].append(
                np.mean([outcome[1] for outcome in valid_outcomes])
            )
            self.logs['setwise_min_distance'].append(
                np.mean([outcome[2] for outcome in valid_outcomes])
            )
        else:
            rewards_per_original_env_slot = [
                [] for _ in range(self.initial_num_envs)
            ]
            for s_idx in range(self.config.GRPO.sample_num):
                for original_env_idx in range(self.initial_num_envs):
                    reward_val = self.data_buffer[s_idx]["reward"][
                        original_env_idx
                    ]
                    if reward_val is not None:
                        rewards_per_original_env_slot[
                            original_env_idx
                        ].append(reward_val)

            for original_env_idx in range(self.initial_num_envs):
                rewards_for_this_slot = rewards_per_original_env_slot[
                    original_env_idx
                ]
                if len(rewards_for_this_slot) > 1:
                    mean_r = np.mean(rewards_for_this_slot)
                    std_r = np.std(rewards_for_this_slot)
                    for s_idx in range(self.config.GRPO.sample_num):
                        reward_val = self.data_buffer[s_idx]["reward"][
                            original_env_idx
                        ]
                        if reward_val is not None:
                            advantages_all_ep_all_samples[s_idx][
                                original_env_idx
                            ] = (reward_val - mean_r) / (std_r + 1e-8)

        sample_indices = range(self.config.GRPO.sample_num)
        if anchor_repair_only:
            rewards = [sample['reward'] for sample in self.data_buffer]
            advantages_all_ep_all_samples = anchor_relative_advantages(rewards)
            sample_indices = range(1, self.config.GRPO.sample_num)
            reward_deltas = []
            for original_env_idx in range(self.initial_num_envs):
                anchor_reward = rewards[0][original_env_idx]
                if anchor_reward is None:
                    continue
                for s_idx in sample_indices:
                    reward_val = rewards[s_idx][original_env_idx]
                    if reward_val is not None:
                        reward_deltas.append(reward_val - anchor_reward)
            self.logs['reward_delta_to_anchor'].append(
                np.mean(reward_deltas) if reward_deltas else 0.0
            )
            intervention_count = sum(
                sample['intervention_count'] for sample in self.data_buffer[1:]
            )
            decision_count = sum(
                sample['decision_count'] for sample in self.data_buffer[1:]
            )
            self.logs['intervention_rate'].append(
                intervention_count / max(decision_count, 1)
            )

        all_spls_in_buffer = []
        for s_idx in sample_indices:
            all_spls_in_buffer.extend([r for r in self.data_buffer[s_idx]["reward"] if r is not None])
        if all_spls_in_buffer:
            self.logs['reward'].append(np.mean(all_spls_in_buffer))
        else:
            self.logs['reward'].append(0.0)

        if frontier_advantage:
            local_advantages = [
                value.item()
                for sample in self.data_buffer
                for step in sample['data_buffer']
                for value in step['frontier_advantage']
            ]
            self.logs['frontier_advantage'].append(
                np.mean(local_advantages) if local_advantages else 0.0
            )
            self.logs['frontier_advantage_abs'].append(
                np.mean(np.abs(local_advantages))
                if local_advantages else 0.0
            )
            self.logs['frontier_advantage_weight'].append(
                self.frontier_advantage_weight
            )

        if (setwise_group_policy and
                not any(setwise_active_groups)):
            logger.info('All setwise groups are tied. Skipping update.')
            self.logs['policy_loss'].append(0.0)
            self.logs['kl_loss'].append(0.0)
            self.logs['total_loss'].append(0.0)
            self.data_buffer.clear()
            self.optimizer.zero_grad()
            self.scheduler.step()
            return

        total_policy_loss_across_epochs = 0.0
        total_kl_loss_across_epochs = 0.0
        total_combined_loss_across_epochs = 0.0
        actual_epochs_processed = 0

        total_processed_actions_for_ratio = 0
        total_unclipped_actions = 0

        # --- 2. Multi-epoch training loop over the same data buffer ---
        for epoch in range(self.grpo_update_epochs):
            accumulated_policy_loss_this_epoch = 0.0
            accumulated_kl_loss_this_epoch = 0.0
            num_samples_processed_this_epoch = 0 

            self.optimizer.zero_grad() 

            for s_idx in sample_indices:
                current_sample_trajectory_steps_data = self.data_buffer[s_idx]["data_buffer"]

                initial_txt_embeds_cuda = self.data_buffer[s_idx]["initial_txt_embeds"].to(self.device, non_blocking=True)
                initial_txt_masks_cuda = self.data_buffer[s_idx]["initial_txt_masks"].to(self.device, non_blocking=True)
                
                batch_total_policy_loss_for_this_sample = 0.0
                batch_total_kl_loss_for_this_sample = 0.0
                num_valid_steps_for_this_sample = 0
                total_actions_in_this_sample = 0
                with autocast(enabled=self.enable_amp):
                    for step_data in current_sample_trajectory_steps_data:
                        with autocast(enabled=False):
                            nav_inputs_cpu = step_data["input"]
                            taken_actions_cpu = step_data["action"]
                            old_probs_at_sampling_cpu = step_data["probs"]
                            active_indices_in_original_batch = step_data["indices"] 

                            if not active_indices_in_original_batch:
                                print("ERROR!! NO active_indices_in_original_batch")
                                continue
                        
                            nav_inputs_cuda = {}
                            for key, value in nav_inputs_cpu.items():
                                if isinstance(value, torch.Tensor):
                                    nav_inputs_cuda[key] = value.to(self.device, non_blocking=True)
                                else:
                                    nav_inputs_cuda[key] = value
                            txt_embeds_for_step = initial_txt_embeds_cuda[active_indices_in_original_batch]
                            txt_masks_for_step = initial_txt_masks_cuda[active_indices_in_original_batch]
                            nav_inputs_cuda['txt_embeds'] = txt_embeds_for_step
                            nav_inputs_cuda['txt_masks'] = txt_masks_for_step
                            nav_inputs_cuda['mode'] = 'navigation' 

                            taken_actions_cuda = taken_actions_cpu.to(self.device)

                            active_group_mask = None
                            if setwise_group_policy:
                                active_group_mask = torch.tensor(
                                    [setwise_active_groups[index]
                                     for index in active_indices_in_original_batch],
                                    device=self.device, dtype=torch.bool,
                                )
                                if not active_group_mask.any():
                                    continue
                    
                        current_policy_outputs = self.policy.net(**nav_inputs_cuda)
                        if self.need_ref_policy:
                            with torch.no_grad():
                                ref_policy_outputs = self.ref_policy.net(**nav_inputs_cuda)

                        with autocast(enabled=False):
                            current_logits = current_policy_outputs['global_logits']
                            current_log_probs = F.log_softmax(current_logits, dim=1)
                            current_log_probs_taken_action = current_log_probs.gather(1, taken_actions_cuda.unsqueeze(1)).squeeze(1)
                            if active_group_mask is not None:
                                current_log_probs_taken_action = (
                                    current_log_probs_taken_action[
                                        active_group_mask
                                    ]
                                )
                            if self.need_ref_policy:
                                ref_logits = ref_policy_outputs['global_logits']
                                ref_log_probs = F.log_softmax(ref_logits, dim=1)
                                ref_log_probs_taken_action_no_grad = ref_log_probs.gather(1, taken_actions_cuda.unsqueeze(1)).squeeze(1)
                                if active_group_mask is not None:
                                    ref_log_probs_taken_action_no_grad = (
                                        ref_log_probs_taken_action_no_grad[
                                            active_group_mask
                                        ]
                                    )

                            step_advantages_for_active_envs = torch.tensor(
                                [advantages_all_ep_all_samples[s_idx][orig_idx] for orig_idx in active_indices_in_original_batch],
                                device=self.device, dtype=torch.float32
                            )
                            if active_group_mask is not None:
                                step_advantages_for_active_envs = (
                                    step_advantages_for_active_envs[
                                        active_group_mask
                                    ]
                                )
                            if frontier_advantage:
                                local_advantages = step_data[
                                    'frontier_advantage'
                                ].to(self.device, non_blocking=True)
                                if active_group_mask is not None:
                                    local_advantages = local_advantages[
                                        active_group_mask
                                    ]
                                step_advantages_for_active_envs = mix_advantages(
                                    step_advantages_for_active_envs,
                                    local_advantages,
                                    self.frontier_advantage_weight,
                                )

                            if self.grpo_update_epochs == -1:
                                old_log_probs_taken_action = current_log_probs_taken_action.detach()
                            else:
                                old_log_probs_taken_action = torch.log(
                                    old_probs_at_sampling_cpu.to(self.device).gather(1, taken_actions_cuda.unsqueeze(1)).squeeze(1) + 1e-9
                                )
                                if active_group_mask is not None:
                                    old_log_probs_taken_action = (
                                        old_log_probs_taken_action[
                                            active_group_mask
                                        ]
                                    )
                            
                            ratio = torch.exp(current_log_probs_taken_action - old_log_probs_taken_action)
                            surr1 = ratio * step_advantages_for_active_envs
                            surr2 = torch.clamp(ratio, 1.0 - self.grpo_epsilon, 1.0 + self.grpo_epsilon) * step_advantages_for_active_envs
                            policy_loss_this_step = -torch.min(surr1, surr2).mean() 

                            unclipped_mask = (surr1 <= surr2)
                            total_unclipped_actions += unclipped_mask.sum().item()
                            total_processed_actions_for_ratio += len(unclipped_mask)

                            if self.need_ref_policy:
                                ratio_ref_over_current = torch.exp(ref_log_probs_taken_action_no_grad - current_log_probs_taken_action)
                                log_ratio_ref_over_current = ref_log_probs_taken_action_no_grad - current_log_probs_taken_action
                                kl_div_this_step = (ratio_ref_over_current - log_ratio_ref_over_current - 1).mean()
                            
                            batch_total_policy_loss_for_this_sample += policy_loss_this_step
                            if self.need_ref_policy:
                                batch_total_kl_loss_for_this_sample += kl_div_this_step
                            num_valid_steps_for_this_sample += 1
                
                if num_valid_steps_for_this_sample > 0:
                    avg_policy_loss_for_sample = batch_total_policy_loss_for_this_sample / num_valid_steps_for_this_sample
                    if self.need_ref_policy:
                        avg_kl_loss_for_sample = batch_total_kl_loss_for_this_sample / num_valid_steps_for_this_sample

                    if self.need_ref_policy:
                        total_loss_for_this_sample = avg_policy_loss_for_sample + self.grpo_beta * avg_kl_loss_for_sample
                    else:
                        total_loss_for_this_sample = avg_policy_loss_for_sample
                    scaled_loss_for_this_sample = total_loss_for_this_sample

                    if (s_idx < self.config.GRPO.sample_num - 1) and (self.world_size > 1):
                        with self.policy.net.no_sync():
                            self.scaler.scale(scaled_loss_for_this_sample).backward()
                    else:
                        self.scaler.scale(scaled_loss_for_this_sample).backward()

                    accumulated_policy_loss_this_epoch += avg_policy_loss_for_sample.item()
                    if self.need_ref_policy:
                        accumulated_kl_loss_this_epoch += avg_kl_loss_for_sample.item()
                    num_samples_processed_this_epoch +=1
                else:
                    logger.info("ERROR! total_actions_in_this_sample is 0")
            
            expected_samples = (
                self.config.GRPO.sample_num - 1
                if anchor_repair_only else self.config.GRPO.sample_num
            )
            if num_samples_processed_this_epoch == expected_samples:
                self.scaler.unscale_(self.optimizer)

                trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
                if trainable_params:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                    self.logs['grad_norm'].append(grad_norm.item())
                
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_policy_loss_across_epochs += (accumulated_policy_loss_this_epoch / num_samples_processed_this_epoch)
                if self.need_ref_policy:
                    total_kl_loss_across_epochs += (accumulated_kl_loss_this_epoch / num_samples_processed_this_epoch)
                if self.need_ref_policy:
                    total_combined_loss_across_epochs += ( (accumulated_policy_loss_this_epoch + self.grpo_beta * accumulated_kl_loss_this_epoch) / num_samples_processed_this_epoch )
                else:
                    total_combined_loss_across_epochs += ( (accumulated_policy_loss_this_epoch) / num_samples_processed_this_epoch )
                actual_epochs_processed += 1
            else:
                logger.info(
                    f"Epoch {epoch+1}/{self.grpo_update_epochs}: "
                    f"processed {num_samples_processed_this_epoch}/"
                    f"{expected_samples} policy samples."
                )

        if actual_epochs_processed > 0:
            self.logs['policy_loss'].append(total_policy_loss_across_epochs / actual_epochs_processed)
            if self.need_ref_policy:
                self.logs['kl_loss'].append(total_kl_loss_across_epochs / actual_epochs_processed)
            self.logs['total_loss'].append(total_combined_loss_across_epochs / actual_epochs_processed)

            if total_processed_actions_for_ratio > 0:
                unclipped_ratio = total_unclipped_actions / total_processed_actions_for_ratio
                self.logs['unclipped_ratio'].append(unclipped_ratio)
            else:
                self.logs['unclipped_ratio'].append(0.0)
        else:
            logger.info("No valid data processed across all epochs. Skipping logging of losses.")
            self.logs['policy_loss'].append(0.0)
            self.logs['kl_loss'].append(0.0)
            self.logs['total_loss'].append(0.0)

        self.data_buffer.clear() 
        self.optimizer.zero_grad()
        self.scheduler.step()

    def _update_hindsight_stop(self):
        self.set_policy_mode("train")
        sample = self.data_buffer[0]
        steps = sample["data_buffer"]
        if not steps:
            logger.info("Hindsight-stop trajectory is empty. Skipping update.")
            self.data_buffer.clear()
            return

        self.optimizer.zero_grad()
        total_loss = 0.0
        prediction_count = 0
        rollback_count = 0
        recovery_count = 0
        regression_count = 0
        stop_advantages = []

        initial_txt_embeds = sample["initial_txt_embeds"].to(
            self.device, non_blocking=True
        )
        initial_txt_masks = sample["initial_txt_masks"].to(
            self.device, non_blocking=True
        )
        training_examples = sum(
            len(step_data['indices']) for step_data in steps
        )

        for step_data in steps:
            active_indices = step_data["indices"]
            nav_inputs = {
                key: value.to(self.device, non_blocking=True)
                if isinstance(value, torch.Tensor) else value
                for key, value in step_data["input"].items()
            }
            nav_inputs.update({
                'txt_embeds': initial_txt_embeds[active_indices],
                'txt_masks': initial_txt_masks[active_indices],
                'mode': 'navigation',
            })
            outputs = self.policy.net(**nav_inputs)
            logits = outputs['hindsight_stop_logits']
            stop_returns = step_data['stop_returns'].to(
                self.device, non_blocking=True
            )
            valid_mask = step_data['stop_valid_mask'].to(
                self.device, non_blocking=True
            )
            continue_returns = torch.tensor(
                [sample['reward'][index] for index in active_indices],
                dtype=stop_returns.dtype,
                device=self.device,
            )
            targets = hindsight_stop_targets(
                stop_returns, continue_returns, valid_mask
            )
            loss = F.cross_entropy(
                logits, targets, reduction='sum'
            ) / training_examples
            self.scaler.scale(loss).backward()
            total_loss += loss.item()

            with torch.no_grad():
                predictions = logits.argmax(dim=1)
                stop_success = step_data['stop_success'].to(
                    self.device, non_blocking=True
                )
                rows = torch.arange(predictions.size(0), device=self.device)
                predicted_success = stop_success[rows, predictions]
                continue_success = torch.tensor(
                    [sample['success'][index] for index in active_indices],
                    dtype=torch.bool,
                    device=self.device,
                )
                predicted_stop = predictions != 0
                prediction_count += predictions.numel()
                rollback_count += predicted_stop.sum().item()
                recovery_count += (
                    predicted_stop & predicted_success &
                    continue_success.logical_not()
                ).sum().item()
                regression_count += (
                    predicted_stop & predicted_success.logical_not() &
                    continue_success
                ).sum().item()
                best_stop = stop_returns[:, 1:].max(dim=1).values
                stop_advantages.extend(
                    (best_stop - continue_returns).cpu().tolist()
                )

        self.scaler.unscale_(self.optimizer)
        trainable_params = [
            parameter for parameter in self.policy.parameters()
            if parameter.requires_grad
        ]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params, self.max_grad_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        final_success = [
            value for value in sample['success'] if value is not None
        ]
        oracle_success = [
            value for value in sample['oracle_success'] if value is not None
        ]
        self.logs['hindsight_stop_loss'].append(total_loss)
        self.logs['total_loss'].append(total_loss)
        self.logs['grad_norm'].append(grad_norm.item())
        self.logs['rollback_rate'].append(
            rollback_count / max(prediction_count, 1)
        )
        self.logs['stop_recoveries'].append(
            recovery_count / max(prediction_count, 1)
        )
        self.logs['stop_regressions'].append(
            regression_count / max(prediction_count, 1)
        )
        self.logs['counterfactual_stop_advantage'].append(
            np.mean(stop_advantages) if stop_advantages else 0.0
        )
        if final_success and oracle_success:
            self.logs['osr_sr_gap'].append(
                np.mean(oracle_success) - np.mean(final_success)
            )
        self.logs['reward'].extend(sample['reward'])

        self.data_buffer.clear()
        self.optimizer.zero_grad()

    def _update_terminal_commit(self):
        self.set_policy_mode("train")
        sample = self.data_buffer[0]
        examples = sample["terminal_examples"]
        if not examples:
            logger.info("Terminal-commit trajectory has no terminal example.")
            self.data_buffer.clear()
            return

        self.optimizer.zero_grad()
        total_loss = 0.0
        reselections = 0
        recoveries = 0
        regressions = 0
        reach_rewards = []
        set_losses = []
        success_set_sizes = []
        use_success_set = getattr(
            self.config.GRPO, 'success_set_commit', False
        )
        initial_txt_embeds = sample["initial_txt_embeds"].to(
            self.device, non_blocking=True
        )
        initial_txt_masks = sample["initial_txt_masks"].to(
            self.device, non_blocking=True
        )

        for example in examples:
            nav_inputs = {
                key: value.to(self.device, non_blocking=True)
                if isinstance(value, torch.Tensor) else value
                for key, value in example["input"].items()
            }
            index = example["index"]
            nav_inputs.update({
                'txt_embeds': initial_txt_embeds[index:index + 1],
                'txt_masks': initial_txt_masks[index:index + 1],
                'mode': 'navigation',
            })
            logits = self.policy.net(**nav_inputs)[
                'terminal_commit_logits'
            ]
            returns = example['returns'].to(
                self.device, non_blocking=True
            )
            success = example['success'].to(self.device)
            if use_success_set:
                loss = success_set_nll(logits, success)
                if loss is not None:
                    set_losses.append(loss)
                    success_set_sizes.extend(
                        (success & torch.isfinite(logits)).sum(
                            dim=1
                        ).cpu().tolist()
                    )
            else:
                target = returns.argmax(dim=1)
                loss = F.cross_entropy(logits, target) / len(examples)
                self.scaler.scale(loss).backward()
                total_loss += loss.item()

            with torch.no_grad():
                prediction = logits.argmax(dim=1)
                base_choice = example['base_choice'].to(self.device)
                rows = torch.arange(prediction.size(0), device=self.device)
                predicted_success = success[rows, prediction]
                base_success = success[rows, base_choice]
                reselections += (prediction != base_choice).sum().item()
                recoveries += (
                    predicted_success & base_success.logical_not()
                ).sum().item()
                regressions += (
                    predicted_success.logical_not() & base_success
                ).sum().item()
                reach_rewards.extend(returns.max(dim=1).values.cpu().tolist())

        if use_success_set:
            if not set_losses:
                logger.info(
                    'Success-set trajectory has no recoverable terminal.'
                )
                self.optimizer.step()
                self.scheduler.step()
                self.logs['recoverable_terminal_rate'].append(0.0)
                self.logs['success_set_size'].append(0.0)
                self.logs['final_reward'].extend(sample['reward'])
                self.data_buffer.clear()
                self.optimizer.zero_grad()
                return
            loss = torch.stack(set_losses).mean()
            self.scaler.scale(loss).backward()
            total_loss = loss.item()

        self.scaler.unscale_(self.optimizer)
        trainable_params = [
            parameter for parameter in self.policy.parameters()
            if parameter.requires_grad
        ]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params, self.max_grad_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        count = len(examples)
        loss_name = (
            'success_set_commit_loss' if use_success_set
            else 'terminal_commit_loss'
        )
        self.logs[loss_name].append(total_loss)
        self.logs['total_loss'].append(total_loss)
        self.logs['grad_norm'].append(grad_norm.item())
        self.logs['terminal_reselection_rate'].append(reselections / count)
        self.logs['terminal_recoveries'].append(recoveries / count)
        self.logs['terminal_regressions'].append(regressions / count)
        self.logs['reach_reward'].append(np.mean(reach_rewards))
        self.logs['final_reward'].extend(sample['reward'])
        if use_success_set:
            self.logs['recoverable_terminal_rate'].append(
                len(set_losses) / count
            )
            self.logs['success_set_size'].append(
                np.mean(success_set_sizes)
            )

        self.data_buffer.clear()
        self.optimizer.zero_grad()
    
    def get_pos_ori(self):
        pos_ori = self.envs.call(['get_pos_ori']*self.envs.num_envs)
        pos = [x[0] for x in pos_ori]
        ori = [x[1] for x in pos_ori]
        return pos, ori

    def copy_nav_inputs_dict(self, input_dict):
        copied_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                copied_dict[key] = value.cpu()
            elif isinstance(value, list):
                copied_dict[key] = copy.deepcopy(value)
            elif isinstance(value, str):
                copied_dict[key] = value
            else:
                copied_dict[key] = copy.deepcopy(value) 
        return copied_dict

    def _frontier_action_advantages(self, nav_inputs, cur_pos):
        valid_mask = (
            nav_inputs['gmap_masks'] &
            nav_inputs['gmap_visited_masks'].logical_not()
        ).detach().cpu()
        candidate_distances = torch.full(
            valid_mask.shape, float('inf'), dtype=torch.float32
        )
        current_distances = torch.full(
            (valid_mask.size(0),), float('inf'), dtype=torch.float32
        )

        for i, gmap in enumerate(self.gmaps):
            stop_vp = max(
                gmap.node_stop_scores,
                key=gmap.node_stop_scores.get,
            )
            action_indices = [0]
            positions = [cur_pos[i], gmap.node_pos[stop_vp]]
            for action_index in torch.nonzero(
                    valid_mask[i, 1:], as_tuple=False).flatten().tolist():
                action_index += 1
                ghost_vp = nav_inputs['gmap_vp_ids'][i][action_index]
                action_indices.append(action_index)
                positions.append(gmap.ghost_aug_pos[ghost_vp])

            distances = self.envs.call_at(
                i,
                'point_dists_to_goal',
                {'positions': positions, 'is_train': True},
            )
            current_distances[i] = distances[0]
            candidate_distances[i, action_indices] = torch.tensor(
                distances[1:], dtype=torch.float32
            )

        return action_set_advantages(
            current_distances,
            candidate_distances,
            valid_mask,
            success_distance=float(
                self.config.TASK_CONFIG.TASK.SUCCESS_DISTANCE
            ),
            progress_clip=float(
                self.config.GRPO.frontier_progress_clip
            ),
        )

    def sample_data(self, sample_num):
        if not self.dropout_in_sampling:
            self.set_policy_mode("eval")
        else:
            self.set_policy_mode("train")
        
        anchor_repair_only = getattr(
            self.config.GRPO, 'anchor_repair_only', False
        )
        anchor_episode_ids = None
        for i in range(sample_num):
            self.envs.resume_all()
            if i == 0:
                observations = self.envs.reset()
            else:
                observations = self.envs.call(['reset_current_episode']*self.envs.num_envs)
            
            episodes_reset_ids = [ep.episode_id for i, ep in enumerate(self.envs.current_episodes())]
            if anchor_episode_ids is None:
                anchor_episode_ids = episodes_reset_ids
            elif episodes_reset_ids != anchor_episode_ids:
                raise RuntimeError('GRPO samples did not reset to the anchor episodes')

            data_this_sample = self.sample_once(
                observations,
                use_base_policy=anchor_repair_only and i == 0,
            )
            self.data_buffer.append(data_this_sample)

    def sample_once(self, initial_obs, use_base_policy=False):
        mode = 'train'
        hindsight_stop_only = getattr(
            self.config.GRPO, 'hindsight_stop_only', False
        )
        terminal_commit_only = getattr(
            self.config.GRPO, 'terminal_commit_only', False
        )
        setwise_group_policy = getattr(
            self.config.GRPO, 'setwise_group_policy', False
        )
        frontier_advantage = getattr(
            self.config.GRPO, 'frontier_advantage', False
        )

        instr_max_len = self.config.GRPO.max_text_len
        instr_pad_id = 1
        if self.config.MODEL.task_type == 'r2r':
            task_type = 1
        elif self.config.MODEL.task_type == 'rxr':
            task_type = 2
        else:
            print("self.config.MODEL.task_type Error")

        observations = extract_instruction_tokens(initial_obs, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID, 
                                                  max_length=instr_max_len, pad_id=instr_pad_id, task_type=task_type)
        batch = batch_obs(observations, self.device) 
        batch = apply_obs_transforms_batch(batch, self.obs_transforms) 

        # encode instructions
        all_txt_ids = batch['instruction'] 
        all_txt_task_encoding = batch['txt_task_encoding']
        all_txt_masks = (all_txt_ids != instr_pad_id)
        all_txt_embeds = self.policy.net(
            mode='language',
            txt_ids=all_txt_ids,
            txt_task_encoding=all_txt_task_encoding,
            txt_masks=all_txt_masks,
        )

        data_this_sample = {
            "reward": [None] * self.envs.num_envs,
            "data_buffer": [],
            "initial_txt_embeds": all_txt_embeds.detach().cpu(),
            "initial_txt_masks": all_txt_masks.detach().cpu(),
            "intervention_count": 0,
            "decision_count": 0,
            "success": [None] * self.envs.num_envs,
            "oracle_success": [None] * self.envs.num_envs,
            "setwise_outcome": [None] * self.envs.num_envs,
            "terminal_examples": [],
        }

        total_actions = 0.
        
        not_done_index = list(range(self.envs.num_envs)) 
        have_real_pos = (mode == 'train' or self.config.VIDEO_OPTION) 
        ghost_aug = self.config.GRPO.ghost_aug if mode == 'train' else 0
        self.gmaps = [GraphMap(have_real_pos, 
                               self.config.GRPO.loc_noise, 
                               self.config.MODEL.merge_ghost, 
                               ghost_aug,
                               gauss_feat_size=getattr(
                                   self.config.MODEL, 'gauss_feat_size', 0
                               )) for _ in range(self.envs.num_envs)]
        prev_vp = [None] * self.envs.num_envs
        path_lengths = [0.0] * self.envs.num_envs
        shortest_path_lengths = [None] * self.envs.num_envs
        goal_distance_cache = [dict() for _ in range(self.envs.num_envs)]

        for stepk in range(self.max_len): 
            total_actions += self.envs.num_envs
            txt_masks = all_txt_masks
            txt_embeds = all_txt_embeds
            
            wp_outputs = self.policy.net(
                mode = "waypoint",
                waypoint_predictor = self.waypoint_predictor,
                observations = batch,
                in_train = (mode == 'train' and self.config.GRPO.waypoint_aug),
            )

            # pano encoder
            vp_inputs = self._vp_feature_variable(wp_outputs)
            vp_inputs.update({
                'mode': 'panorama',
            })
            pano_embeds, pano_masks = self.policy.net(**vp_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            cur_pos, cur_ori = self.get_pos_ori()
            cur_vp, cand_vp, cand_pos = [], [], []
            for i in range(self.envs.num_envs):
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                    cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                )
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i) 
            
            if mode == 'train' or self.config.VIDEO_OPTION:
                cand_real_pos = []
                for i in range(self.envs.num_envs):
                    cand_real_pos_i = [
                        self.envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis})
                        for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                    ]
                    cand_real_pos.append(cand_real_pos_i)
            else:
                cand_real_pos = [None] * self.envs.num_envs

            candidate_targets = []
            for i in range(self.envs.num_envs):
                cur_embeds = avg_pano_embeds[i]
                cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i]==1] 
                candidate_targets.append(self.gmaps[i].update_graph(
                    prev_vp[i], stepk+1,
                    cur_vp[i], cur_pos[i], cur_embeds,
                    cand_vp[i], cand_pos[i], cand_embeds,
                    cand_real_pos[i],
                ))

            if hindsight_stop_only or terminal_commit_only:
                for i, gmap in enumerate(self.gmaps):
                    distance = self.envs.call_at(
                        i, 'point_dist_to_goal',
                        {'pos': gmap.node_pos[cur_vp[i]], 'is_train': True},
                    )
                    goal_distance_cache[i][cur_vp[i]] = distance
                    if shortest_path_lengths[i] is None:
                        shortest_path_lengths[i] = distance

            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori, task_type)
            if 'cand_geo_tokens' in wp_outputs:
                geo_tokens, geo_masks = align_candidate_tokens(
                    nav_inputs['gmap_vp_ids'], candidate_targets,
                    wp_outputs['cand_geo_tokens'],
                    wp_outputs['cand_geo_masks'],
                )
                nav_inputs['gmap_geo_tokens'] = geo_tokens
                nav_inputs['gmap_geo_masks'] = geo_masks
            nav_inputs.update({
                'mode': 'navigation',
            })
            no_vp_left = nav_inputs.pop('no_vp_left') 

            if terminal_commit_only:
                gmap_stop_scores = torch.zeros_like(
                    nav_inputs['gmap_step_ids'], dtype=torch.float32
                )
                for i, gmap in enumerate(self.gmaps):
                    for j, vp in enumerate(nav_inputs['gmap_vp_ids'][i][1:], 1):
                        if vp in gmap.node_stop_scores:
                            gmap_stop_scores[i, j] = gmap.node_stop_scores[vp]
                nav_inputs['gmap_stop_scores'] = gmap_stop_scores

            stop_returns = None
            stop_success = None
            stop_valid_mask = None
            if hindsight_stop_only:
                max_gmap_len = nav_inputs['gmap_step_ids'].size(1)
                stop_returns = torch.full(
                    (self.envs.num_envs, max_gmap_len),
                    -float('inf'), dtype=torch.float32,
                )
                stop_success = torch.zeros(
                    self.envs.num_envs, max_gmap_len, dtype=torch.bool
                )
                stop_valid_mask = (
                    nav_inputs['gmap_masks'] &
                    nav_inputs['gmap_visited_masks']
                ).detach().cpu()
                stop_valid_mask[:, 0] = True
                success_distance = float(
                    self.config.TASK_CONFIG.TASK.SUCCESS_DISTANCE
                )
                for i, gmap in enumerate(self.gmaps):
                    node_vps = nav_inputs['gmap_vp_ids'][i][
                        1:1 + len(gmap.node_pos)
                    ]
                    goal_distances = torch.tensor([
                        goal_distance_cache[i][vp] for vp in node_vps
                    ], dtype=torch.float32).unsqueeze(0)
                    rollback_distances = torch.tensor([
                        gmap.shortest_dist[cur_vp[i]][vp]
                        for vp in node_vps
                    ], dtype=torch.float32).unsqueeze(0)
                    values = counterfactual_stop_returns(
                        goal_distances,
                        rollback_distances,
                        path_length=torch.tensor([path_lengths[i]]),
                        shortest_path_length=torch.tensor([
                            shortest_path_lengths[i]
                        ]),
                        success_distance=success_distance,
                        distance_scale=6.0,
                    ).squeeze(0)
                    stop_returns[i, 1:1 + len(node_vps)] = values
                    stop_success[i, 1:1 + len(node_vps)] = (
                        goal_distances.squeeze(0) <= success_distance
                    )

            nav_inputs_for_gpu = nav_inputs.copy()
            nav_inputs_for_gpu['txt_embeds'] = txt_embeds
            nav_inputs_for_gpu['txt_masks'] = txt_masks

            nav_inputs_copy_for_cpu = self.copy_nav_inputs_dict(nav_inputs)
            nav_outs = self.policy.net(**nav_inputs_for_gpu)
            if use_base_policy:
                nav_logits = nav_outs['base_global_logits']
            else:
                nav_logits = nav_outs['global_logits']
            nav_probs = F.softmax(nav_logits, 1)
            stop_score_logits = nav_outs.get(
                'base_global_logits', nav_logits
            )
            stop_score_probs = F.softmax(stop_score_logits, 1)

            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = (
                    stop_score_probs[i, 0].data.item()
                )

            # determine action
            if hindsight_stop_only or terminal_commit_only or use_base_policy:
                a_t = nav_logits.argmax(dim=-1)
            else:
                c = torch.distributions.Categorical(nav_probs)
                a_t = c.sample().detach()
                if getattr(self.config.GRPO, 'anchor_repair_only', False):
                    base_actions = nav_outs['base_global_logits'].argmax(dim=-1)
                    data_this_sample['intervention_count'] += (
                        a_t != base_actions
                    ).sum().item()
                    data_this_sample['decision_count'] += a_t.numel()
            cpu_a_t = a_t.cpu().numpy()

            chosen_frontier_advantages = None
            if frontier_advantage:
                if stepk == self.max_len - 1:
                    chosen_frontier_advantages = torch.zeros(
                        self.envs.num_envs, dtype=torch.float32
                    )
                else:
                    frontier_advantages = self._frontier_action_advantages(
                        nav_inputs, cur_pos
                    )
                    chosen_frontier_advantages = frontier_advantages.gather(
                        1, a_t.detach().cpu().unsqueeze(1)
                    ).squeeze(1)
                    chosen_frontier_advantages[
                        torch.tensor(no_vp_left, dtype=torch.bool)
                    ] = 0.0

            if terminal_commit_only:
                success_distance = float(
                    self.config.TASK_CONFIG.TASK.SUCCESS_DISTANCE
                )
                for i, gmap in enumerate(self.gmaps):
                    will_stop = (
                        cpu_a_t[i] == 0 or stepk == self.max_len - 1 or
                        no_vp_left[i]
                    )
                    if not will_stop:
                        continue
                    node_vps = nav_inputs['gmap_vp_ids'][i][
                        1:1 + len(gmap.node_pos)
                    ]
                    goal_distances = torch.tensor([
                        goal_distance_cache[i][vp] for vp in node_vps
                    ], dtype=torch.float32).unsqueeze(0)
                    rollback_distances = torch.tensor([
                        gmap.shortest_dist[cur_vp[i]][vp]
                        for vp in node_vps
                    ], dtype=torch.float32).unsqueeze(0)
                    values = counterfactual_stop_returns(
                        goal_distances,
                        rollback_distances,
                        path_length=torch.tensor([path_lengths[i]]),
                        shortest_path_length=torch.tensor([
                            shortest_path_lengths[i]
                        ]),
                        success_distance=success_distance,
                        distance_scale=6.0,
                    ).squeeze(0)
                    returns = torch.full(
                        (1, nav_inputs['gmap_step_ids'].size(1)),
                        -float('inf'), dtype=torch.float32,
                    )
                    success = torch.zeros_like(returns, dtype=torch.bool)
                    returns[0, 1:1 + len(node_vps)] = values
                    success[0, 1:1 + len(node_vps)] = (
                        goal_distances.squeeze(0) <= success_distance
                    )
                    base_choice = max(
                        range(1, 1 + len(node_vps)),
                        key=lambda index: gmap.node_stop_scores[node_vps[index - 1]],
                    )
                    example_input = {}
                    for key, value in nav_inputs_copy_for_cpu.items():
                        if isinstance(value, torch.Tensor):
                            example_input[key] = value[i:i + 1]
                        elif isinstance(value, list):
                            example_input[key] = [copy.deepcopy(value[i])]
                        else:
                            example_input[key] = copy.deepcopy(value)
                    data_this_sample['terminal_examples'].append({
                        'input': example_input,
                        'index': not_done_index[i],
                        'returns': returns,
                        'success': success,
                        'base_choice': torch.tensor([base_choice]),
                    })

            # ------------------- start store data ------------------- 
            data_this_stepk = {}
            data_this_stepk["input"] = nav_inputs_copy_for_cpu 
            data_this_stepk["action"] = a_t.detach().cpu() 
            data_this_stepk["probs"] = nav_probs.detach().cpu() 
            data_this_stepk["indices"] = copy.deepcopy(not_done_index) 
            if frontier_advantage:
                data_this_stepk['frontier_advantage'] = (
                    chosen_frontier_advantages
                )
            if hindsight_stop_only:
                data_this_stepk["stop_returns"] = stop_returns
                data_this_stepk["stop_success"] = stop_success
                data_this_stepk["stop_valid_mask"] = stop_valid_mask
            if not terminal_commit_only:
                data_this_sample['data_buffer'].append(data_this_stepk)
            # ------------------- end store data ------------------- 

            # make equiv action
            env_actions = []
            use_tryout = (self.config.GRPO.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING) 
            for i, gmap in enumerate(self.gmaps):
                if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]: 
                    vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                    stop_scores = [s[1] for s in vp_stop_scores]
                    stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                    stop_pos = gmap.node_pos[stop_vp]

                    if self.config.GRPO.back_algo == 'control': 
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': stop_pos,
                    }
                    env_actions.append(
                        {
                            'action': {
                                'act': 0,
                                'cur_vp': cur_vp[i],
                                'stop_vp': stop_vp, 'stop_pos': stop_pos,
                                'back_path': back_path,
                                'tryout': use_tryout
                            },
                            'vis_info': vis_info,
                        }
                    )
                    
                else:
                    ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp) 
                    front_pos = gmap.node_pos[front_vp]
                    vis_info = None
                    if self.config.GRPO.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    env_actions.append(
                        {
                            'action': {
                                'act': 4,
                                'cur_vp': cur_vp[i],
                                'front_vp': front_vp, 'front_pos': front_pos,
                                'ghost_vp': ghost_vp, 'ghost_pos': ghost_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                    prev_vp[i] = front_vp
                    if self.config.MODEL.consume_ghost:
                        gmap.delete_ghost(ghost_vp)

            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            if hindsight_stop_only or terminal_commit_only:
                for i, info in enumerate(infos):
                    prefix_path = np.asarray(
                        info['position_train']['position']
                    )
                    path_lengths[i] = float(np.linalg.norm(
                        prefix_path[1:] - prefix_path[:-1], axis=1
                    ).sum())

            # calculate metric
            curr_eps = self.envs.current_episodes()
            if self.config.MODEL.task_type == 'r2r':
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    pred_path = np.array(info['position_train']['position'])
                    distances = np.array(info['position_train']['distance'])
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric['distance_to_goal'] = distances[-1]
                    success_distance = (
                        float(self.config.TASK_CONFIG.TASK.SUCCESS_DISTANCE)
                        if (hindsight_stop_only or terminal_commit_only or
                            setwise_group_policy or frontier_advantage)
                        else 1.5
                    )
                    metric['success'] = 1. if distances[-1] <= success_distance else 0.
                    metric['oracle_success'] = 1. if (distances <= success_distance).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1],axis=1).sum())
                    gt_length = distances[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    distance_to_goal_reward = 1 - 1 * (metric['distance_to_goal']) / 6
                    data_this_sample["reward"][not_done_index[i]] = metric['spl'] + metric['success'] + distance_to_goal_reward
                    data_this_sample["success"][not_done_index[i]] = bool(
                        metric['success']
                    )
                    data_this_sample["oracle_success"][not_done_index[i]] = any(
                        distance <= success_distance
                        for distance in goal_distance_cache[i].values()
                    ) if (hindsight_stop_only or terminal_commit_only) else bool(metric['oracle_success'])
                    if setwise_group_policy:
                        data_this_sample["setwise_outcome"][
                            not_done_index[i]
                        ] = recoverable_outcome(
                            pred_path, distances, success_distance
                        )
                    self.logs['spl_reward'].append(metric['spl'])
                    self.logs['success_reward'].append(metric['success'])
                    self.logs['NE'].append(metric['distance_to_goal'])
                    self.logs['distance_to_goal_reward'].append(distance_to_goal_reward)
            elif self.config.MODEL.task_type == 'rxr':
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
                    pred_path = np.array(info['position_train']['position'])
                    distances = np.array(info['position_train']['distance'])
                    gt_length = max(self.gt_data[str(ep_id)]['forward_steps']*0.25, distances[0])
                    metric = {}
                    metric['distance_to_goal'] = distances[-1]
                    metric['success'] = 1. if distances[-1] <= 1.5 else 0.
                    dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    metric['sdtw'] = metric['ndtw'] * metric['success']
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1],axis=1).sum())
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    distance_to_goal_reward = 1 - 1 * (metric['distance_to_goal']) / 6
                    data_this_sample["reward"][not_done_index[i]] = metric['ndtw'] + metric['sdtw'] + distance_to_goal_reward + metric['spl']
                    self.logs['ndtw'].append(metric['ndtw'])
                    self.logs['success'].append(metric['success'])
                    self.logs['spl'].append(metric['spl'])
                    self.logs['sdtw'].append(metric['sdtw'])
                    self.logs['NE'].append(metric['distance_to_goal'])
                    self.logs['distance_to_goal_reward'].append(distance_to_goal_reward)

            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:
                        stopped_env_index = not_done_index.pop(i)
                        self.envs.pause_at(i)
                        observations.pop(i)
                        self.gmaps.pop(i)
                        prev_vp.pop(i)
                        path_lengths.pop(i)
                        shortest_path_lengths.pop(i)
                        goal_distance_cache.pop(i)
                        all_txt_ids = torch.cat((all_txt_ids[:i], all_txt_ids[i + 1:]), dim=0)
                        all_txt_task_encoding = torch.cat((all_txt_task_encoding[:i], all_txt_task_encoding[i + 1:]), dim=0)
                        all_txt_masks = torch.cat((all_txt_masks[:i], all_txt_masks[i + 1:]), dim=0)
                        all_txt_embeds = torch.cat((all_txt_embeds[:i], all_txt_embeds[i + 1:]), dim=0)

            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens(observations,self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID, \
                                                                         max_length=instr_max_len, pad_id=instr_pad_id, task_type=task_type)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        return data_this_sample
