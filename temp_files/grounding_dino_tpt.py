# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union,List
# from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from mmengine.runner.amp import autocast
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from .dino import DINO
from .grounding_dino import GroundingDINO
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   run_ner)
from mmengine.optim import OptimWrapper                   

def clean_label_name(name: str) -> str:
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    return name


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert (counter == len(lst))

    return all_

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
@MODELS.register_module()
class PromptLearner(nn.Module):
    def __init__(self, language_model,text_feat_map, classnames,device, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        # dtype = clip_model.dtype
        self.dtype = torch.float32
        # self.device = clip_model.visual.conv1.weight.device
        # ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim =256 #768 #256#768 
        self.batch_size = batch_size
        # self.language_model = language_model
        # self.tokenizer = self.language_model.tokenizer
        
        self.device = device
        # self.text_feat_map = text_feat_map

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:#NONE
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = self.tokenizer(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix
        

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized
        # self.ctx = torch.autograd.Variable(ctx_vectors, requires_grad=True)

        
# new_text_prompts=['X X X X X X X X X X X X X X X X person.. X X X X X X X X X X X X X X X X rider.. X X X X X X X X X X X X X X X X car.. X X X X X X X X X X X X X X X X truck.. X X X X X X X X X X X X X X X X bus.. X X X X X X X X X X X X X X X X motorcycle.. X X X X X X X X X X X X X X X X bicycle.. ', 'X X X X X X X X X X X X X X X X person.. X X X X X X X X X X X X X X X X rider.. X X X X X X X X X X X X X X X X car.. X X X X X X X X X X X X X X X X truck.. X X X X X X X X X X X X X X X X bus.. X X X X X X X X X X X X X X X X motorcycle.. X X X X X X X X X X X X X X X X bicycle.. ', 'X X X X X X X X X X X X X X X X person.. X X X X X X X X X X X X X X X X rider.. X X X X X X X X X X X X X X X X car.. X X X X X X X X X X X X X X X X truck.. X X X X X X X X X X X X X X X X bus.. X X X X X X X X X X X X X X X X motorcycle.. X X X X X X X X X X X X X X X X bicycle.. ', 'X X X X X X X X X X X X X X X X person.. X X X X X X X X X X X X X X X X rider.. X X X X X X X X X X X X X X X X car.. X X X X X X X X X X X X X X X X truck.. X X X X X X X X X X X X X X X X bus.. X X X X X X X X X X X X X X X X motorcycle.. X X X X X X X X X X X X X X X X bicycle.. ']
        if not self.learned_cls: #true
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(language_model.tokenizer.encode(name)) for name in classnames]
            # prompts = [prompt_prefix + " " + name + "." for name in classnames]
            prompts = [prompt_prefix + " " + name  for name in classnames]
            # self.prompts = prompts
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]
            # self.prompts = prompts

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized
        
        tokenized_prompts = torch.cat([torch.tensor(language_model.tokenizer(p)['input_ids']).to(device) for p in prompts]).to(self.device)
        # breakpoint()
        with torch.no_grad():
            text_dict = language_model(prompts)
            text_dict['embedded'] = text_feat_map(text_dict['embedded'])
            embedding = text_dict['embedded']
            
            # breakpoint()
           

            # embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.ctx_init = ctx_init
        self.prompts = prompts
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames
        # breakpoint()

    def reset(self):
        ctx_vectors = self.ctx_init_state
        # ctx_vectors = torch.autograd.Variable(ctx_vectors, requires_grad=True)
        # breakpoint()
        self.ctx.data.copy_(ctx_vectors.data)
        # self.ctx.copy_(ctx_vectors) # to be optimized

        # breakpoint()
        
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.data.copy_(cls_vectors.data)
            #self.cls.copy_(cls_vectors)

    # def reset_classnames(self, classnames, arch):
    def reset_classnames(self,language_model,text_feat_map):
        self.n_cls = len(self.classnames)
        classnames = self.classnames
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(language_model.tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        # breakpoint()
        tokenized_prompts = torch.cat([torch.Tensor(language_model.tokenizer(p)['input_ids'][0]) for p in prompts]).to(self.device)#TypeError: expected Tensor as element 0 in argument 0, but got BatchEncoding

        # clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            # embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)
            text_dict = language_model(prompts)
            text_dict['embedded'] = text_feat_map(text_dict['embedded'])
            embedding = text_dict['embedded']#torch.Size([7, 20, 256])
        # breakpoint()
        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames
        self.text_dict = text_dict
        # breakpoint()

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:#false
            ctx = init
        else:
            # breakpoint()
            ctx = self.ctx #torch.Size([4, 16, 768]) nn.Parameters()
        if ctx.dim() == 2: #False # ctx.dim()=4
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix #torch.Size([7, 1, 256])
        suffix = self.token_suffix #torch.Size([7, 3, 256])

        # breakpoint()
        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            # prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            # suffix = suffix.repeat(self.batch_size, 1, 1, 1)
            prefix = prefix.repeat(self.batch_size, 1, 1) #torch.Size([28, 1, 256])
            suffix = suffix.repeat(self.batch_size, 1, 1)
            # prefix = prefix.repeat(1,self.batch_size, 1)#torch.Size([7, 4, 256])
            # suffix = suffix.repeat(1,self.batch_size, 1)
            # breakpoint()
            # prefix.reshape([self.batch_size,-1,256]) #prefix.size()=torch.Size([7, 4, 256])
            # suffix.reshape([self.batch_size,-1,256])
            # prompts = torch.cat([prefix.reshape([self.batch_size,-1,768]), ctx,suffix.reshape([self.batch_size,-1,768]), ],dim=-2,)
        if self.learned_cls: #False
            assert self.class_token_position == "end"
        # breakpoint()
        if self.class_token_position == "end": #True
            if self.learned_cls: #False
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim) torch.Size([4, 7, 1, 256])
                        ctx,     # (n_cls, n_ctx, dim) #torch.Size([4, 7, 16, 256])
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                
                # prompts = torch.cat([prefix,ctx,suffix,],dim=-2,)
                # prompts = torch.cat(
                #     [
                #         prefix,  # (n_cls, 1, dim)#torch.Size([4, 7, 1, 256]) #torch.Size([28, 1, 256])
                #         ctx,     # (n_cls, n_ctx, dim) #torch.Size([4, 7, 16, 256])
                #         suffix,  # (n_cls, *, dim)#torch.Size([4, 7, 3, 256])
                #     ],
                #     dim=-2,
                # )
                # breakpoint()
                prompts = torch.cat(
                    [
                        prefix.reshape([self.batch_size,-1,256]),  # (n_cls, 1, dim)#torch.Size([4, 7, 256])
                        ctx.reshape([self.batch_size,-1,256]),     # (n_cls, n_ctx, dim) #torch.Size([4, 112, 256])
                        suffix.reshape([self.batch_size,-1,256]),  # (n_cls, *, dim)#torch.Size([4, 21, 256])
                    ],
                    dim=-2,
                )
                # prompts = torch.cat([prefix.reshape([self.batch_size,-1,768]), ctx,suffix.reshape([self.batch_size,-1,768]), ],dim=-2,)

                # breakpoint()
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts, self.text_dict

@MODELS.register_module()
class GDINOTestTimeTuning(GroundingDINO):
    def __init__(self,
                 classnames,
                 batch_size,
                 language_model,
                 *args,
                 use_autocast=False,
                 device='cuda',
                 criterion='cosine', arch="ViT-L/14",
                 n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False,
                 **kwargs):
        super(GDINOTestTimeTuning, self).__init__(language_model,*args,**kwargs)
        # text modules
        self.language_model_cfg = language_model
        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_encoder = self.language_model 
        # self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)
        # breakpoint()
        # image modules
        #self.image_encoder = self.extract_feat()
        
        #temperature
        # self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.classnames = classnames
        self.device = device
        # self.language_model = MODELS.build(self.language_model_cfg)
        #self.prompt_learner = MODELS.build(self.prompt_learner_cfg)
        self.prompt_learner = PromptLearner(self.language_model,self.text_feat_map, classnames, self.device,batch_size,n_ctx, ctx_init, ctx_position, learned_cls)
        
        self.criterion = criterion
        self.text_prompt = True
        self.image_prompt = True
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, language_model,text_feat_map):
        self.prompt_learner.reset_classnames(language_model,text_feat_map)

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)

    def inference(self, image):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))

        text_features = self.get_text_features()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    # def forward(self, input):
    #     breakpoint()
    #     if isinstance(input, Tuple):
    #         view_0, view_1, view_2 = input
    #         return self.contrast_prompt_tuning(view_0, view_1, view_2)
    #     elif len(input.size()) == 2:
    #         return self.directional_prompt_tuning(input)
    #     else:
    #         return self.inference(input)
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        x = self.backbone(batch_inputs)
        # x1 = self.backbone1(batch_inputs)#augmented feature 1
        # breakpoint()
        if self.with_neck:
            x = self.neck(x)
        return x
    def forward(self, ##base.py
                inputs: torch.Tensor,
                inputs_aug: torch.Tensor,
                inputs_aug2: torch.Tensor,
                inputs_aug3: torch.Tensor,
                inputs_aug4: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        """Returns losses or predictions of training, validation, testing, and
        simple inference process.

        ``forward`` method of BaseModel is an abstract method, its subclasses
        must implement this method.

        Accepts ``batch_inputs`` and ``data_sample`` processed by
        :attr:`data_preprocessor`, and returns results according to mode
        arguments.

        During non-distributed training, validation, and testing process,
        ``forward`` will be called by ``BaseModel.train_step``,
        ``BaseModel.val_step`` and ``BaseModel.test_step`` directly.

        During distributed data parallel training process,
        ``MMSeparateDistributedDataParallel.train_step`` will first call
        ``DistributedDataParallel.forward`` to enable automatic
        gradient synchronization, and then call ``forward`` to get training
        loss.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (list, optional):
                data samples collated by :attr:`data_preprocessor`.
            mode (str): mode should be one of ``loss``, ``predict`` and
                ``tensor``

                - ``loss``: Called by ``train_step`` and return loss ``dict``
                  used for logging
                - ``predict``: Called by ``val_step`` and ``test_step``
                  and return list of results used for computing metric.
                - ``tensor``: Called by custom use to get ``Tensor`` type
                  results.

        Returns:
            dict or list:
                - If ``mode == loss``, return a ``dict`` of loss tensor used
                  for backward and logging.
                - If ``mode == predict``, return a ``list`` of inference
                  results.
                - If ``mode == tensor``, return a tensor or ``tuple`` of tensor
                  or ``dict`` of tensor for custom use.
        """       
        # def forward(self, input):
        # breakpoint()
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        elif mode == 'tpt':
            if inputs_aug2 == []:
                breakpoint()
            return self.loss(inputs,inputs_aug,inputs_aug2,inputs_aug3, inputs_aug4, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor and tpt mode')
        # if isinstance(input, Tuple):
        #     view_0, view_1, view_2 = input
        #     return self.contrast_prompt_tuning(view_0, view_1, view_2)
        # elif len(input.size()) == 2:
        #     return self.directional_prompt_tuning(input)
        # else:
        #     # breakpoint()
        #     return self.inference(inputs)

    # def _forward(
    #         self,
    #         batch_inputs: Tensor,
    #         batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
    #     """Network forward process. Usually includes backbone, neck and head
    #     forward without any post-processing.

    #      Args:
    #         batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
    #         batch_data_samples (List[:obj:`DetDataSample`], optional): The
    #             batch data samples. It usually includes information such
    #             as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
    #             Defaults to None.

    #     Returns:
    #         tuple[Tensor]: A tuple of features from ``bbox_head`` forward.
    #     """
        
    #     img_feats = self.extract_feat(batch_inputs)
    #     head_inputs_dict = self.forward_transformer(img_feats,batch_data_samples)
    #     results = self.bbox_head.forward(**head_inputs_dict)
    #     breakpoint()
    #     return results    
# @MODELS.register_module()
# class GroundingDINOTPT(GroundingDINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    # def __init__(self,
    #              language_model,
    #              *args,
    #              use_autocast=False,
    #              **kwargs) -> None:

    #     self.language_model_cfg = language_model
    #     self._special_tokens = '. '
    #     self.use_autocast = use_autocast
    #     super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # # text modules
        # self.language_model = MODELS.build(self.language_model_cfg)
        # self.text_feat_map = nn.Linear(
        #     self.language_model.language_backbone.body.language_dim,
        #     self.embed_dims,
        #     bias=True)

    ###based on mmengine/mmengine/model/base_model/base_model.py
    def train_tpt_step(self, data: Union[dict, tuple, list],
                        data_aug: Union[dict, tuple, list],
                        data_aug2: Union[dict, tuple, list],
                        data_aug3: Union[dict, tuple, list],
                        data_aug4: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        # breakpoint()
        trainable_param = self.prompt_learner.parameters()
        # breakpoint()
        optimizer = torch.optim.AdamW(trainable_param, lr=0.0004)
        optim_wrapper = OptimWrapper(optimizer)
        with optim_wrapper.optim_context(self):
            
            data = self.data_preprocessor(data, True)
            data_aug = self.data_preprocessor(data_aug, True)
            data_aug2 = self.data_preprocessor(data_aug2, True)
            data_aug3 = self.data_preprocessor(data_aug3, True)
            data_aug4 = self.data_preprocessor(data_aug4, True)
            losses = self._run_forward(data,data_aug,data_aug2,data_aug3,data_aug4, mode='tpt')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        # parsed_losses = torch.autograd.Variable(parsed_losses, requires_grad = True)
        # breakpoint()
        # for parameter in self.prompt_learner(): print(parameter.requires_grad)
        optim_wrapper.update_params(parsed_losses) #update model.
        
        return log_vars

    def _run_forward(self, data: Union[dict, tuple, list],
                        data_aug: Union[dict, tuple, list],
                        data_aug2: Union[dict, tuple, list],
                        data_aug3: Union[dict, tuple, list],
                        data_aug4: Union[dict, tuple, list],
                        mode: str) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            # breakpoint()
            data['inputs_aug']= data_aug['inputs']
            data['inputs_aug2']= data_aug2['inputs']
            data['inputs_aug3']= data_aug3['inputs']
            data['inputs_aug4']= data_aug4['inputs']
            results = self(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)

    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]
                if 'prefix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['prefix']
                start_i = len(caption_string)
                if 'name' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['name']
                else:
                    caption_string += word
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                if 'suffix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['suffix']
            else:
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def to_plain_text_prompts(self, original_caption):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string),
                  len(caption_string) + len(word)]])
            caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def to_prompted_text_prompts(self, original_caption): #based on self.to_plain_text_prompts)()
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string),
                  len(caption_string) + len(word)]])
            caption_string += word
            caption_string += self._special_tokens
        # breakpoint()
        return caption_string, tokens_positive
    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None
    ) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities: #True
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            original_caption = [clean_label_name(i) for i in original_caption]

            if custom_entities and enhanced_text_prompts is not None: #false
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption)
            if self.text_prompt:
                caption_string, tokens_positive = self.to_prompted_text_prompts(
                    original_caption)


            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                breakpoint()
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption
        # breakpoint()
        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=self.bbox_head.cls_branches[
                self.decoder.num_layers].max_text_len)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompt: Optional[ConfigType] = None,
        tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        """
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                positive_map_label_to_token, positive_map = \
                    self.get_positive_map(tokenized, tokens_positive)

                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0]:t[1]])
                    entities.append(' / '.join(instance_entities))
                return positive_map_label_to_token, original_caption, \
                    positive_map, entities

        chunked_size = self.test_cfg.get('chunked_size', -1)
        if not self.training and chunked_size > 0:
            assert isinstance(original_caption,
                              (list, tuple)) or custom_entities is True
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt)
            positive_map_label_to_token, \
                caption_string, \
                positive_map, \
                entities = all_output
        else:
            tokenized, caption_string, tokens_positive, entities = \
                self.get_tokens_and_prompts(
                    original_caption, custom_entities, enhanced_text_prompt)
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)
        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
            self,
            original_caption: Union[list, tuple],
            enhanced_text_prompts: Optional[ConfigType] = None):
        chunked_size = self.test_cfg.get('chunked_size', -1)
        original_caption = [clean_label_name(i) for i in original_caption]

        original_caption_chunked = chunks(original_caption, chunked_size)
        ids_chunked = chunks(
            list(range(1,
                       len(original_caption) + 1)), chunked_size)

        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        for i in range(len(ids_chunked)):
            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption_chunked[i], enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption_chunked[i])
            tokenized = self.language_model.tokenizer([caption_string],
                                                      return_tensors='pt')
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the --chunked-size.')
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)

            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(
                positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(original_caption_chunked[i])

        return positive_map_label_to_token_chunked, \
            caption_string_chunked, \
            positive_map_chunked, \
            entities_chunked

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        # breakpoint()
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict, text_dict=text_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        # breakpoint()
        return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        # breakpoint()
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict
    
    def loss(self, batch_inputs: Tensor,
                batch_inputs_aug: Tensor,
                batch_inputs_aug2: Tensor,
                batch_inputs_aug3: Tensor,
                batch_inputs_aug4: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]
        # breakpoint()
        prompts = self.prompt_learner.prompts
        # tokenized_prompts = self.prompt_learner.tokenized_prompts
        # breakpoint()
        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        if 'tokens_positive' in batch_data_samples[0]: #False
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                # tokenized, caption_string, tokens_positive, _ = \
                #     self.get_tokens_and_prompts(
                #         text_prompts[0], True)
                tokenized, caption_string, tokens_positive, _ = self.get_tokens_and_prompts(prompts, True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)
        ##new text promts: get require_grad
        # def get_text_features(self):
        # text_features = []
        # prompts = self.prompt_learner()
        # tokenized_prompts = self.prompt_learner.tokenized_prompts
        # t_features = self.language_model(tokenized_prompts)
        # text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        # text_features = torch.stack(text_features, dim=0)

        # return torch.mean(text_features, dim=0)
        tokenized_prompts = torch.cat([torch.tensor(self.language_model.tokenizer(p)['input_ids']).to(self.device) for p in new_text_prompts]).to(self.device)

        
        text_dict = self.language_model(new_text_prompts)
        # text_dict.keys()=dict_keys(['embedded', 'masks', 'hidden', 'position_ids', 'text_token_mask'])
        # text_dict['embedded'].size()=torch.Size([4, 128, 768])
        # text_dict['masks'].size()=torch.Size([4, 128, 128])
        # text_dict['hidden'].size()=torch.Size([4, 128, 768])
        # text_dict[ 'position_ids'].size()=torch.Size([4, 128])
        # text_dict['text_token_mask'].size()=torch.Size([4, 128])
        

        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])
        # ###########add tpt tuning prompt parameters: ctx to [new_text_prompts]########
        #step 1 self.prompt_learner.reset()
        ctx_vectors = self.prompt_learner.ctx_init_state
        self.prompt_learner.ctx.data.copy_(ctx_vectors.data)
        if self.prompt_learner.learned_cls:
            cls_vectors = self.prompt_learner.cls_init_state
            self.cls.copy_(cls_vectors)
        #step 2 self.prompt_learner.reset_classnames
        self.prompt_learner.n_cls = len(self.classnames)
        # if not self.prompt_learner.learned_cls:
        #     self.classnames = [name.replace("_", " ") for name in self.classnames]
        #     name_lens = [len(self.language_model.tokenizer.encode(name)) for name in self.classnames]
        #     prompts = [self.prompt_learner.prompt_prefix + " " + name + "." for name in self.classnames]
        # breakpoint()
        embedding = text_dict['embedded']#torch.Size([4, 128, 256])
        self.token_prefix = embedding[:, :1, :]#torch.Size([4, 1, 256])
        # self.prompt_learner.n_ctx=16
        self.token_suffix = embedding[:, 1 + self.prompt_learner.n_ctx :, :]#torch.Size([4, 111, 256])
        # self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        #step 3 forward
        ctx = self.prompt_learner.ctx#torch.Size([4, 16, 256])
        # if ctx.dim() == 2:
        #     ctx = ctx.unsqueeze(0).expand(self.prompt_learner.n_cls, -1, -1)
        # elif not ctx.size()[0] == self.prompt_learner.n_cls:
        #     ctx = ctx.unsqueeze(1).expand(-1, self.prompt_learner.n_cls, -1, -1)
        prefix = self.token_prefix ##prefix.size()=torch.Size([4, 1, 256])
        suffix = self.token_suffix ##suffix.size()=torch.Size([4, 111, 256])
        # if self.prompt_learner.batch_size is not None: 
        #     # This way only works for single-gpu setting (could pass batch size as an argument for forward())
        #     prefix = prefix.repeat(self.prompt_learner.batch_size, 1, 1, 1)
        #     suffix = suffix.repeat(self.prompt_learner.batch_size, 1, 1, 1)
        if self.prompt_learner.learned_cls:
            assert self.class_token_position == "end"
        try:
            prompts = torch.cat(
                    [
                        self.token_prefix,  # (n_cls, 1, dim)##torch.Size([4, 1, 256])
                        ctx,     # (n_cls, n_ctx, dim)##torch.Size([4, 16, 256])
                        self.token_suffix,  # (n_cls, *, dim)##torch.Size([4, 111, 256])
                    ],
                    dim=-2,
                )
        except:
            prompts = torch.cat(
                    [
                        self.token_prefix,  # (n_cls, 1, dim)##torch.Size([4, 1, 256])
                        ctx[0].reshape([-1,16,256]),     # (n_cls, n_ctx, dim)##torch.Size([4, 16, 256])
                        self.token_suffix,  # (n_cls, *, dim)##torch.Size([4, 111, 256])
                    ],
                    dim=-2,
                )
            # breakpoint()
        # prompts = torch.cat(
        #             [
        #                 prefix.reshape([self.prompt_learner.batch_size,-1,256]),  # (n_cls, 1, dim)##torch.Size([4, 4, 1, 256])
        #                 ##prefix.reshape([self.prompt_learner.batch_size,-1,256]).size()=torch.Size([4, 4, 256])
        #                 ctx.reshape([self.prompt_learner.batch_size,-1,256]),     # (n_cls, n_ctx, dim)##torch.Size([4, 7, 16, 256])
        #                 suffix.reshape([self.prompt_learner.batch_size,-1,256]),  # (n_cls, *, dim)##torch.Size([4, 4, 111, 256])
        #             ],
        #             dim=-2,
        #         )###torch.Size([4, 560, 256])
        # breakpoint()


        text_dict['embedded']=prompts
        #new_text_prompts=['X X X X X X X X X X X X X X X X person. X X X X X X X X X X X X X X X X rider. X X X X X X X X X X X X X X X X car. X X X X X X X X X X X X X X X X truck. X X X X X X X X X X X X X X X X bus. X X X X X X X X X X X X X X X X motorcycle. X X X X X X X X X X X X X X X X bicycle. ', 'X X X X X X X X X X X X X X X X person. X X X X X X X X X X X X X X X X rider. X X X X X X X X X X X X X X X X car. X X X X X X X X X X X X X X X X truck. X X X X X X X X X X X X X X X X bus. X X X X X X X X X X X X X X X X motorcycle. X X X X X X X X X X X X X X X X bicycle. ', 'X X X X X X X X X X X X X X X X person. X X X X X X X X X X X X X X X X rider. X X X X X X X X X X X X X X X X car. X X X X X X X X X X X X X X X X truck. X X X X X X X X X X X X X X X X bus. X X X X X X X X X X X X X X X X motorcycle. X X X X X X X X X X X X X X X X bicycle. ', 'X X X X X X X X X X X X X X X X person. X X X X X X X X X X X X X X X X rider. X X X X X X X X X X X X X X X X car. X X X X X X X X X X X X X X X X truck. X X X X X X X X X X X X X X X X bus. X X X X X X X X X X X X X X X X motorcycle. X X X X X X X X X X X X X X X X bicycle. ']
        #text_dict = self.language_model(new_text_prompts)
        # text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])
        
        #embedding = text_dict['embedded']#torch.Size([4, 128, 256])
        # self.token_prefix = embedding[:, :1, :]
        # self.token_suffix = embedding[:, 1 + self.prompt_learner.n_ctx :, :]
        # self.prompt_learner.ctx=nn.Parameter(ctx_vectors)
        #prompts = torch.cat([self.token_prefix,self.prompt_learner.ctx,self.token_suffix])#torch.Size([4, 560, 256])
        #prompts = text_dict['embedded']
        
        #text_dict.keys()=dict_keys(['embedded', 'masks', 'hidden', 'position_ids', 'text_token_mask'])
        #text_dict['embedded']=?#torch.Size([4, 128, 256])
        #text_dict['masks']=?#torch.Size([4, 128, 128])
        #text_dict['hidden']=?#torch.Size([4, 128, 768])
        #text_dict['position_ids']=?#torch.Size([4, 128])
        #text_dict['text_token_mask']=?#torch.Size([4, 128])
        
        # self.prompt_learner.reset()
        # self.prompt_learner.reset_classnames(self.language_model, self.text_feat_map)
        

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)
            visual_features_aug = self.extract_feat(batch_inputs_aug)
            visual_features_aug2 = self.extract_feat(batch_inputs_aug2)
            visual_features_aug3 = self.extract_feat(batch_inputs_aug3)
            visual_features_aug4 = self.extract_feat(batch_inputs_aug4)
        # breakpoint()
        # text_dict['hidden'].size()=torch.Size([4, 135, 768])
        # text_dict['embedded'].size()=torch.Size([4, 140, 256])
        head_inputs_dict = self.forward_transformer(visual_features, text_dict,batch_data_samples)
        head_inputs_dict_aug = self.forward_transformer(visual_features_aug, text_dict,batch_data_samples)
        head_inputs_dict_aug2 = self.forward_transformer(visual_features_aug2, text_dict,batch_data_samples)
        head_inputs_dict_aug3 = self.forward_transformer(visual_features_aug3, text_dict,batch_data_samples)
        head_inputs_dict_aug4 = self.forward_transformer(visual_features_aug4, text_dict,batch_data_samples)
        head_inputs_dict['enc_outputs_class_aug'] = head_inputs_dict_aug['enc_outputs_class']
        head_inputs_dict['enc_outputs_coord_aug'] = head_inputs_dict_aug['enc_outputs_coord']
        head_inputs_dict['enc_outputs_class_aug2'] = head_inputs_dict_aug2['enc_outputs_class']
        head_inputs_dict['enc_outputs_coord_aug2'] = head_inputs_dict_aug2['enc_outputs_coord']
        head_inputs_dict['enc_outputs_class_aug3'] = head_inputs_dict_aug3['enc_outputs_class']
        head_inputs_dict['enc_outputs_coord_aug3'] = head_inputs_dict_aug3['enc_outputs_coord']
        head_inputs_dict['enc_outputs_class_aug4'] = head_inputs_dict_aug4['enc_outputs_class']
        head_inputs_dict['enc_outputs_coord_aug4'] = head_inputs_dict_aug4['enc_outputs_coord']
        # breakpoint()
        if head_inputs_dict['enc_outputs_class_aug2'] == []:
            breakpoint()
        losses = self.bbox_head.loss(**head_inputs_dict, batch_data_samples=batch_data_samples)
        # losses = self.avg_entropy(**head_inputs_dict)
        # breakpoint()
        return losses
# loss(self, batch_inputs: Tensor,
#              batch_data_samples: SampleList) -> Union[dict, list]:
    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        self.training=False
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        ###aug visual_feats 

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            # breakpoint()
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)
            # breakpoint()

        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples