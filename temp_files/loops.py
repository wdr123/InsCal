# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import is_model_wrapper
from mmengine.runner import ValLoop
from mmengine.runner import EpochBasedTrainLoop
from mmdet.registry import LOOPS
import torch
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmengine.optim import OptimWrapper
# from typing import Any, Dict, Union
# from mmengine.utils import calc_dynamic_intervals
from torch.utils.data import DataLoader
from itertools import cycle

@LOOPS.register_module()
class TeacherStudentValLoop(ValLoop):
    """Loop for validation of model teacher and student."""

    def run(self):
        """Launch validation for model teacher and student."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        model = self.runner.model
        if is_model_wrapper(model):
            model = model.module
        assert hasattr(model, 'teacher')
        assert hasattr(model, 'student')

        predict_on = model.semi_test_cfg.get('predict_on', None)
        multi_metrics = dict()
        for _predict_on in ['teacher', 'student']:
            model.semi_test_cfg['predict_on'] = _predict_on
            
            for idx, data_batch in enumerate(self.dataloader):
                
                self.run_iter(idx, data_batch)
            # compute metrics
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            multi_metrics.update(
                {'/'.join((_predict_on, k)): v
                 for k, v in metrics.items()})
        model.semi_test_cfg['predict_on'] = predict_on

        self.runner.call_hook('after_val_epoch', metrics=multi_metrics)
        self.runner.call_hook('after_val')

@LOOPS.register_module()
class TPTLoop(EpochBasedTrainLoop):
    def __init__(self, runner, 
            dataloader: Union[DataLoader, Dict],
            aug_dataloader: Union[DataLoader, Dict],
            aug_dataloader2: Union[DataLoader, Dict],
            aug_dataloader3: Union[DataLoader, Dict],
            aug_dataloader4: Union[DataLoader, Dict],
            max_epochs: int,
            idx_list: Optional[List[Tuple[int, int]]] = None,
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        
        # super().__init__(runner, dataloader,aug_dataloader,max_epochs,idx_list)
        
        # breakpoint()
        self._runner = runner
        # SEED = runner.seed
        self._dataloader = dataloader
        self._aug_dataloader = aug_dataloader
        self._aug_dataloader2 = aug_dataloader2
        self._aug_dataloader3 = aug_dataloader3
        self._aug_dataloader4 = aug_dataloader4
        if isinstance(dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            # breakpoint()
            self.dataloader = runner.build_dataloader(dataloader,[], seed=runner.seed, diff_rank_seed=diff_rank_seed)
            
            # self.dataloader = runner.build_dataloader(dataloader, aug_dataloader,seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.dataloader = dataloader

        
        # ##EpochBasedLoops
        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)

        # self.dynamic_milestones, self.dynamic_intervals = \
        #     calc_dynamic_intervals(
        #         self.val_interval, dynamic_intervals)

    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')
        # names= []
        for name, param in self.runner.model.named_parameters():
            # names.append(name)
            if "prompt_learner" not in name: #'prompt_learner.ctx'
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
                # breakpoint()
        # names_set.update(set(group['params']))
        # breakpoint()
        # for parameter in self.runner.model.parameters(): 
        #     print(parameter.requires_grad)
        # breakpoint()
        # optim_state = deepcopy(optimizer.state_dict())
        self.runner.model.reset_classnames(self.runner.model.language_model, self.runner.model.text_feat_map)

        
        # else:
        #     if "text_encoder" not in name:
        #         param.requires_grad_(False)

        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and (self._epoch % self.val_interval == 0
                         or self._epoch == self._max_epochs)):
                self.runner.val_loop.run()
            # breakpoint()

        self.runner.call_hook('after_train')
        return self.runner.model
    # def run(self) -> torch.nn.Module:
    #     """Launch test-time prompt-tuning for model prompt learner."""
    #     self.runner.call_hook('before_train')
        

    #     while self._epoch < self._max_epochs and not self.stop_training:
    #         # breakpoint() #breaked
    #         self.run_epoch()
            
    #         # breakpoint()
    #         self._decide_current_val_interval()
            
    #         if (self.runner.val_loop is not None
    #                 and self._epoch >= self.val_begin
    #                 and self._epoch % self.val_interval == 0):
                
    #             self.runner.val_loop.run()
    #     #         breakpoint()
    #     #     breakpoint()
    #     # breakpoint()#not breaking
    #     self.runner.call_hook('after_train')
    #     return self.runner.model
    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        # for idx, data_batch in enumerate(self.dataloader):
        #     # breakpoint()
        #     self.run_iter(idx, data_batch)
        # breakpoint()
        for idx, data_batch in enumerate(self.dataloader):
            idx_list = [data_batch[ 'data_samples'][x].img_id for x in range(4)]
            # breakpoint()

            
            # data_batch[ 'data_samples'][0].img_path
            # data_batch_aug[ 'data_samples'][0].img_path
            # data_batch[ 'data_samples'][0].img_id
            # data_batch_aug[ 'data_samples'][0].img_id
            if isinstance(self._aug_dataloader, dict):
        #     # Determine whether or not different ranks use different seed.
                diff_rank_seed = self.runner._randomness_cfg.get('diff_rank_seed', False)
                # 
                # self.aug_dataloader = self.runner.build_dataloader(self._aug_dataloader,idx_list, seed=self.runner.seed, diff_rank_seed=diff_rank_seed)
            
        #     # runner.seed=1714878002
                self.aug_dataloader = self.runner.build_dataloader(
                    self._aug_dataloader, idx_list, seed=self.runner.seed, diff_rank_seed=diff_rank_seed)
                self.aug_dataloader2 = self.runner.build_dataloader(
                    self._aug_dataloader2, idx_list, seed=self.runner.seed, diff_rank_seed=diff_rank_seed)
                self.aug_dataloader3 = self.runner.build_dataloader(
                    self._aug_dataloader3, idx_list, seed=self.runner.seed, diff_rank_seed=diff_rank_seed)
                self.aug_dataloader4 = self.runner.build_dataloader(
                    self._aug_dataloader4, idx_list, seed=self.runner.seed, diff_rank_seed=diff_rank_seed)
                data_batch_aug = list(enumerate(self.aug_dataloader))[0]
                data_batch_aug2 = list(enumerate(self.aug_dataloader2))[0]
                data_batch_aug3 = list(enumerate(self.aug_dataloader3))[0]
                data_batch_aug4 = list(enumerate(self.aug_dataloader4))[0]
                # img_list = [data_batch['data_samples'][x].img_path for x in range(4)]
                # img_id = [data_batch['data_samples'][x].img_id for x in range(4)]
                # aug_img_id = [data_batch_aug[1]['data_samples'][x].img_id for x in range(4)]
                # aug_img_list=[data_batch_aug[1][ 'data_samples'][x].img_path for x in range(4)]

                # breakpoint()
                
        # else:
        #     self.aug_dataloader = aug_dataloader
        
        # if isinstance(aug_dataloader2, dict):
        #     # Determine whether or not different ranks use different seed.
        #     # diff_rank_seed = runner._randomness_cfg.get(
        #     #     'diff_rank_seed', False)
        #     self.aug_dataloader2 = runner.build_dataloader(
        #         aug_dataloader2, seed=SEED, diff_rank_seed=diff_rank_seed)
        # else:
        #     self.aug_dataloader2 = _aug_dataloader2
        
        # if isinstance(aug_dataloader3, dict):
        #     # Determine whether or not different ranks use different seed.
        #     # diff_rank_seed = runner._randomness_cfg.get(
        #     #     'diff_rank_seed', False)
        #     self.aug_dataloader3 = runner.build_dataloader(
        #         aug_dataloader3, seed=SEED, diff_rank_seed=diff_rank_seed)
        # else:
        #     self.aug_dataloader3 = aug_dataloader3
        
        # if isinstance(aug_dataloader4, dict):
        #     # # Determine whether or not different ranks use different seed.
        #     # diff_rank_seed = runner._randomness_cfg.get(
        #     #     'diff_rank_seed', False)
        #     self.aug_dataloader4 = runner.build_dataloader(
        #         aug_dataloader4, seed=SEED, diff_rank_seed=diff_rank_seed)
        # else:
        #     self.aug_dataloader4 = aug_dataloader4
            self.run_iter(idx, data_batch,data_batch_aug[1],data_batch_aug2[1],data_batch_aug3[1],data_batch_aug4[1])
        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, 
                    data_batch: Sequence[dict],
                    data_batch_aug: Sequence[dict],
                    data_batch_aug2: Sequence[dict],
                    data_batch_aug3: Sequence[dict],
                    data_batch_aug4: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_tpt_step(
            data_batch, 
            data_batch_aug,
            data_batch_aug2,
            data_batch_aug3,
            data_batch_aug4,
            optim_wrapper=self.runner.optim_wrapper)
        # breakpoint()

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1
    
    # @property
    # def max_iters(self):
    #     """int: Total iterations to train model."""
    #     return self._max_iters
    
    # @property
    # def iter(self):
    #     """int: Current iteration."""
    #     return self._iter
    
   
    
    


