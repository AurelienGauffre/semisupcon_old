# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import wandb
from .hook import Hook


class WANDBHook(Hook):
    """
    Wandb Hook
    """

    def __init__(self):
        super().__init__()
        self.log_key_list = ['train/sup_loss', 'train/unsup_loss', 'train/total_loss', 'train/util_ratio', 
                             'train/run_time', 'train/prefetch_time', 'lr',
                             'eval/top-1-acc', 'eval/precision', 'eval/recall', 'eval/F1',
                             'train/pseudolabel_accuracy'
                             ]

    def before_run(self, algorithm):
        # job_id = '_'.join(algorithm.args.save_name.split('_')[:-1])
        name = algorithm.save_name
        project = algorithm.args.wandb_project #algorithm.save_dir.split('/')[-1]

        # tags
        benchmark = f'benchmark: {project}'
        dataset = f'dataset: {algorithm.args.dataset}'
        data_setting = f'setting: {algorithm.args.dataset}_lb{algorithm.args.num_labels}_{algorithm.args.lb_imb_ratio}_ulb{algorithm.args.ulb_num_labels}_{algorithm.args.ulb_imb_ratio}'
        alg = f'alg: {algorithm.args.algorithm}'
        imb_alg = f'imb_alg: {algorithm.args.imb_algorithm}'
        tags = [benchmark, dataset, data_setting, alg, imb_alg] 
        if algorithm.args.resume:
            resume = 'auto'
        else:
            resume = 'never'
        # resume = 'never'
        
        save_dir = os.path.join(algorithm.args.save_dir, 'wandb', algorithm.args.save_name_wandb)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if 'WANDB_MODE' in os.environ:
            wandb_mode = os.environ['WANDB_MODE']
        
        else :
            wandb_mode = "online"

        self.run = wandb.init(name=name, 
                              tags=tags, 
                              config=algorithm.args.__dict__, 
                              project=project, 
                              resume=resume,
                              dir=save_dir,
                              mode=wandb_mode) #PERSO Added support for offline mode


    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_log_iter):
            log_dict = {}
            # for key, item in algorithm.log_dict.items():
            #     if key in self.log_key_list:
            #         log_dict[key] = item
            for key, item in algorithm.log_dict.items():
               #J'enleve la condition de logging d'appartenance a log_key_list ci dessus
               log_dict[key] = item
            self.run.log(log_dict, step=algorithm.it)
    
        if self.every_n_iters(algorithm, algorithm.num_eval_iter):
            self.run.log({'eval/best-acc': algorithm.best_eval_acc}, step=algorithm.it)
    
    def after_run(self, algorithm):
        self.run.finish()