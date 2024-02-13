import os
import sys
from omegaconf import OmegaConf
import wandb
import time

os.environ['WANDB_MODE'] = 'online'


def upload_metrics(file_path, mode='finished'):
    data = OmegaConf.load(file_path)
    if mode == 'finished':
        wandb.init(project=data.params.wandb_project, name=data.params.save_name_wandb,
                   config=OmegaConf.to_container(data.params))
    else:
        wandb.init(project=data.params.wandb_project, name=f"TEMP_{data.params.save_name_wandb}",
                   config=OmegaConf.to_container(data.params))

    for metric in data.logged_metrics:
        wandb.log(metric, step=metric['Step'])
    wandb.finish()


def should_upload(filename, mode, directory_path):
    file_path = os.path.join(directory_path, filename)
    if mode == 'recent':
        modification_time = os.path.getmtime(file_path)
        current_time = time.time()
        one_hour_ago = current_time - 3600
        return not filename.endswith(('finished.yaml', 'uploaded.yaml')) and modification_time > one_hour_ago
    elif mode == 'all':
        return not filename.endswith(('finished.yaml', 'uploaded.yaml'))
    else:  # 'finished' mode
        return filename.endswith('finished.yaml')


def main(directory_path, mode='finished'):
    for filename in os.listdir(directory_path):
        print(filename)
        if should_upload(filename, mode, directory_path):
            file_path = os.path.join(directory_path, filename)
            upload_metrics(file_path, mode)
            if mode == 'finished':
                new_file_path = file_path.replace('finished.yaml', 'uploaded.yaml')
                os.rename(file_path, new_file_path)


if __name__ == "__main__":
    directory_path = './'
    mode = sys.argv[1] if len(sys.argv) > 1 else 'finished'
    main(directory_path, mode)

# import os
# from omegaconf import OmegaConf
# import wandb
#
# os.environ['WANDB_MODE'] = 'online'
#
# def upload_metrics(file_path):
#     data = OmegaConf.load(file_path)
#     # Initialize a wandb run with the experiment name
#     print(data.params)
#     print(type(data.params))
#     wandb.init(project=data.params.wandb_project, name=data.params.save_name_wandb,config=OmegaConf.to_container(data.params))
#     # Log hyperparameters
#
#     # Log metrics
#     for metric in data.logged_metrics:
#         wandb.log(metric,step = metric['Step'])
#     # Mark run as finished
#     wandb.finish()
#
# def main(directory_path):
#     for filename in os.listdir(directory_path):
#         if filename.endswith('finished.yaml'):
#             file_path = os.path.join(directory_path, filename)
#             upload_metrics(file_path)
#             # Rename the file to indicate it has been uploaded
#             new_file_path = file_path.replace('finished.yaml', 'uploaded.yaml')
#             os.rename(file_path, new_file_path)
#
# if __name__ == "__main__":
#     directory_path = './'
#     main(directory_path)
