import os
from omegaconf import OmegaConf
import wandb

def upload_metrics(file_path):
    data = OmegaConf.load(file_path)
    # Initialize a wandb run with the experiment name
    wandb.init(project=data.params.wandb_project, name=data.params.experiment_name)
    # Log hyperparameters
    wandb.config.update(data.params)
    # Log metrics
    for metric in data.metrics:
        wandb.log(metric)
    # Mark run as finished
    wandb.finish()

def main(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('finished.yaml'):
            file_path = os.path.join(directory_path, filename)
            upload_metrics(file_path)
            # Rename the file to indicate it has been uploaded
            new_file_path = file_path.replace('finished.yaml', 'uploaded.yaml')
            os.rename(file_path, new_file_path)

if __name__ == "__main__":
    directory_path = './save/'
    main(directory_path)
