import os
import shutil

# Define the function to copy files
def copy_config_files(current_dir):
    # List all files in the current directory
    files = os.listdir(current_dir)

    # Loop through each file in the directory
    for file in files:
        # Check if the file starts with "configD" and is a YAML file
        if file.startswith("configEcifar100-400") and file.endswith(".yaml"):

        # Construct the new file name by replacing "configD" with "configDflex"
        #new_file = file.replace("configD", "configEcifar100-400-")

            new_file = file[:-19] + 'stl-25' + file[-7:]
            # Copy the file to the new file with updated name
            shutil.copyfile(os.path.join(current_dir, file), os.path.join(current_dir, new_file))

# Call the function with the current directory
print(os.getcwd())
copy_config_files(os.getcwd())