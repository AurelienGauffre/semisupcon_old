# Upload metrics to the server on JZ
# Change directory to 'save'
cd save

# Load the pytorch-gpu module
module load pytorch-gpu/py3/1.13.0

# Execute the Python script with the provided argument
python3 upload.py "$1"