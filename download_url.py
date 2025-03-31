import zipfile

# Specify the path to your zip file and extraction directory
with zipfile.ZipFile('datasetdownloaded_file.zip', 'r') as zip_ref:
    zip_ref.extractall('raw')  # Omit the directory to extract to current folder