import kagglehub

# Download latest version
path = kagglehub.dataset_download("pavansanagapati/images-dataset")
print("Path to dataset files:", path)