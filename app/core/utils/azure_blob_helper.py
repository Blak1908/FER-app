import os
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
from app.core.settings import get_settings

settings = get_settings()
connection_string = settings.CONNECTION_STRING
connection_string_speaker = settings.CONNECTION_STRING_SPEAKER
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
blob_service_client.max_single_put_size = 16*1024*1024

connection_string_model = settings.CONNECTION_STRING_MODEL

def download_blob(domain, url, destination_file):
    domain_container_name = f"{domain}-speaker-video"
    print('domain_container_name: ',domain_container_name)
    container_client = blob_service_client.get_container_client(domain_container_name)
    file_name = url.split(f'/{domain_container_name}/')[-1]
    blob_client = container_client.get_blob_client(file_name)
    exists = blob_client.exists()
    print(f"{url} exists: {exists}")
    if exists:
        with open(destination_file, "wb") as my_blob:
            blob_data = blob_client.download_blob()
            blob_data.readinto(my_blob)
        return destination_file
    return 404

def download_blob_bg(domain_container_name, url, destination_file):
    print('Download background: ', url)
    print('domain_container_name: ',domain_container_name)
    container_client = blob_service_client.get_container_client(domain_container_name)
    file_name = url.split(f'/{domain_container_name}/')[-1]
    blob_client = container_client.get_blob_client(file_name)
    exists = blob_client.exists()
    print(f"{url} exists: {exists}")
    if exists:
        with open(destination_file, "wb") as my_blob:
            blob_data = blob_client.download_blob()
            blob_data.readinto(my_blob)
        return destination_file
    return 404
    

def upload_blob(domain, local_file_name):
    print(f"Upload blob: {local_file_name}")
    domain_container_name = f"{domain}-speaker-video"
    
    blob_file_name = os.path.basename(local_file_name)
    blob_client = blob_service_client.get_blob_client(container=domain_container_name, blob=blob_file_name)
    with open(local_file_name, "rb") as data:
        # print('data: ', data)
        # data = data.encode('utf-8')
        blob_client.upload_blob(data, overwrite=True)
    url_file = blob_client.url
    print('url_file: ', url_file)
    return url_file

def download_blob_image(domain, url, destination_file):
    domain_container_name = f"{domain}-pictures"
    print('domain_container_name: ',domain_container_name)
    container_client = blob_service_client.get_container_client(domain_container_name)
    file_name = url.split(f'/{domain_container_name}/')[-1]
    blob_client = container_client.get_blob_client(file_name)
    exists = blob_client.exists()
    print(f"Blob file: {file_name} exists: {exists}")
    if exists:
        with open(destination_file, "wb") as my_blob:
            blob_data = blob_client.download_blob()
            blob_data.readinto(my_blob)
        return destination_file
    return 404

def upload_blob_image(domain, local_file_name):
    domain_container_name = f"{domain}-pictures"
    print("local_file_name: ", local_file_name)
    
    blob_file_name = os.path.basename(local_file_name)
    print("blob_file_name: ", blob_file_name)
    blob_client = blob_service_client.get_blob_client(container=domain_container_name, blob=blob_file_name)
    with open(local_file_name, "rb") as data:
        # print('data: ', data)
        # data = data.encode('utf-8')
        blob_client.upload_blob(data, overwrite=True)
    url_file = blob_client.url
    print('url_file: ', url_file)
    return url_file

def upload_folder_blob(domain, local_folder):
    domain_container_name = f"{domain}-speaker-video"

    folder_blob = os.path.basename(local_folder)
    files = os.listdir(local_folder)

    for f in files:
        local_file_name = os.path.join(local_folder, f)
        blob_file_name = os.path.join(folder_blob, f)
        blob_client = blob_service_client.get_blob_client(container=domain_container_name, blob=blob_file_name)
        with open(local_file_name, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        folder_url = blob_client.url
        print(blob_client.url)
    return os.path.dirname(folder_url)

def delete_blobs(domain_container_name, folder_blob):
    
    folder_blob_client = blob_service_client.get_container_client(domain_container_name)
    folder_blobs = [blob.name for blob in folder_blob_client.list_blobs() if blob.name.startswith(folder_blob)]
    
    # If the folder exists, delete all the blobs in the folder
    if folder_blobs:
        for blob_name in folder_blobs:
            blob_client = folder_blob_client.get_blob_client(blob_name)
            blob_client.delete_blob()
            print("Delete blob: ",blob_name)

def upload_speaker(local_folder):
    domain_container_name = f"speakers"

    folder_blob = os.path.basename(local_folder)
    files = os.listdir(local_folder)
    
    # Reset blob folder 
    delete_blobs(domain_container_name, folder_blob)
    
    for f in files:
        local_file_name = os.path.join(local_folder, f)
        blob_file_name = os.path.join(folder_blob, f)
        blob_client = blob_service_client.get_blob_client(container=domain_container_name, blob=blob_file_name)
        with open(local_file_name, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        folder_url = blob_client.url
        print(blob_client.url)
    return os.path.dirname(folder_url)
        
def download_folder_blob(domain, url, path_temp):
    domain_container_name = f"{domain}-speaker-video"
    base_path = os.path.basename(url)
    container_client = blob_service_client.get_container_client(domain_container_name)
    temp_folder = f'{path_temp}/{base_path}'
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)

    blob_names = container_client.list_blobs(name_starts_with=base_path)
    for blob in blob_names:
        blob_client = container_client.get_blob_client(blob.name)
        destination = os.path.join(path_temp, blob.name)
        with open(destination, "wb") as my_blob:
            blob_data = blob_client.download_blob()
            blob_data.readinto(my_blob)
    return temp_folder

def download_model(url_model, destination_file):
    blob_service_client_model = BlobServiceClient.from_connection_string(connection_string_model)
    domain_container_name = "hmt-dev-ai-model"
    container_client = blob_service_client_model.get_container_client(domain_container_name)
    file_name = url_model.split(f'/{domain_container_name}/')[-1]
    blob_client = container_client.get_blob_client(file_name)
    exists = blob_client.exists()
    print(f"exists: {exists}")
    if exists:
        with open(destination_file, "wb") as my_blob:
            blob_data = blob_client.download_blob()
            blob_data.readinto(my_blob)
        return 200
    return 404

def download_speaker(path_temp, speaker):
    container = 'speakers'
    if "dizim" in connection_string_speaker:
        print("download speaker from dizim")
    else:
        print("download speaker from dev")
    temp_folder = f'{path_temp}/{speaker}'
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    blob_service_client = BlobServiceClient.from_connection_string(connection_string_speaker)
    container_client = blob_service_client.get_container_client(container)
    blob_names = container_client.list_blobs(name_starts_with=speaker)
    for blob in blob_names:
        if len(blob.name.split("_"))>1:
            if not blob.name.split("_")[1].startswith("original"): 
                blob_client = container_client.get_blob_client(blob.name)
                destination = os.path.join(path_temp, blob.name)
                with open(destination, "wb") as my_blob:
                    blob_data = blob_client.download_blob()
                    blob_data.readinto(my_blob)
                    
        if len(blob.name.split("_"))==1:
            blob_client = container_client.get_blob_client(blob.name)
            destination = os.path.join(path_temp, blob.name)
            with open(destination, "wb") as my_blob:
                blob_data = blob_client.download_blob()
                blob_data.readinto(my_blob)
            
    return temp_folder


def get_blob_names_in_folder(path_temp, speaker):
    container = 'speakers'
    if "dizim" in connection_string_speaker:
        print("download speaker from dizim")
    else:
        print("download speaker from dev")
    temp_folder = f'{path_temp}/{speaker}'
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    blob_service_client = BlobServiceClient.from_connection_string(connection_string_speaker)
    container_client = blob_service_client.get_container_client(container)

    blob_names = container_client.list_blobs(name_starts_with=speaker)
    for blob in blob_names:
        print(blob)
        
def download_folder_model(url, model_path):
    blob_service_client_model = BlobServiceClient.from_connection_string(connection_string_model)
    domain_container_name = "hmt-dev-ai-model"
    container_client = blob_service_client_model.get_container_client(domain_container_name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_folder = os.path.basename(url)
    blob_names = container_client.list_blobs(name_starts_with=model_folder)
    for blob in blob_names:
        blob_client = container_client.get_blob_client(blob.name)
        destination = os.path.join(model_path, blob.name.split("/")[-1])
        with open(destination, "wb") as my_blob:
            blob_data = blob_client.download_blob()
            blob_data.readinto(my_blob)
    return model_path