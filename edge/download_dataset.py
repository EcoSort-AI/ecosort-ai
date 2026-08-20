import os
import boto3
from dotenv import load_dotenv

load_dotenv()

R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME") 
PREFIXO_NUVEM = "dataset/" 
LOCAL_FOLDER=os.getenv(r"LOCAL_FOLDER")

s3_client = boto3.client(
    service_name='s3',
    endpoint_url=f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
    aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
    region_name="auto"
)

paginator = s3_client.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=R2_BUCKET_NAME, Prefix=PREFIXO_NUVEM)

print(f"Iniciando o download da pasta '{PREFIXO_NUVEM}' do R2...")
arquivos_baixados = 0

for page in pages:
    if "Contents" in page:
        for obj in page["Contents"]:
            file_key = obj["Key"]
            
            if file_key.endswith('/'): 
                continue 
            
            caminho_relativo = file_key.replace(PREFIXO_NUVEM, "", 1)
            local_file_path = os.path.join(LOCAL_FOLDER, caminho_relativo)
            
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            print(f"Baixando: {caminho_relativo}")
            s3_client.download_file(R2_BUCKET_NAME, file_key, local_file_path)
            arquivos_baixados += 1

print(f"\nDownload concluído com sucesso! Total de imagens baixadas: {arquivos_baixados}")