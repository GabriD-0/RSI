import os
from PIL import Image

def remove_imagens_corrompidas(caminho):
    total_removidas = 0
    extensoes_validas = ['.jpg']
    
    for root, _, files in os.walk(caminho):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in extensoes_validas:
                continue  # ignora arquivos que não são imagens
            
            caminho_arquivo = os.path.join(root, file)
            try:
                with Image.open(caminho_arquivo) as img:
                    img.verify()  # Verifica se a imagem está íntegra
            except Exception as e:
                print(f"Removendo imagem corrompida: {caminho_arquivo} ({e})")
                os.remove(caminho_arquivo)
                total_removidas += 1
                
    print(f"\nTotal de imagens corrompidas removidas: {total_removidas}")

# Defina o caminho da pasta Categories
pasta_categories = "./dataset/Categories"
remove_imagens_corrompidas(pasta_categories)
