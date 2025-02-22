import csv
import os
import shutil

# Nome do arquivo CSV a ser lido
csv_file = r'.\agrigorev\clothing-dataset-full\versions\1\images.csv'

# Pasta onde as imagens estão atualmente (pode ser o mesmo diretório do script)
origem = r'.\agrigorev\clothing-dataset-full\versions\1\images_compressed'

# Pasta base onde as imagens serão movidas
destino_base = r'.\RSI\dataset'  # defina o destino desejado

# Cria a pasta base de destino se não existir
if not os.path.exists(destino_base):
    os.makedirs(destino_base)
    print(f"Pasta base de destino '{destino_base}' criada.")

# Abre e lê o CSV
with open(csv_file, newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Obtém o nome da imagem (adicionando a extensão .jpg)
        imagem_nome = row['image'] + '.jpg'
        label = row['label']
        
        # Caminho completo da imagem na pasta de origem
        imagem_caminho = os.path.join(origem, imagem_nome)
        if not os.path.exists(imagem_caminho):
            print(f"Imagem {imagem_nome} não encontrada na pasta de origem.")
            continue

        # Define a pasta de destino específica para o label
        destino_pasta = os.path.join(destino_base, label)
        if not os.path.exists(destino_pasta):
            os.makedirs(destino_pasta)
            print(f"Pasta '{destino_pasta}' criada.")

        # Caminho completo de destino para a imagem
        destino_caminho = os.path.join(destino_pasta, imagem_nome)
        shutil.move(imagem_caminho, destino_caminho)
        print(f"Imagem {imagem_nome} movida para {destino_caminho}.")