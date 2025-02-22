import os
import io
import json
import glob
import time
import pickle # Modulo padrão de serialização e desserialização
import shutil
import base64
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from tqdm import tqdm # barras de progresso para loops
from scipy import spatial
from shutil import move
from pathlib import Path
from pandas import read_csv
from annoy import AnnoyIndex # Approximate Nearest Neighbors Oh Yeah
from flask import Flask, request, jsonify, send_file
from flask import redirect, url_for, flash, render_template

_PPATH = os.path.join(os.getcwd(), 'folderzinha') # utilizado para colab saber qual pasta está

def carregar_imagem(path):
  img = tf.io.read_file(path)
  img = tf.io.decode_jpeg(img, channels=3)
  img = tf.image.resize_with_pad(img, 244, 244)
  img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

  return img

MODULE_HANDLE = 'https://tfhub.dev/google/bit/m-r50x3/1'
module = hub.load(MODULE_HANDLE)

def carregar_imagem(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    # Note que usamos 224 para manter o mesmo tamanho que no treinamento
    img = tf.image.resize_with_pad(img, 224, 224)
    img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    return img

def arquivos_permitidos(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower () in ALLOWED_EXTENSIONS


topK = 5
threshold = 0.3
UPLOAD_FOLDER = _PPATH
ALLOWED_EXTENSIONS = set(['zip'])

application = Flask(__name__)


@application.route('/')
def home():
    return "Hello world!"


@application.route('/base_imagens', methods=['POST'])
def zip_upload():
  os.makedirs(_PPATH, exist_ok=True)
  shutil.rmtree(_PPATH)
  os.chdir(_PPATH)
  file = request.files['files']
  if file and arquivos_permitidos(file.filename):
    file.save(os.path.join(UPLOAD_FOLDER, 'folder1.zip'))

  folder1_path = os.path.join(_PPATH, 'folder1.zip')
  folder1_ipath = os.path.join(_PPATH, 'folder1')
  os.makedirs(folder1_ipath, exist_ok=True)
  with zipfile.ZipFile(folder1_path, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(_PPATH, 'folder1'))


  img_paths = []

  for path in Path(folder1_ipath).rglob('*.jpg'):
    img_paths.append(path)

  img_vector_path = os.path.join(_PPATH, 'vecs1')
  Path(img_vector_path).mkdir(parents=True, exist_ok=True)


  for filename in tqdm(img_paths):
    outfile_name = os.path.basename(filename).split('.')[0] + ".npz" # Extensão para armazenar vetores
    out_path_file = os.path.join(img_vector_path, outfile_name)
    if not os.path.exists(out_path_file):
      img = carregar_imagem(str(filename))
      features = module(img)
      feature_set = np.squeeze(features)
      print(features.shape)
      np.savetxt(out_path_file, feature_set, delimiter=',')

  index_arquivo_para_nome_arquivo = {}
  index_arquivo_para_nome_vector = {}

  # annoy config:
  dimension = 2048
  n_nearest_neighbors = 20
  trees = 10000

  allfiles = glob.glob(os.path.join(_PPATH, 'vecs1', '*.npz'))
  t = AnnoyIndex(dimension, metric='angular') # Distância angular (ou cosseno).

  for findex, fname in tqdm(enumerate(allfiles)):
    file_vector = np.loadtxt(fname)
    file_name = os.path.basename(fname).split('.')[0]
    index_arquivo_para_nome_arquivo[findex] = file_name
    index_arquivo_para_nome_vector[findex] = file_vector
    t.add_item(findex, file_vector)

  t.build(trees)

  file_path = os.path.join(_PPATH,'models/indices/')
  Path(file_path).mkdir(parents=True, exist_ok=True)

  t.save(file_path+'indexer.ann')
  pickle.dump(index_arquivo_para_nome_arquivo, open(file_path+"index_arquivo_para_nome_arquivo.p", "wb"))
  pickle.dump(index_arquivo_para_nome_vector, open(file_path+"index_arquivo_para_nome_vector.p", "wb"))

  return 'Processamento de arquivos no servidor, OK!'



@application.route('/test_imagens', methods=['POST'])
def zip2_upload():
  os.chdir(_PPATH)
  file = request.files['file']
  if file and arquivos_permitidos(file.filename):
      file.save(os.path.join(UPLOAD_FOLDER, 'folder2.zip'))

  folder2_path = os.path.join(_PPATH, '.zip')
  folder2_ipath = os.path.join(_PPATH, 'folder2')
  os.makedirs(folder2_ipath, exist_ok=True)
  with zipfile.ZipFile(folder2_path, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(_PPATH, 'folder2'))

  query_files = []
  for path in Path(folder2_ipath).rglob('*.jpg'):
    query_files.append(path)

  dimension = 2048
  indexer = AnnoyIndex(dimension, 'angular')
  indexer.load(os.path.join(_PPATH,'models/indices/indexer.ann'))
  index_arquivo_para_nome_arquivo = pickle.load(open(os.path.join(_PPATH,'models/indices/index_arquivo_para_nome_arquivo.p'), 'rb'))

  results = pd.DataFrame(columns=['qid','fname','dist'])

  for q in query_files:
    img = carregar_imagem(str(q))
    features = module(img)
    feature_vector = tf.squeeze(features).numpy()
    temp_vec = np.squeeze(module(carregar_imagem(str(q))))
    nns = indexer.get_nns_by_vector(temp_vec, n=topK, include_distances=True)
    col1 = [q.stem]*topK
    col2 = [index_arquivo_para_nome_arquivo[x] for x in nns[0]]
    col3 = nns[1]
    results = results.append(pd.DataFrame({'qid':col1,'fname':col2,'dist':col3}))
    results = results[results.dist<=threshold]

  results = results.reset_index(drop=True).T.to_json()
  return results


if __name__ == "__main__":
    application.debug = True
    application.run()

