
# Se estiver em local diferente do Colab, adapte a instalação:
# priorize estas versões
# pip install --upgrade --force-reinstall "tensorflow==2.13.*" "tensorflow_hub==0.14.*"

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

# Limita as threads de execução dentro de cada operação e entre operações
tf.config.threading.set_intra_op_parallelism_threads(3)
tf.config.threading.set_inter_op_parallelism_threads(3)

# Se estiver usando seu próprio dataset organizado em pastas, não é necessário carregar o Fashion MNIST.
# from keras.datasets import fashion_mnist

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


# Defina o diretório do seu dataset. Certifique-se de que a estrutura seja:
# ./dataset/Categories/<classe1>, <classe2>, ...
data_dir = "./dataset/Categories"
os.makedirs(data_dir, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
datagen_kwargs = dict(rescale=1./255, validation_split=0.20)
dataflow_kwargs = dict(target_size=IMAGE_SIZE,
                       batch_size=BATCH_SIZE,
                       interpolation="bilinear")

# Gerador para dados de validação (20% dos dados)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    data_dir,
    subset="validation",
    shuffle=False,
    **dataflow_kwargs
)

# Gerador para dados de treinamento (80% dos dados)
do_data_augmentation = False  # Altere para True se desejar data augmentation
if do_data_augmentation:
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        **datagen_kwargs
    )
else:
    train_datagen = valid_datagen

train_generator = train_datagen.flow_from_directory(
    data_dir,
    subset="training",
    shuffle=True,
    **dataflow_kwargs
)

# Definindo o número de classes a partir do gerador
num_classes = len(train_generator.class_indices)
print("Número de classes:", num_classes)

# Construção do modelo usando o módulo BiT (Transfer Learning)
MODULE_HANDLE = 'https://tfhub.dev/google/bit/m-r50x3/1'
N_FEATURES = 256  # Dimensão do penúltimo Dense

print("Building model with", MODULE_HANDLE)
model = tf.keras.Sequential([
    hub.KerasLayer(MODULE_HANDLE,
                   input_shape=(224, 224, 3),
                   trainable=False),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(N_FEATURES,
                          activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(num_classes,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])

# Ajuste dinâmico do LR de acordo com o batch size
lr = 0.003 * BATCH_SIZE / 512
SCHEDULE_BOUNDARIES = [200, 300, 400]
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=SCHEDULE_BOUNDARIES,
    values=[lr, lr*0.1, lr*0.001, lr*0.0001]
)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])

model.summary()

# Cálculo dos passos por época
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = valid_generator.samples // valid_generator.batch_size

if steps_per_epoch == 0 or validation_steps == 0:
    print("Não há imagens suficientes para treinar. Verifique o conteúdo de", data_dir)
else:
    # Treinamento
    hist = model.fit(
        train_generator,
        epochs=2,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_generator,
        validation_steps=validation_steps
    ).history

    # Plot do Loss
    plt.figure()
    plt.ylabel("Loss (treinamento e validação)")
    plt.xlabel("Passos de Treinamento")
    plt.ylim([0, 2])
    plt.plot(hist["loss"], label='Training Loss')
    plt.plot(hist["val_loss"], label='Validation Loss')
    plt.legend()
    plt.show()

    # Plot da Acurácia
    plt.figure()
    plt.ylabel("Acurácia (treinamento e validação)")
    plt.xlabel("Passos de Treinamento")
    plt.ylim([0, 1])
    plt.plot(hist["accuracy"], label='Training Accuracy')
    plt.plot(hist["val_accuracy"], label='Validation Accuracy')
    plt.legend()
    plt.show()

    # Salvando o modelo completo
    saved_model_path = './bit_model'
    tf.saved_model.save(model, saved_model_path)

    # Salvando somente o extrator de features (usando a saída da camada intermediária)
    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.layers[-3].output  # Camada Dense com N_FEATURES
    )
    feature_extractor_path = './bit_feature_extractor'
    feature_extractor.save(feature_extractor_path, save_format='tf')

    print("Treino finalizado (se houveram imagens) e modelos salvos.")
