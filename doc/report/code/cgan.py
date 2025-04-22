# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras import layers
from IPython import display
import time

# ============ CONFIG ============
class cfg:
    BATCH_SIZE = 64
    EPOCHS = 500
    NOISE_DIM = 100
    NUM_EXAMPLES_TO_GENERATE = 16

# ============ LABELS & DATA ============
dirs = ['neutral', 'fear', 'angry', 'surprise', 'disgust', 'happy', 'sad']
label_map = {label: idx for idx, label in enumerate(dirs)}
num_classes = len(label_map)

train_images = []
train_labels = []

for folder in dirs:
    directory = f'../input/fer2013/train/{folder}/'
    for image_name in os.listdir(directory):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(directory, image_name)
            image = cv2.imread(image_path, 0)
            image = np.expand_dims(image, 2)
            image = (image - 127.5) / 127.5
            train_images.append(image)
            train_labels.append(label_map[folder])

train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
dataset = dataset.shuffle(1000).batch(cfg.BATCH_SIZE)

# ============ CGAN Generator ============
def make_generator_model():
    noise_input = layers.Input(shape=(cfg.NOISE_DIM,))
    label_input = layers.Input(shape=(1,), dtype='int32')

    label_embedding = layers.Embedding(num_classes, cfg.NOISE_DIM)(label_input)
    label_embedding = layers.Flatten()(label_embedding)

    combined = layers.multiply([noise_input, label_embedding])

    x = layers.Dense(6 * 6 * 256, use_bias=False)(combined)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Reshape((6, 6, 256))(x)
    x = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)

    return tf.keras.Model([noise_input, label_input], x)

# ============ CGAN Discriminator ============
def make_discriminator_model():
    image_input = layers.Input(shape=(48, 48, 1))
    label_input = layers.Input(shape=(1,), dtype='int32')

    label_embedding = layers.Embedding(num_classes, 48 * 48)(label_input)
    label_embedding = layers.Reshape((48, 48, 1))(label_embedding)

    combined = layers.Concatenate()([image_input, label_embedding])

    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(combined)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    return tf.keras.Model([image_input, label_input], x)

# ============ Losses and Optimizers ============
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# ============ Training Loop ============
@tf.function
def train_step(images, labels):
    noise = tf.random.normal([cfg.BATCH_SIZE, cfg.NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise, labels], training=True)

        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([generated_images, labels], training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def generate_and_save_images(model, epoch, test_input, test_labels):
    predictions = model([test_input, test_labels], training=False)
    predictions = (predictions * 127.5 + 127.5).numpy().astype(np.uint8)

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.title(dirs[int(test_labels[i])], fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.show()

def train(dataset, epochs):
    seed = tf.random.normal([cfg.NUM_EXAMPLES_TO_GENERATE, cfg.NOISE_DIM])
    seed_labels = tf.convert_to_tensor(np.random.randint(0, num_classes, size=(cfg.NUM_EXAMPLES_TO_GENERATE, 1)))

    for epoch in range(epochs):
        start = time.time()

        for image_batch, label_batch in dataset:
            train_step(image_batch, tf.expand_dims(label_batch, axis=1))

        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed, seed_labels)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'Time for epoch {epoch+1} is {time.time() - start:.2f} sec')

    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed, seed_labels)

# ============ Start Training ============
# tf.config.run_functions_eagerly(True)
# train(dataset, cfg.EPOCHS)

# ============ Generate Images After Training ============
def generate_images_after_training():
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    seed = tf.random.normal([cfg.NUM_EXAMPLES_TO_GENERATE, cfg.NOISE_DIM])
    seed_labels = tf.convert_to_tensor(np.random.randint(0, num_classes, size=(cfg.NUM_EXAMPLES_TO_GENERATE, 1)))
    generate_and_save_images(generator, 9999, seed, seed_labels)

# Example call after training:
# generate_images_after_training()
