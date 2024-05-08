This is a coursera project on generative adversarial networks
# Generative Adversarial Network (GAN)

This Python script uses PyTorch to implement Deep learning with pytorch: Generative Adversarial Network (GAN) for generating images. In this project there were images with numbers which were blurry and were dificult to read and using GAN, those numbers were enhanced so that it can be understood easily. It is just one simple application of GAN network among many other like data augmentation, image enhancement, text generation etc. 

## Dependencies

The script uses the following libraries:
- torch: For creating and training the GAN.
- numpy: For numerical operations.
- matplotlib: For plotting and visualizing images.
- tqdm: For displaying a progress bar during training.
- torchsummary: For displaying a summary of the model.
- torchvision: For loading datasets and performing image transformations.

## How it works

The script sets up a GAN with the following parameters:
- `device`: The device to run the training on. This is set to "cuda", which means the training will run on the GPU if available.
- `batch_size`: The number of samples per batch during training.
- `noise_dim`: The dimension of the noise vector for the generator.
- `lr`: The learning rate for the optimizer.
- `beta_1` and `beta_2`: The coefficients used for computing running averages of gradient and its square for the Adam optimizer.
- `epochs`: The number of times the training loop should run.

The script uses the Adam optimizer and binary cross entropy loss for training the GAN.

## Usage

To run the script, simply execute the following command:

```shell
python generative_adversarial_network.py
