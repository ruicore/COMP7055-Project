## How to Run

This code requires Python3.12

To train StyleGAN2-ADA, we need [NVlabs](https://github.com/NVlabs/stylegan2-ada-pytorch) lab, download this repo and run the code inside stylegan2-ada-pytorch directory.


1. syntheticmodel.ipynb is used to train StyleGAN2-ADA model.
2. single.py is used to train the classifier model, it requires the GAN model to generate images.
3. distribution.py is used to visualize the distribution of the generated images, it requires the generated images and the trained classifier model.
4. cgan.py is used to train cgan model.
5. classifier.py is used to train the classifier model for supplement, it requires the GAN model to generate images.