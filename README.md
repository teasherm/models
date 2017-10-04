# Models

Implementations of research driven models in TensorFlow, to enhance understanding

- [Deep Q Network](deep_q): Deep Q Learning in simple OpenAI Gym environment
  - *sources*: [OpenAI Baselines](https://github.com/openai/baselines/tree/master/baselines/deepq), [PyTorch Tutorial](http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [GAN](vanilla_gan): Simple GAN
  - *sources*: [wiseodd's Generative Models repo](https://github.com/wiseodd/generative-models)
- [VAE](vanilla_vae): Simple Variational Autoencoder
  - *sources*: [wiseodd's Generative Models repo](https://github.com/wiseodd/generative-models)

## Training models

```sh
make train MODEL={model_folder}

# e.g., for Deep Q
make train MODEL=deep_q
```
