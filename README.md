# Models

Research driven models in NumPy and Tensorflow

- [Deep NumPy](deep_q): Neural network implementations in pure NumPy
  - *sources*: [wiseodd's NumPy implementations](https://github.com/wiseodd/hipsternet), [karpathy's pg-pong gist](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
- [Deep Q Network](deep_q): Deep Q Learning in simple OpenAI Gym environment
  - *sources*: [OpenAI Baselines](https://github.com/openai/baselines/tree/master/baselines/deepq), [PyTorch Tutorial](http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [GAN](vanilla_gan): Simple GAN
  - *sources*: [wiseodd's Generative Models repo](https://github.com/wiseodd/generative-models)
- [VAE](vanilla_vae): Simple Variational Autoencoder
  - *sources*: [wiseodd's Generative Models repo](https://github.com/wiseodd/generative-models)

## Training

```sh
make train MODEL={model_directory}

# e.g., for Deep Q
make train MODEL=deep_q
```
