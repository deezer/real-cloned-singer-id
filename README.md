# From Real to Cloned Singer Identification

by [Dorian Desblancs](https://www.linkedin.com/in/dorian-desblancs), [Gabriel Meseguer-Brocal](https://www.linkedin.com/in/gabriel-meseguer-brocal-1032a42b), [Romain Hennequin](http://romain-hennequin.fr/En/index.html), and [Manuel Moussallam](https://mmoussallam.github.io/).

## About

This repository contains the code we used to train and evaluate the models from the paper. The cloned and closed datasets will unfortunately never be made public due to copyright concerns. However, the [FMA](https://github.com/mdeff/fma) and [MTG](https://mtg.github.io/mtg-jamendo-dataset/) splits used to evaluate our models can be found in the `splits.zip` file from the [Releases](https://github.com/deezer/real-cloned-singer-id/releases) section! We highly recommend you use them to evaluate your singer identification models!

## Getting Started

In order to explore our repository, one can start with the following:
```bash
# Clone and enter repository
git clone https://github.com/deezer/real-cloned-singer-id
cd real-cloned-singer-id

# Build and run docker container with dependencies installed
make build
make run
```

From there, you can expand upon or use the parts of this repo you need. The `foundation/` directory contains the base Transformer and Audio model, that are then used to create our embedding models. The code for training these backbone models can then be found in the `training/` directory. Finally, these embeddings are evaluated in the `evaluation/` section. Note that these are pickled for each track to accelerate singer identification result computation (see `evaluation/pickle_embeddings.py`).

## Disclaimer

This repository is not meant to run as is. It has been trimmed down a lot since most of the experimental setup cannot be made public. However, some interesting bits, such as the data pipelines for large-scale training or artist-level constrastive learning, are left here to serve as inspiration for future research in singer identification, music information retrieval, or even audio in general. Happy hacking!

## Reference

If you use this repository, please consider citing:

```
@article{desblancs2024real,
  title={From Real to Cloned Singer Identification},
  author={Desblancs, Dorian and Meseguer-Brocal, Gabriel and Hennequin, Romain and Moussallam, Manuel},
  journal={arXiv preprint arXiv:2407.08647},
  year={2024}
}
```

Our paper can be found on [arXiv](https://arxiv.org/abs/2407.08647) 🌟 It will be presented at ISMIR 2024 🌉
