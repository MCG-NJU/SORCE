# SORCE: Small Object Retrieval in Complex Environments

<p align="center">
      <a href="https://scholar.google.com.hk/citations?user=dvUKnKEAAAAJ&hl=en" target='_blank'>Chunxu Liu*</a>,&nbsp;
      <a href="https://scholar.google.com/citations?user=1Yz2NrEAAAAJ&hl=en" target='_blank'>Chi Xie*</a>,&nbsp;
      <a>Xiaxu Chen</a>,&nbsp;
      <a href="https://scholar.google.com/citations?user=oO53gjEAAAAJ&hl=en" target='_blank'>Feng Zhu</a>,&nbsp;
      <a href="https://scholar.google.com/citations?user=1c9oQNMAAAAJ&hl=en" target='_blank'>Rui Zhao</a>,&nbsp;
      <a href="https://scholar.google.com.hk/citations?user=HEuN8PcAAAAJ&hl=en" target='_blank'>Limin Wang</a>,&nbsp;
    <br>
  Nanjing University, &nbsp; SenseTime Research
  </p>

<p align="center">
  <a href="https://arxiv.org/abs/2505.24441" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-📕-red">
  </a> 
  <a href="https://huggingface.co/datasets/lcxrocks/sorce-1k" target='_blank'>
    <img src="https://img.shields.io/badge/Dataset Page-🤗-yellow">
  </a>
</p>

## Overview

**TL; DR.** We introduce **S**mall **O**bject **R**etrieval in **C**omplex **E**nvironments (**SORCE**) task, 
which is a new subfield of T2IR, focusing on retrieving small objects in complex images. 

![Overview](./assets/overview.png)

We introduce a new dataset, **SORCE-1K**, comprising 1,023 image-text pairs in which each caption describes only 
a localized object region. This design explicitly avoids providing contextual clues from the broader scene, thereby 
preventing models from exploiting shortcut cues. 

Additionally, we demonstrate that with the use of simple yet effective Regional Prompts (ReP), multimodal large 
language models (MLLMs) can accurately attend to and embed the corresponding image regions. Our fine-tuned models 
are available for evaluation here.



## Dataset Preparation
Please download SORCE-1K dataset from [Hugging Face](https://huggingface.co/datasets/sgm-vfi/sorce-1k) and place it in the `datasets` folder.

```commandline
mkdir datasets
huggingface-cli download --repo-type dataset --resume-download lcxrocks/sorce-1k --local-dir ./datasets/sorce-1k
```


## Environment Setup
Please make sure the `transformers` version is compatible. 

```commandline
conda create -n sorce python=3.11
pip install -r requirements.txt
```
## Evaluation
To evaluate the model, please run the following command, which will download the [🤗hugginface pretrained model](https://huggingface.co/lcxrocks/e5-v-ReP/).

```commandline
bash dist_eval.sh
```


## Citation


If you think this project is helpful in your research or for application, please feel free to leave a star⭐️ and cite our paper:


```

@misc{liu2025sorcesmallobjectretrieval,
      title={SORCE: Small Object Retrieval in Complex Environments}, 
      author={Chunxu Liu and Chi Xie and Xiaxu Chen and Wei Li and Feng Zhu and Rui Zhao and Limin Wang},
      year={2025},
      eprint={2505.24441},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.24441}, 
}

```

## License and Acknowledgement
This project is released under the Apache 2.0 license. 
The codes are based on [E5-V](https://github.com/kongds/E5-V). 
Please also follow their licenses. Thanks for their awesome work!