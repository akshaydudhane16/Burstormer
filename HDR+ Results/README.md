# Burstormer: Burst Image Restoration and Enhancement Transformer (CVPR 2023)

[Akshay Dudhane](https://scholar.google.com/citations?hl=en), [Syed Waqas Zamir](https://scholar.google.es/citations?user=WNGPkVQAAAAJ&hl=en), [Salman Khan](https://salman-h-khan.github.io/), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en), and [Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2304.01194.pdf)


<hr />

> **Abstract:** *On a shutter press, modern handheld cameras capture multiple images in rapid succession and merge them to generate a single image. However, individual frames in a burst are misaligned due to inevitable motions and contain multiple degradations. The challenge is to properly align the
successive image shots and merge their complimentary information to achieve high-quality outputs. Towards this direction, we propose Burstormer: a novel
transformer-based architecture for burst image restoration and enhancement. In comparison to existing works, our approach exploits multi-scale local and non-local features to achieve improved alignment and feature fusion. Our key idea is to enable inter-frame communication in the burst neighborhoods for information aggregation and progressive fusion while modeling the burst-wide context. However, the input burst frames need to be properly aligned before fusing their information. Therefore, we propose an enhanced deformable alignment module for aligning burst features with regards to the reference frame. Unlike existing methods, the proposed alignment module not only aligns burst features but also exchanges feature information and maintains focused communication with the reference frame through the proposed referencebased feature enrichment mechanism, which facilitates handling complex motions. After multi-level alignment and enrichment, we re-emphasize on inter-frame communication within burst using a cyclic burst sampling module. Finally, the inter-frame information is aggregated using the proposed burst feature fusion module followed by progressive upsampling. Our Burstormer outperforms state-ofthe-art methods on burst super-resolution, burst denoising and burst low-light enhancement.*
<hr />

## Network Architecture

<img src = '../block_diagram.png'>

## Visual Results of the proposed Burstormer and existing HDR+ approach

<img src = 'visual_rsults.png'>


## Cross dataset testing on HDR+ dataset

We have validated our Burstormer network architecture on HDR+ dataset. Visual results are available [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/akshay_dudhane_mbzuai_ac_ae/EvpXnSs8S_ZGrnNl0AlQ3AcBUA97tyAmMGzelAnsPVmVsA?e=1vnzbX).
            Stay tuned for further updates.

## Citation
            
    @article{dudhane2023burstormer,
              title={Burstormer: Burst Image Restoration and Enhancement Transformer},
              author={Dudhane, Akshay and Zamir, Syed Waqas and Khan, Salman and Khan, Fahad Shahbaz and Yang, Ming-Hsuan},
              journal={arXiv preprint arXiv:2304.01194},
              year={2023}
            }
            
    @article{hasinoff2016burst,
              title={Burst photography for high dynamic range and low-light imaging on mobile cameras},
              author={Hasinoff, Samuel W and Sharlet, Dillon and Geiss, Ryan and Adams, Andrew and Barron, Jonathan T and Kainz, Florian and Chen, Jiawen and Levoy, Marc},
              journal={ACM Transactions on Graphics (ToG)},
              volume={35},
              number={6},
              pages={1--12},
              year={2016}
            }
            
