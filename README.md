# Token Merging For HybridModel

This repository contains modifications to the following open-source projects:
- [pytorch-image-models](https://github.com/huggingface/pytorch-image-models): PyTorch Image Models (Apache 2.0 License)
- [ToMe](https://github.com/facebookresearch/ToMe): Token Merging for Vision Transformers (CC BY 4.0 License)

## Overview
- This project integrates Token Merging (ToMe) into MobileViTs to reduce computational costs.
- The modifications are organized into separate directories for each project:
  - [`timm`](./timm/): Modifications for timm library included in pytorch-image-models.
  - [`ToMe`](./ToMe/): Modifications for ToMe.

## Licenses
- This repository contains modified code from `pytorch-image-models` and `ToMe`.
- Each subdirectory (`timm/` and `ToMe/`) includes its respective license:
  - `pytorch-image-models`: Apache 2.0 License
  - `ToMe`: CC BY 4.0 License
- **Please ensure that you comply with the respective licenses when using this repository.**

## How to Use
1. Clone the original repositories:
   - pytorch-image-models: `git clone https://github.com/huggingface/pytorch-image-models`
   - ToMe: `git clone https://github.com/facebookresearch/ToMe`

2. Refer to the modification details provided in each subdirectory:
   - For `timm`: See [`timm/README.md`](./timm/README.md)
   - For `ToMe`: See [`ToMe/README.md`](./ToMe/README.md)

3. Manually apply the described changes to the original code.

4. Use the modified code for your experiments or projects.

