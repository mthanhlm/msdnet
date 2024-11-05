## MSDNet: Multi-Scale Decoder for Few-Shot Semantic Segmentation via Transformer-Guided Prototyping
This is the implementation of the paper "MSDNet: Multi-Scale Decoder for Few-Shot Semantic Segmentation via Transformer-Guided Prototyping" by Fateh Amirreza, Mohammadi Mohammadreza, Jahed-Motlagh Mohammadreza.

For more information, check out our paper on [[arXiv](https://arxiv.org/abs/2409.11316)], [[paperswithcode](https://paperswithcode.com/paper/msdnet-multi-scale-decoder-for-few-shot)].

<p align="middle">
    <img src="data/assets/overview_git.png">
</p>



Conda environment settings:
```bash
git clone https://github.com/mthanhlm/msdnet
conda create -n msd python=3.8
conda activate msd
pip install -r requirements.txt
```
Download this file: [[Drive](https://drive.google.com/file/d/12f-OC8SCA3mnIwUFDKTcAiriub9TWArq/view?usp=drive_link)] and extract inside folder msdnet
## Run demo

> ```bash
> python app.py
> ```

 1. Select **pascal_20itr.pt** in model or other pretrained file.
 2. Select *support and mask image* in folder: 

> test image/*/support

 3. Select *query image* in folder: 

> test image/

## Acknowledgements
This project is implement from MSDNet: https://github.com/amirrezafateh/msdnet
## Citation
If you use this repository in your work, please cite the following paper:
```bibtex
@article{fateh2024msdnet,
  title={MSDNet: Multi-Scale Decoder for Few-Shot Semantic Segmentation via Transformer-Guided Prototyping},
  author={Fateh, Amirreza and Mohammadi, Mohammad Reza and Motlagh, Mohammad Reza Jahed},
  journal={arXiv preprint arXiv:2409.11316},
  year={2024}
}