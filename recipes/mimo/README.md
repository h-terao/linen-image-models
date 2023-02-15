# MIMO networks

*NOTE*: now in progress

This recipe trains MIMO networks.
Official tensorflow implementation can be found [here](https://github.com/google/edward2/tree/main/experimental/mimo).

## Preparing ILSVRC2012 datasets

1. Download `ILSVRC2012_devkit_t12.tar.gz`, `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` from ImageNet website and place them under `$IN_DIR`. (Below, save them as ILSVRC2012/ILSVRC2012_img_train.tar, for example.)
2. Type the following commands:

```bash
IN_DIR="ILSVRC2012"
OUT_DIR="data/ILSVRC2012"

python util_scripts/extract_imagenet.py --in_dir ${IN_DIR} --out_dir ${OUT_DIR}/temp
python util_scripts/image_folder_to_tfrecords --in_dir ${OUT_DIR}/temp --out_dir ${OUT_DIR}
rm -rf ${OUT_DIR}/temp
```

## Results

I trained resnet18, resnet34 and resnet50 with the following options `--ensemble_size=3 --half_precision`. Training of resnet50 with two RTX3090 GPUs consumed about 46GB of RAM and 42GB of VRAM and took about 45 minutes per epoch.



## Citation

```
@inproceedings{havasi2021training,
  author = {Marton Havasi and Rodolphe Jenatton and Stanislav Fort and Jeremiah Zhe Liu and Jasper Snoek and Balaji Lakshminarayanan and Andrew M. Dai and Dustin Tran},
  title = {Training independent subnetworks for robust prediction},
  booktitle = {International Conference on Learning Representations},
  year = {2021},
}
```