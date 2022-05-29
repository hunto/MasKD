# Learning masks

## Preparations  
* Install mmdetection `v2.14.0` and mmcv-full  
    ```shell
    pip install mmdet==2.14.0
    ```
    For mmcv-full, please check [here](https://mmdetection.readthedocs.io/en/latest/get_started.html#install-mmdetection) and [compatibility](https://mmdetection.readthedocs.io/en/latest/compatibility.html).

* Put the COCO dataset into `./data` folder following [[this url]](https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html#prepare-datasets).  

**Note:** if you use a different mmdet version, please replace the `configs`, `tools` with the corresponding files of the version, then add `import maskd_hook` into `tools/train.py` to register the `MasKDHook`.

## Train mask tokens  

The configs are in `configs/maskd`.

Example of training mask tokens on `cascade_mask_rcnn_x101`:
```shell
sh tools/dist_train.sh mmdetection/configs/maskd/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco.py 8 work_dirs/cmx101
```
Then the obtained checkpoint `work_dirs/cmx101/iter_2000.pth` will be used in the KD stage.