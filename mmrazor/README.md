## MMRazor-MasKD  
This is a modified version of MMRazor (v0.1.0), which adds supports of knowledge distillation on RCNN heads of two-stage detectors.

### Preparations  
* Install requirements:
    The MM packages used in our experiments:
    ```
    mmcv-full==1.4.0
    mmdet==2.14.0
    mmcls==0.16.0
    ```

* Put the COCO dataset into `./data` folder following [[this url]](https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html#prepare-datasets).  
