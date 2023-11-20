# FE-Net: Biomedical image segmentation for efficient segmentation
This is the official implementation for above paper.

## Requirements
* torch == 1.13.1
* tensorboard == 2.11.2
* ...

## Datasets
All datasets used in paper are public, you can download online

Split the datasets for train, validation and test

## Results


| Dataset\Type       | Original image           | GT           |  Prediction           |
|---------------|----------------|----------------|----------------|
| Kvasir-SEG           | ![Image 1](examples/9_origin_kvasir.png) | ![Image 2](examples/9_gt_mask_kvasir.png) | ![Image 3](examples/9_pred_mask_kvasir.png) |
| DSB 2018           | ![Image 4](examples/18_origin_DSB.png) | ![Image 5](examples/18_gt_mask_DSB.png) | ![Image 6](examples/18_pred_mask_DSB.png) |
| CVC-clinicDB           | ![Image 7](examples/1_origin_cvc.png) | ![Image 8](examples/1_gt_mask_cvc.png) | ![Image 9](examples/1_pred_mask_cvc.png) |
