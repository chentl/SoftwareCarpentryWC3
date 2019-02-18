# Image Processing using Pillow

## Blur effect

|                        |                Original                 |             Blurred             |
| :--------------------: | :-------------------------------------: | :-----------------------------: |
|          RGB           |  ![rgb_original](img/rgb_original.png)  |  ![rgb_blur](img/rgb_blur.png)  |
| RGB with Alpha channel | ![rgba_original](img/rgba_original.png) | ![rgba_blur](img/rgba_blur.png) |



## Luminance adjustment

|                                                         |           Original (average_luminance=0.666)            |                                                           |
| :-----------------------------------------------------: | :-----------------------------------------------------: | :-------------------------------------------------------: |
|                                                         |          ![rgb_original](img/rgb_original.png)          |                                                           |
|    **target_luminance = 0.4; fixed scaling method**     |   **target_luminance = 0.4; average of two methods**    |      **target_luminance = 0.4; fixed delta method**       |
| ![rgb_luminance_0.4_1.0](img/rgb_luminance_0.4_1.0.png) | ![rgb_luminance_0.4_0.0](img/rgb_luminance_0.4_0.0.png) | ![rgb_luminance_0.4_-1.0](img/rgb_luminance_0.4_-1.0.png) |


