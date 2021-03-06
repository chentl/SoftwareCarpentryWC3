# Image Processing using Pillow

## Blur effect

|                        |                       Original                       |                   Blurred                    |
| :--------------------: | :--------------------------------------------------: | :------------------------------------------: |
|          RGB           |        ![rgb_original](img/rgb_original.png)         |        ![rgb_blur](img/rgb_blur.png)         |
| RGB with Alpha channel | ![rgba_original](img/rgba_original_checkerboard.png) | ![rgba_blur](img/rgba_blur_checkerboard.png) |



## Luminance adjustment

|                                                     |           Original (average_luminance=0.666)            |                                                     |
| :-------------------------------------------------: | :-----------------------------------------------------: | :-------------------------------------------------: |
|                                                     |          ![rgb_original](img/rgb_original.png)          |                                                     |
|  **target_luminance = 0.4; fixed scale method**   |   **target_luminance = 0.4; average of two methods**    |   **target_luminance = 0.4; fixed shift method**    |
| ![rgb_luminance_scale](img/rgb_luminance_scale.png) | ![rgb_luminance_average](img/rgb_luminance_average.png) | ![rgb_luminance_shift](img/rgb_luminance_shift.png) |


<p style="padding-top: 2em; text-align: center;">
 <a href="https://github.com/chentl/SoftwareCarpentryWC3/tree/master/img_processing" class="btn" style=" color: #159957; border-color: #159957;">View on GitHub</a>
</p>