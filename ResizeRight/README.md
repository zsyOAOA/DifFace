# ResizeRight
This is a resizing packge for images or tensors, that supports both Numpy and PyTorch seamlessly. The main motivation for creating this is to address some **crucial incorrectness issues** that exist in all other resizing packages I am aware of. As far as I know, it is the only one that performs correctly in all cases.  ResizeRight is specially made for machine learning, image enhancement and restoration challenges.

In the Pytorch case, it is **fully differentiable and can be used inside a neural network** both as a dynamic function (eg. inside some "forward" method) or as a PyTorch layer (nn.Module).
 
The code is inspired by MATLAB's imresize function, but with crucial differences. It is specifically useful due to the following reasons:

1. ResizeRight produces results **identical (PSNR>60dB) to MATLAB for the simple cases** (scale_factor * in_size is integer). None of the Python packages I am aware of, currently resize images with results similar to MATLAB's imresize (which is a common benchmark for image resotration tasks, especially super-resolution). 

2. No other **differntiable** method I am aware of supports **AntiAliasing** as in MATLAB. Actually very few non-differentiable ones, including popular ones, do. This causes artifacts and inconsistency in downscaling. (see [this tweet](https://twitter.com/jaakkolehtinen/status/1258102168176951299) by [Jaakko Lehtinen](https://users.aalto.fi/~lehtinj7/)
 for example).

3. The most important part: In the general case where scale_factor * in_size is non-integer, **no existing resizing method I am aware of (including MATLAB) performs consistently.** ResizeRight is accurate and consistent due to its ability to process **both scale-factor and output-size** provided by the user. This is a super important feature for super-resolution and learning. One must acknowledge that the same output-size can be resulted with varying scale-factors as output-size is usually determined by *ceil(input_size * scale_factor)*. This situation creates dangerous lack of consistency. Best explained by example: say you have an image of size 9x9 and you resize by scale-factor of 0.5. Result size is 5x5. now you resize with scale-factor of 2. you get result sized 10x10. "no big deal", you must be thinking now, "I can resize it according to output-size 9x9", right? but then you will not get the correct scale-fcator which is calculated as output-size / input-size = 1.8.
Due to a simple observation regarding the projection of the output grid to the input grid, ResizeRight is the only one that consistently maintains the image centered, as in optical zoom while complying with the exact scale-factor and output size the user requires. 
This is one of the main reasons for creating this repository. this downscale-upscale consistency is often crucial for learning based tasks (e.g. ["Zero-Shot Super-Resolution"](http://www.wisdom.weizmann.ac.il/~vision/zssr/)), and does not exist in other python packages nor in MATLAB.

4. Misalignment in resizing is a pandemic! Many existing packages actually return misaligned results. it is visually not apparent but can cause great damage to image enhancement tasks.(for example, see [how tensorflow's image resize stole 60 days of my life](https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35)). I personally also suffered from many misfortunate consequences of such missalignment before and throughout making this method.

5. Resizing supports **both Numpy and PyTorch** tensors seamlessly, just by the type of input tensor given. Results are checked to be identical in both modes, so you can safely apply to different tensor types and maintain consistency. No Numpy <-> Torch conversion takes part at any step. The process is done exclusively with one of the frameworks. No direct dependency is needed, so you can run it without having PyTorch installed at all, or without Numpy. You only need one of them.

6. For PyTorch you can either use a **ResizeLayer (nn.Module)** that calculates the resizing weights once on initialization and then applies them at each network pass. This also means resizing can be performed efficiently on **GPU** and on batches of images.

7. On the other hand, you can use a **dynamic resize** function to scale to different scal-factors at each pass. Both ways built upon the same building-blocks and produce identical results. Both are differntiable and can be used inside a neural network.

8. Differently from some existing methods, including MATLAB, You can **resize N-D tensors in M-D dimensions.** for any M<=N.

9. You can specify a list of scale-factors to resize each dimension using a different scale-factor.

10. You can easily add and embed your own interpolation methods for the resizer to use (see interp_mehods.py)

11. Some calculations are done more efficiently than the MATLAB version (one example is that MATLAB extends the kernel size by 2, and then searches for zero columns in the weights and cancels them. ResizeRight uses an observation that resizing is actually a continuous convolution and avoids having these redundancies ahead, see Shocher et al. ["From Discrete to  Continuous Convolution Layers"](https://arxiv.org/abs/2006.11120)).
--------

### Usage:
For dynamic resize using either Numpy or PyTorch:
```
resize_reight.resize(input, scale_factors=None, out_shape=None, 
                     interp_method=interp_methods.cubic, support_sz=None, 
                     antialiasing=True)
```
For a PyTorch layer (nn.Module):
```
resize_layer = resize_reight.ResizeLayer(in_shape, scale_factors=None, out_shape=None,
                                         interp_method=interp_methods.cubic, support_sz=None,
                                         antialiasing=True
                           
resize_layer(input)
```

__input__ :   
the input image/tensor, a Numpy or Torch tensor.

__in_shape__  (only specified for a static layer):   
the input tensor shape. a list of integers.

__scale_factors__:    
can be specified as-  
1. one scalar scale - then it will be assumed that you want to resize first two dims with this scale for Numpy or last two dims for PyTorch.  
2. a list or tupple of scales - one for each dimension you want to resize. note: if length of the list is L then first L dims will be rescaled for Numpy and last L for PyTorch. 
3. not specified - then it will be calculated using output_size. this is not recomended (see advantage 3 in the list above).   

__out_shape__:   
A list or tupple. if shorter than input.shape then only the first/last (depending np/torch) dims are resized. if not specified, can be calcualated from scale_factor.

__interp_method__:   
The type of interpolation used to calculate the weights. this is a scalar to scalar function that can be applied to tensors pointwise. The classical methods are implemented and can be found in interp_methods.py. (cubic, linear, laczos2, lanczos3, box).

__support_sz__:   
This is the support of the interpolation function, i.e length of non-zero segment over its 1d input domain. this is a characteristic of the function. eg. for bicubic 4, linear 2, laczos2 4, lanczos3 6, box 1.

__antialiasing__:   
This is an option similar to MATLAB's default. only relevant for downscaling. if true it basicly means that the kernel is stretched with 1/scale_factor to prevent aliasing (low-pass filtering)

--------

### Cite / credit
If you find our work useful in your research or publication, please cite this work:
```
@misc{ResizeRight,
  author = {Shocher, Assaf},
  title = {ResizeRight},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/assafshocher/ResizeRight}},
}
```
