# EleFourierTutorial timing

## Operations

The tutorial consists in a convolution in Fourier space of N real-valued images by a real-valued filter.
A dummy transform immediately followed by the inverse transform is introduced, just to test complex DFTs.

- Filter -> Transform -> Complex half-plane
- Image stack -> Transform -> Complex half-cube -> Transform -> Dummy complex half-cube -> Inverse transform -> Normalize -> Complex half-cube
- Complex half-plane & Complex half-cube -> Multiply element-wise -> Complex half-cube
- Complex half-cube -> Inverse transform -> Normalize -> Image stack

## Results

|	Operation	|	Type	|	Size	|	Duration	|
| --------- | ----  | ----  | --------  |
|	Forward planning	|	Real	|	1024x1024	|	378ms	|
|	Forward planning	|	Real	|	1024x1024x10	|	0ms	|
|	Backward planning	|	Real	|	1024x1024x10	|	329ms	|
|	Forward planning	|	Complex	|	513x1024x10	|	3ms	|
|	Backward planning	|	Complex	|	513x1024x10	|	206ms	|
|	Forward transform	|	Real	|	1024x1024	|	5ms	|
|	Forward transform	|	Real	|	1024x1024x10	|	96ms	|
|	Forward transform	|	Complex	|	513x1024x10	|	148ms	|
|	Backward transform	|	Complex	|	513x1024x10	|	99ms	|
|	Normalization	|	Complex	|	513x1024x10	|	15ms	|
|	Element-wise multiplication	|	Complex	|	513x1024x10	|	20ms	|
|	Backward transform	|	Real	|	1024x1024x10	|	60ms	|
|	Normalization	|	Real	|	1024x1024x10	|	15ms	|
