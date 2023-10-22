---
layout: post
title:  awosome_similarity_loss
category: ºº ı 
stickie: False
---

For awosome loss functions that is verified in pytorch.
We will make sure all the function here are listed with code.

<!--more-->
<div id="toc"></div>
<!-- csdn -->


## Loss 

- SSIM Loss
[SSIM](https://github.com/Po-Hsun-Su/pytorch-ssim)
```python
import pytorch_ssim
import torch
from torch.autograd import Variable

img1 = Variable(torch.rand(1, 1, 256, 256))
img2 = Variable(torch.rand(1, 1, 256, 256))

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()

print(pytorch_ssim.ssim(img1, img2))

ssim_loss = pytorch_ssim.SSIM(window_size = 11)

print(ssim_loss(img1, img2))
```
[may not useful](https://github.com/pranjaldatta/SSIM-PyTorch/blob/master/SSIM_notebook.ipynb)

- MSSSIM Loss
[muti scale SSIM](https://github.com/jorge-pessoa/pytorch-msssim)







