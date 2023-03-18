# C2F-net

### This repo is mainly focued on Three-view reconstruction.Initial idea is from this <a href="https://ieeexplore.ieee.org/document/9925147">paper.</a>

### After reading paper 	&laquo;Contrastive Multiview Coding	&raquo;,and with the knowledge of Contrastive Learning,I add a contrastive module to lead
* Pointclouds from same object with different views should still have similar semantic feature.
* Different Pointclouds in the same batch should have different semantic feature.
  
### And on airplane dataset from ShapeNet,the finnal average loss is 0.0018 but the IOU value is 0.4584.Due to the result is Pointcloud instead of voxel,IOU value seems to be an unfair metric.

### It's a trend to use Coarse-to-fine strategy and Encoder-Decoder architecture.Re-think in Encoder-Decoder,how to encode essential information to reconstruction still need to be studied.






