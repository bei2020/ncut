# ncut instance segmentation

## ncut
3 partitions `segment.py`
![sample3p](imgs/sample3parts.png)
 
ncut with eigensolver Lanczos `segment.py`
![lanc](imgs/butt_lanc_pair_iter2000_w30.png)
![lanc](imgs/rece_lanc_pair_iter2000_w50.png)

## edge detection
![zero crossing](imgs/reci2-2_Dog_sig2_minmax1.png)

## pixel label
`cluster.py`
![denoise](imgs/butt_denoise.png)
### binary label
`ms_bina.py`
![bina](imgs/butt_edge_bina.png)


## Ref

1. Jianbo Shi and J. Malik. Normalized cuts and image segmentation. IEEE Transactions on Pattern Analysis.
2. Yizong Cheng. Mean shift, mode seeking, and clustering. IEEE transactions on pattern analysis and machine
intelligence, 17(8):790–799, 1995.
3. Lowe D G. Distinctive image features from scale-invariant keypoints[J]. International journal of computer vision, 2004, 60(2): 91-110.
