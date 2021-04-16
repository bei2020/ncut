# ncut instance segmentation

## ncut of 2 partitions and 3 partitions
`segment.py`

![sample2p](imgs/sample2parts.png)

![sample3p](imgs/sample3parts.png)

### recuresive 2 way partition
```
python recipt_seg.py
```
![recuresive 2 way partition](imgs/img6_d3_ncut.png)

## edge segmentation
``` python anisodiff2D.py ```

![3part edge seg](imgs/sample3parts_edge_seg.png)

![receipt edge seg](imgs/rece0_edgep.png)

## edge detection
![zero crossing](imgs/reci2-2_Dog_sig2_minmax1.png)

## pixel label
![label](imgs/reci_label.png)
![label](imgs/reci_label_.05.08.1.2.3_8000.png)
## ncut
`cluster.py`

![denoise](imgs/butt_denoise.png)

## Ref

1. Jianbo Shi and J. Malik. Normalized cuts and image segmentation. IEEE Transactions on Pattern Analysis.
2. Yizong Cheng. Mean shift, mode seeking, and clustering. IEEE transactions on pattern analysis and machine
intelligence, 17(8):790–799, 1995.
3. Lowe D G. Distinctive image features from scale-invariant keypoints[J]. International journal of computer vision, 2004, 60(2): 91-110.
