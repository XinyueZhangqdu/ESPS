
## Lightweight portrait segmentation via edge_optimized_attention

## News
October 25, 2022: model codes and some results are released 🔥

October 24, 2022: Paper submission 🎉

## Abstract

With the outbreak of COVID-19 around the world, the frequency of video conferencing at home is increasing. Therefore, a segmentation architecture that can quickly carry out close-range portrait segmentation has become a current need. However, the current portrait segmentation architectures cannot meet the requirements of lightweight and edge-friendly. We built architecture with 0.06G FLOPs and 0.02M parameters to overcome this phenomenon. This lightweight architecture can be better embedded and run on mobile devices that only support CPU computing. Our network achieves an FPS of 39.02 on CPU, which is more than three times faster than other networks. In addition, we pay special attention to the enhancement of edge features. The independent edge feature enhancement is embedded, and the edge-optimized attention mechanism (EOAM) is designed to collect specific edge areas for the bottom features and the high-level features in the process of feature fusion. 

## Visual comparisons



![demo](https://user-images.githubusercontent.com/71067558/197715651-f8e72b23-d03f-4d43-bc7f-acf792718eeb.png)






