# Tackling fluffy clouds: field boundaries detection using time series of S2 and/or S1 imagery
Official repository for our manuscript (published in RSE) [Tackling fluffy clouds: robust agricultural field boundary delineation from Sentinel-1 and Sentinel-2 satellite image time series](https://www.sciencedirect.com/science/article/pii/S0034425726004657) - [ArxivVersion](https://arxiv.org/abs/2409.13568). 

# Model Brief: 3D Vision Transformer for Field Boundary Delineation

This repository hosts the implementation of a 3D Vision Transformer designed for efficient field boundary delineation using time series satellite imagery. The model leverages spatio-temporal correlations to improve accuracy and robustness, particularly in challenging conditions such as partial cloud cover.

## Key Features:
- **3D Vision Transformer Architecture**: Adapted for handling time series data, the model processes either Sentinel-2 (S2) optical imagery or Sentinel-1 (S1) SAR data, or a combination of both, through a memory-efficient attention mechanism.
- **Handling Cloud Contamination**: The model is capable of extracting field boundaries even from cloud-contaminated imagery by leveraging S2 time series. For dense cloud coverage, the model effectively switches to S1 data, which is unaffected by clouds.
- **Dual Implementation**: The repository provides two models: 
  - **PTAViT3D**: Processes either S2 or S1 time series independently.
  - **PTAViT3D-CA**: A cross-attention model designed for fusing S2 and S1 time series data.
- **High-Resolution Predictions**: When trained on S1 inputs, the model achieves spatial resolution comparable to S2 (10m), offering precise boundary delineation.
- **Extensive Coverage**: Demonstrated on the large-scale agricultural area in Australia, showcasing the model's scalability and efficiency.

## Results:
### Example  of inference using time series of S2 imagery.
<div align="center">
<img src="images/54hxe_cloud_20_40.png" alt="Example S2, 1" width="800"/>         
</div>
<div align="center">
<img src="images/demo_cloud_inf.png" alt="Example S2, 2" width="800"/>
</div>

### Example of inference using time series of S1 imagery.        
<div align="center">
<img src="images/demo_s1_inf.png" alt="Example S1, 1" width="800"/>
</div>


## Software envinment    
Container ready for use can be found on      
+ NVIDIA: docker pull fdiakogiannis/trchprosthesis_requirements:24.07-py3



## Model forward 
We provide demo notebooks that show how the models, PTAViT3D and PTAViT3D-CA can be used (demo forward). Please see the [ssg2](https://github.com/feevos/ssg2) repository for further details on RocksDB dataset creation and additional information. We recommend using the dataset [ai4boundaries](https://github.com/waldnerf/ai4boundaries/tree/main) for your experiments, that provides time series of Sentinel2 images and corresponding ground truth labels.  **Update**: added demo training notebook for the PTAViT3D model too.



## A song too?   
In the era of AI, we decided to make a song for our paper using suno. You can listen to it [here](https://suno.com/song/3ff72217-2a85-4af2-87d5-9f5b50c9c68c).

## Lyrics
**Verse 1**
Fluffy clouds up in the sky  
Drawing maps from way up high  
Boundaries in the field we find  
With time and space we're intertwined  

**Verse 2**
Satellite's eye sees far and wide  
SAR and S2A both provide  
Imagery that guides our way  
To detect the lines we survey  

**Chorus**
Oh fluffy clouds they show the route  
Through fields where patterns sprout  
With every image scanned and seen  
We paint the picture on the screen  

**Verse 3**
In the fields the borders blur  
Technology makes them clearer  
In the waves and colors found  
We trace the lines upon the ground  

**Verse 4**
From the skies we gather clues  
Through the earth's ever-changing hues  
Data streams like melodies  
Unlock the secrets of the trees  

**Bridge**
Oh the future's looking bright  
With every satellite in flight  
Mapping out the earth below  
In fields of green and golden glow  



# License
CSIRO MIT/BSD LICENSE

As a condition of this licence, you agree that where you make any adaptations, modifications, further developments, or additional features available to CSIRO or the public in connection with your access to the Software, you do so on the terms of the BSD 3-Clause Licence template, a copy available at: http://opensource.org/licenses/BSD-3-Clause.

**If you find this repository helpful please star it as this helps us continue our work. Thank you.**

# CITATION     
```
@article{DIAKOGIANNIS2026115695,
	abstract = {Accurate delineation of agricultural field boundaries is essential for effective crop monitoring and resource management. However, competing methodologies often face significant challenges, particularly in their reliance on extensive manual efforts for cloud-free data curation and their limited adaptability to diverse global conditions. In this paper, we introduce PTAViT3D, a deep learning architecture specifically designed for processing three-dimensional time series of satellite imagery from either Sentinel-1 (S1) or Sentinel-2 (S2). Additionally, we present PTAViT3D-CA, an extension of the PTAViT3D model incorporating cross-attention mechanisms to fuse S1 and S2 datasets, enhancing robustness in cloud-contaminated scenarios. The proposed methods leverage spatio-temporal correlations through a memory-efficient 3D Vision Transformer architecture, facilitating accurate boundary delineation directly from preprocessed, cloud-affected imagery. We comprehensively validate our models through extensive testing on various datasets, including Australia’s ePaddocks™ – CSIRO’s national, continental-scale agricultural field boundary product covering Australia’s cropping regions – alongside public benchmarks Fields-of-the-World, PASTIS, and AI4SmallFarms. Our results consistently demonstrate state-of-the-art performance, highlighting excellent global transferability and robustness. Crucially, our approach significantly simplifies data preparation workflows by reliably processing cloud-affected imagery, thereby offering strong adaptability across diverse agricultural environments. Our code and models are publicly available at https://github.com/feevos/tfcl.},
	author = {Foivos I. Diakogiannis and Zheng-Shu Zhou and Jeff Wang and Gonzalo Mata and Dave Henry and Roger Lawes and Amy Parker and Peter Caccetta and Suzanne Furby and Rodrigo Ibata and Ondrej Hlinka and Jonathan Richetti and Kathryn Batchelor and Chris Herrmann and Andrew Toovey and John Taylor},
	doi = {10.1016/j.rse.2026.115695},
	issn = {0034-4257},
	journal = {Remote Sensing of Environment},
	keywords = {Agricultural field delineation, Agricultural parcel segmentation, Satellite image time series, Semantic segmentation, Vision transformer, Multisensor data fusion, Cloud contamination},
	pages = {115695},
	title = {Tackling fluffy clouds: robust agricultural field boundary delineation from Sentinel-1 and Sentinel-2 satellite image time series},
	url = {https://www.sciencedirect.com/science/article/pii/S0034425726004657},
	volume = {347},
	year = {2026}
}
```
