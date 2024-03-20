## [**X-CAUNET: Cross-Channel Attention with Underwater Image-Enhancement Transformer**](https://ieeexplore.ieee.org/document/10445832)
- This paper has been accepted in **ICASSP 2024**.
## Checkpoints
- You can download the trained models from [here](https://drive.google.com/drive/folders/1pKXJ2kaYg1DrjNUvagyk3BAAROn_4wWx?usp=drive_link).
## Datasets
  - [**UIEB**](https://li-chongyi.github.io/proj_benchmark.html).
  - [**SUIM-E**](https://drive.google.com/drive/folders/1gA3Ic7yOSbHd3w214-AgMI9UleAt4bRM).
  - Please modify the line no. 31 in dataset.py for differnet datasets.
```
self.filesA, self.filesB = self.get_file_paths(self.data_path, 'UIEB') ---> UIEB or SUIM
```
  - Dataset file structure should be as follows:
```
├── UIEB/SUIM
    ├── Train
        ├── inp
            ├── *.jpg/*.png
            ├── *.jpg/*.png
            └── ...
        ├── gt
            ├── *.jpg/*.png
            ├── *.jpg/*.png
            └── ...
    ├── Test
        ├── inp
            ├── *.jpg/*.png
            ├── *.jpg/*.png
            └── ...
        ├── gt
            ├── *.jpg/*.png
            ├── *.jpg/*.png
            └── ...
```
## Train
``` 
python train.py
```
## Test
-Please modify the line no. 28 in test.py with **uieb.pt** or **suim.pt** for different test data.
```
python test.py
```
## Citation
```
@INPROCEEDINGS{10445832,
  author={Pramanick, Alik and Sarma, Sandipan and Sur, Arijit},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={X-CAUNET: Cross-Color Channel Attention with Underwater Image-Enhancing Transformer}, 
  year={2024},
  volume={},
  number={},
  pages={3550-3554},
  keywords={Correlation;Image color analysis;Message passing;Speech enhancement;Transformers;Colored noise;Image enhancement;Cross-attention;transformer;underwater image enhancement;deep learning},
  doi={10.1109/ICASSP48485.2024.10445832}}
```
## Acknowledgements
- https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration
