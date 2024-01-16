## X-CAUNET: Cross-Channel Attention with Underwater Image-Enhancement Transformer
- This paper has been accepted in ICASSP 2024.
## Checkpoints
- You can download the trained models from [here](https://drive.google.com/drive/folders/1pKXJ2kaYg1DrjNUvagyk3BAAROn_4wWx?usp=drive_link).
## Datasets
  - [**UIEB**](https://li-chongyi.github.io/proj_benchmark.html).
  - [**SUIM-E**](https://drive.google.com/drive/folders/1gA3Ic7yOSbHd3w214-AgMI9UleAt4bRM).
  - Please change the line no. 31 on dataset.py for differnet datasets.
```
self.filesA, self.filesB = self.get_file_paths(self.data_path, 'UIEB') ---> UIEB or SUIM
```
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
```
python test.py
```
## Citation

## Acknowledgements
- https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration
