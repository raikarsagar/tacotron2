# Tacotron2 + Waveglow with ipex


## Setup
1. Compile pytorch from source with ipex (include numpy)
2. install python dependencies
 `pip install -r requirements_inference.txt`



## Inference demo
1. Download NVIDIA's published [Tacotron 2] model
2. Download NVIDIA's published [WaveGlow] model - Convert this model with convert.py in waveglow
3. Run `python inference_taco2_waveglow_ipex.py`


[WaveGlow]: https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF
[Tacotron 2]: https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing
[pytorch 1.0]: https://github.com/pytorch/pytorch#installation
[website]: https://nv-adlr.github.io/WaveGlow
[ignored]: https://github.com/NVIDIA/tacotron2/blob/master/hparams.py#L22
[Apex]: https://github.com/nvidia/apex
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp


