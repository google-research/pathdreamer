# Pathdreamer: A World Model for Indoor Navigation

This repository hosts the open source code for [Pathdreamer](https://arxiv.org/abs/2105.08756), to be presented at [ICCV 2021](http://iccv2021.thecvf.com/).

[![Video Results](./video_results.gif)]({https://www.youtube.com/watch?v=StklIENGqs0} "Video Results")

[Paper](https://arxiv.org/abs/2105.08756) | [Project Webpage]()


## Setup instructions

### Environment
Set up virtualenv, and install required libraries:
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add the Pathdreamer library to PYTHONPATH:
```
export PYTHONPATH=$PYTHONPATH:/home/path/to/pathdreamer_root/
```

### Downloading Pretrained Checkpoints

We provide a pretrained checkpoint which can be acquired by running:
```
wget https://storage.googleapis.com/gresearch/pathdreamer/ckpt.tar -P data/
tar -xf data/ckpt.tar --directory data/
```

The results will be extracted to the `data/ckpt` directory. Two checkpoints are provided, one for the Stage 1 model (Structure Generator), and another for the Stage 2 model (Image Generator).

## Colab Demo

<!-- copybara:strip_begin(google-internal) -->
`Pathdreamer_Example_Colab.ipynb` [[click to launch in Google Colab]]() shows how to setup and run the pretrained Pathdreamer model for inference. It includes examples on synthesizing image sequences and continuous video sequences for arbitrary navigation trajectories.
<!-- copybara:strip_end -->

<!-- copybara:insert(google-internal)
Coming soon!
-->


## Citation

If you find this work useful, please consider citing:

```
@inproceedings{koh2021pathdreamer,
  title={Pathdreamer: A World Model for Indoor Navigation},
  author={Koh, Jing Yu and Lee, Honglak and Yang, Yinfei and Baldridge, Jason and Anderson, Peter},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

## License

Pathdreamer is released under the Apache 2.0 license. The Matterport3D dataset is governed by the
[Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf).

## Disclaimer

Not an official Google product.
