# MotionMaster: Training-free Camera Motion Transfer For Video Generation
###  [Paper](https://arxiv.org/abs/2404.15789) |   [Page](https://sjtuplayer.github.io/projects/MotionMaster/)
<!-- <br> -->
[Teng Hu](https://github.com/sjtuplayer), 
[Jiangning Zhang](https://zhangzjn.github.io/),
[Ran Yi](https://yiranran.github.io/), 
[Yating Wang](https://github.com/sjtuplayer/MotionMaster),
[Hongrui Huang](https://github.com/sjtuplayer/MotionMaster),
[Jieyu Weng](https://github.com/sjtuplayer/MotionMaster),
[Yabiao Wang](https://scholar.google.com/citations?hl=zh-CN&user=xiK4nFUAAAAJ),
 and [Lizhuang Ma](https://dmcv.sjtu.edu.cn/) 
<!-- <br> -->

![image](imgs/teaser.gif)

[//]: # ([![Alt text]&#40;imgs/Motionmaster.png&#41;]&#40;https://www.youtube.com/watch?v=o3Fk4RgWC4A&#41;)


![image](imgs/teaser.png)
## Schedule
- [ ] **Expected to release the full-version before 2024.10.10

## Prepare


### 🛠️ Prepare the environment
```
python 3.9
cuda==11.8
gcc==7.5.0
cd AnimateDiff

conda env create -f environment.yaml
conda activate MotionMaster
```


### 🍺 Checkpoint for AnimateDiff

```
Download the official checkpoint of AnimateDiff:

mkdir -p models/Motion_Module
wget -O models/Motion_Module/mm_sd_v15_v2.ckpt

mkdir -p models/DreamBooth_LoRA
wget -O models/DreamBooth_LoRA/realisticVisionV51_v51VAE.safetensors

mkdir -p models/StableDiffusion
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 models/StableDiffusion
```

### 🚀 Generating new videos based on the reference motion


#### Prepare reference video and prompts

Edit `animatediff\configs\prompts\v2\v2-1-RealisticVision.yaml` to make sure `video_name` is the file path to your reference video.



Run MotionMaster with:
```
python scripts/motionconvert.py --config configs/prompts/v2/v2-1-RealisticVision.yaml
```
The generated samples can be found in `samples/` folder.

### 🚀 One-shot camera motion disentanglement

Coming Soon.

### 🚀 Few-shot camera motion disentanglement

Coming Soon.

## Citation

If you find this code helpful for your research, please cite:

```
@misc{hu2024motionmaster,
      title={MotionMaster: Training-free Camera Motion Transfer For Video Generation}, 
      author={Teng Hu and Jiangning Zhang and Ran Yi and Yating Wang and Hongrui Huang and Jieyu Weng and Yabiao Wang and Lizhuang Ma},
      year={2024},
      eprint={2404.15789},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```