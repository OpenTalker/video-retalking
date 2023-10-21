<div align="center">
  
<h1>🎬 VideoReTalking: <br/> Audio-based Lip Synchronization for Talking Head Video Magic! 🪄</h1>

  <a href='https://arxiv.org/abs/2211.14758'><img src='https://img.shields.io/badge/📚 ArXiv-2211.14758-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='https://vinthony.github.io/video-retalking/'><img src='https://img.shields.io/badge/🚀 Project Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinthony/video-retalking/blob/main/quick_demo.ipynb)

<div>
    <a target='_blank'>🌟 Kun Cheng <sup>*,1,2</sup> </a>&emsp;
    <a href='https://vinthony.github.io/' target='_blank'>🎩 Xiaodong Cun <sup>*,2</a>&emsp;
    <a href='https://yzhang2016.github.io/yongnorriszhang.github.io/' target='_blank'>👾 Yong Zhang <sup>2</sup></a>&emsp;
    <a href='https://menghanxia.github.io/' target='_blank'>🌠 Menghan Xia <sup>2</sup></a>&emsp;
    <a href='https://feiiyin.github.io/' target='_blank'>🎨 Fei Yin <sup>2,3</sup></a>&emsp;<br/>
    <a href='https://web.xidian.edu.cn/mrzhu/en/index.html' target='_blank'>🔮 Mingrui Zhu <sup>1</sup></a>&emsp;
    <a href='https://xuanwangvc.github.io/' target='_blank'>🧙‍♂️ Xuan Wang <sup>2</sup></a>&emsp;
    <a href='https://juewang725.github.io/' target='_blank'>🌟 Jue Wang <sup>2</sup></a>&emsp;
    <a href='https://web.xidian.edu.cn/nnwang/en/index.html' target='_blank'>🪄 Nannan Wang <sup>1</sup></a>
</div>
<br>
<div>
    <sup>1</sup> Xidian University &emsp; <sup>2</sup> Tencent AI Lab &emsp; <sup>3</sup> Tsinghua University
</div>
<br>
<i><strong><a href='https://sa2022.siggraph.org/' target='_blank'>🌟 SIGGRAPH Asia 2022 Conference Track</a></strong></i>
<br>
<br>
<img src="./docs/static/images/teaser.png?raw=true" width="768px">

<div align="justify">Prepare to be amazed! 🤩 Introducing VideoReTalking, a groundbreaking system that lets you edit the faces in talking head videos, all driven by audio input. It produces high-quality, lip-synced videos, even with different emotions. Our system splits this into three enchanting tasks: (1) Face video creation with a standard expression; (2) Audio-driven lip-sync; and (3) Face enhancement for mind-blowing realism. These magic tricks are powered by learning-based approaches and can be seamlessly combined in a magical sequence without any user intervention! ✨</div>
<br>

<p>
<img alt='pipeline' src="./docs/static/images/pipeline.png?raw=true" width="768px"><br>
<em align='center'>Witness the Magic! ✨</em>
</p>

</div>

## Results in the Wild 🌍 (includes audio)
https://user-images.githubusercontent.com/4397546/224310754-665eb2dd-aadc-47dc-b1f9-2029a937b20a.mp4

## Setup Your Wizardry Workstation 🧙‍♂️

```
git clone https://github.com/vinthony/video-retalking.git
cd video-retalking
conda create -n video_retalking python=3.8
conda activate video_retalking

conda install ffmpeg

# Dive into the magic realm of PyTorch! 🔮
# Follow the magical instructions at https://pytorch.org/get-started/previous-versions/
# This spell only works with CUDA 11.1 🪄
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

## Quick Inference - Instant Wizardry! 🪄

#### Unleash the Power of Pretrained Models 🌟
Please download our [pre-trained models](https://drive.google.com/drive/folders/18rhjMpxK8LVVxf7PI6XwOidt8Vouv_H0?usp=share_link) and place them in `./checkpoints`.

<!-- We also provide some [example videos and audio](https://drive.google.com/drive/folders/14OwbNGDCAMPPdY-l_xO1axpUjkPxI9Dv?usp=share_link). Please put them in `./examples`. -->

#### Cast Your Spell

```shell
python3 inference.py \
  --face examples/face/1.mp4 \
  --audio examples/audio/1.wav \
  --outfile results/1_1.mp4
```
This script includes data preprocessing steps. You can test any talking face videos without manual alignment. But it is worth noting that DNet cannot handle extreme poses.

You can also control the expression by adding the following parameters:

```--exp_img```: Pre-defined expression template. The default is "neutral". You can choose "smile" or an image path.

```--up_face```: You can choose "surprise" or "angry" to modify the expression of the upper face with [GANimation](https://github.com/donydchen/ganimation_replicate).

## Magical Incantation 🪄

If you find our wizardry useful in your magical research, please consider citing:

```
@misc{cheng2022videoretalking,
        title={VideoReTalking: Audio-based Lip Synchronization for Talking Head Video Magic! 🪄}, 
        author={Kun Cheng and Xiaodong Cun and Yong Zhang and Menghan Xia and Fei Yin and Mingrui Zhu and Xuan Wang and Jue Wang and Nannan Wang},


        year={2022},
        eprint={2211.14758},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
  }
```

## Wizard's Gratitude 🙏

Special thanks to these sorcerers and enchanters for sharing their magic spells:

- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
- [PIRenderer](https://github.com/RenYurui/PIRender)
- [GFP-GAN](https://github.com/TencentARC/GFPGAN)
- [GPEN](https://github.com/yangxy/GPEN)
- [ganimation_replicate](https://github.com/donydchen/ganimation_replicate)
- [STIT](https://github.com/rotemtzaban/STIT)

## Explore More Magical Realms 🌌

- [StyleHEAT: One-Shot High-Resolution Editable Talking Face Generation via Pre-trained StyleGAN (ECCV 2022)](https://github.com/FeiiYin/StyleHEAT)
- [CodeTalker: Speech-Driven 3D Facial Animation with Discrete Motion Prior (CVPR 2023)](https://github.com/Doubiiu/CodeTalker)
- [SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023)](https://github.com/Winfredy/SadTalker)
- [DPE: Disentanglement of Pose and Expression for General Video Portrait Editing (CVPR 2023)](https://github.com/Carlyx/DPE)
- [3D GAN Inversion with Facial Symmetry Prior (CVPR 2023)](https://github.com/FeiiYin/SPI/)
- [T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations (CVPR 2023)](https://github.com/Mael-zys/T2M-GPT)

##  Disclaimer - Our Magic Spell! 🪄

1. Please respect the wizard's code, abide by the open-source license before using our magic.
2. Follow the intellectual property declaration - stealing magic is frowned upon.
3. Our magic works offline and doesn't collect personal information. If you use it for your spells, follow the law!
4. Don't misuse names and logos owned by Tencent or you'll have to deal with the magical consequences.
5. Use our magic responsibly and ethically. Don't use it for dark deeds.
6. Keep the magic alive! Don't spread misinformation, or engage in harmful activities.

</div>