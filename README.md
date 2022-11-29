<div align="center">

<h2>VideoReTalking <br/> <span style="font-size:12px">Audio-based Lip Synchronization for Talking Head Video Editing In the Wild</span> </h2> 

  <a href='https://arxiv.org/abs/2211.14758'><img src='https://img.shields.io/badge/ArXiv-2211.14758-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='https://vinthony.github.io/video-retalking/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>

<div>
    <a target='_blank'>Kun Cheng <sup>*,1,2</sup> </a>&emsp;
    <a href='https://vinthony.github.io/' target='_blank'>Xiaodong Cun <sup>*,2</a>&emsp;
    <a href='https://yzhang2016.github.io/yongnorriszhang.github.io/' target='_blank'>Yong Zhang <sup>2</sup></a>&emsp;
    <a href='https://menghanxia.github.io/' target='_blank'>Menghan Xia <sup>2</sup></a>&emsp;
    <a target='_blank'>Fei Yin <sup>2,3</sup></a>&emsp;<br/>
    <a target='_blank'>Mingrui Zhu <sup>1</sup></a>&emsp;
    <a href='https://xuanwangvc.github.io/' target='_blank'>Xuan Wang <sup>2</sup></a>&emsp;
    <a href='https://juewang725.github.io/' target='_blank'>Jue Wang <sup>2</sup></a>&emsp;
    <a target='_blank'>Nannan Wang <sup>1</sup></a>
</div>
<br>
<div>
    <sup>1</sup> Xidian University &emsp; <sup>2</sup> Tencent AI Lab &emsp; <sup>3</sup> Tsinghua University
</div>
<br>
<i><strong><a href='https://sa2022.siggraph.org/' target='_blank'>SIGGRAPH Asia 2022 Conferenence Track</a></strong></i>
<br>
<br>
<img src="./images/teaser.png?raw=true" width="768px">

<div align="justify"> We present VideoReTalking, a new system to edit the faces of a real-world talking head video according to input audio, producing a high-quality and lip-syncing output video even with a different emotion. Our system disentangles this objective into three sequential tasks: (1) face video generation with a canonical expression; (2) audio-driven lip-sync; and (3) face enhancement for improving photo-realism. Given a talking-head video, we first modify the expression of each frame according to the same expression template using the expression editing network, resulting in a video with the canonical expression. This video, together with the given audio, is then fed into the lip-sync network to generate a lip-syncing video. Finally, we improve the photo-realism of the synthesized faces through an identity-aware face enhancement network and post-processing. We use learning-based approaches for all three steps and all our modules can be tackled in a sequential pipeline without any user intervention.</div>
<br>

<p>
<img alt='pipeline' src="./images/pipeline.png?raw=true" width="768px"><br>
<em align='center'>Pipeline</em>
</p>

</div>


## **Citation**

If you find our work useful in your research, please consider citing:

```
@misc{cheng2022videoretalking,
        title={VideoReTalking: Audio-based Lip Synchronization for Talking Head Video Editing In the Wild}, 
        author={Kun Cheng and Xiaodong Cun and Yong Zhang and Menghan Xia and Fei Yin and Mingrui Zhu and Xuan Wang and Jue Wang and Nannan Wang},
        year={2022},
        eprint={2211.14758},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
  }
```

