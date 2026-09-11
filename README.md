# OmniCamera

### ECCV 2026

Official inference code for **OmniCamera: A Unified Framework for Multi-task
Video Generation with Arbitrary Camera Control**.

[Paper](https://arxiv.org/abs/2604.06010) ·
[Project Page](https://yukun66.github.io/omnicamera-webdemo/) ·
[Model Weights](https://huggingface.co/wykup316/OmniCamera)

OmniCamera provides independent control over camera motion and video content.
One model supports all nine combinations of three camera conditions and three
content conditions.

| Camera condition | Text | Image | Video |
| --- | --- | --- | --- |
| Motion text | ✓ | ✓ | ✓ |
| Camera trajectory | ✓ | ✓ | ✓ |
| Reference camera video | ✓ | ✓ | ✓ |

## Installation

Python 3.10 or 3.11 and a CUDA GPU are recommended.

```bash
git clone https://github.com/Yukun66/OmniCamera.git
cd OmniCamera
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The base model and public OmniCamera checkpoint are downloaded automatically
from Hugging Face. No access token is required.

## Inference

Run one of the provided examples:

```bash
python infer.py --config examples/text_motion_t2v.json
```

The generated video is saved in `outputs/`. Edit the JSON configuration to
change the prompt, camera condition, content input, seed, frame count, or
inference steps.

Examples for all 3 × 3 tasks are available in [`examples/`](examples/). Their
input images, videos, camera references, trajectories, and generated results
are included in [`assets/`](assets/).

Validate an example without loading the model:

```bash
python infer.py --config examples/text_motion_t2v.json --dry-run
```

See the [project page](https://yukun66.github.io/omnicamera-webdemo/) for more
video results.

## Citation

```bibtex
@article{wang2026omnicamera,
  title   = {OmniCamera: A Unified Framework for Multi-task Video Generation with Arbitrary Camera Control},
  author  = {Wang, Yukun and Li, Ruihuang and Tao, Jiale and Yang, Shiyuan and Chen, Liyi and Yang, Zhantao and Handz and Guo, Yulan and Shao, Shuai and Lu, Qinglin},
  journal = {arXiv preprint arXiv:2604.06010},
  year    = {2026}
}
```

## Acknowledgements

This repository includes modified components from
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and example
camera trajectories adapted from
[ReCamMaster](https://github.com/KwaiVGI/ReCamMaster). See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for details.

## License

The OmniCamera code is released under the [Apache License 2.0](LICENSE).
