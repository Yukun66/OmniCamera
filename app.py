import json
import os
import tempfile
import threading
from pathlib import Path

import gradio as gr
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image, ImageOps

from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline


BASE_MODEL_REPO = os.getenv("OMNICAMERA_BASE_MODEL", "Wan-AI/Wan2.2-TI2V-5B")
OMNICAMERA_REPO = os.getenv("OMNICAMERA_MODEL_REPO", "wykup316/OmniCamera")
CHECKPOINT_NAME = os.getenv(
    "OMNICAMERA_CHECKPOINT", "joint_params_step61000.pth"
)
MODEL_VERSION = "3.0"
WIDTH, HEIGHT, DEFAULT_NUM_FRAMES, FPS = 1248, 704, 41, 16
# Wan2.2 builds the i2v timestep sequence assuming the reference-camera latent
# has the same token grid as the generated/content latent. The older CLI defaults
# (416 x 240) only work in the plain r2v_t2v branch and break r2v_i2v.
CAMERA_WIDTH, CAMERA_HEIGHT = WIDTH, HEIGHT

CAMERA_CONDITION_TYPES = ["Text motion", "Trajectory", "Reference camera video"]
CONTENT_CONDITION_TYPES = ["Text", "Image", "Video"]
MODE_TABLE = {
    ("Text motion", "Text"): "t2v",
    ("Text motion", "Image"): "i2v",
    ("Text motion", "Video"): "v2v",
    ("Trajectory", "Text"): "traj2v_t2v",
    ("Trajectory", "Image"): "traj2v_i2v",
    ("Trajectory", "Video"): "traj2v_v2v",
    ("Reference camera video", "Text"): "r2v_t2v",
    ("Reference camera video", "Image"): "r2v_i2v",
    ("Reference camera video", "Video"): "r2v_v2v",
}
CAMERA_MOTIONS = {
    "Truck Right": "trucks right",
    "Truck Left": "trucks left",
    "Dolly In": "dolly in",
    "Dolly Out": "dolly out",
    "Pan Right": "pans right",
    "Pan Left": "pans left",
    "Arc Shot": "moves in an arc trajectory left while panning right",
    "Orbiting Shot": "moves in an arc trajectory right while panning left",
    "Orbiting Forward-Up": "moves in an arc trajectory forward and up while tilting down",
    "Orbiting Right-Up": "moves in an arc trajectory forward and up-right while tilting down and panning left",
    "Boom Up and Truck Left": "booms up and trucks left",
    "Tilt Up": "tilting up",
    "Tilt Down": "tilting down",
    "Boom Up and Tilt Down": "booms up and tilts down",
    "Roll Clockwise": "rolls clockwise, speed slow",
    "Roll Counterclockwise": "rolls counterclockwise, speed slow",
    "Boom Up and Pan Left": "booms up and pans left",
    "Truck Right and Tilt Up": "trucks right and tilts up",
    "Truck Left and Tilt Down": "trucks left and tilts down",
    "Orbiting Left-Up": "moves in an arc trajectory forward and up-left while tilting down and panning right",
    "Orbiting Right-Down": "moves in an arc trajectory forward and down-right while tilting up and panning left",
    "Orbiting Left-Down": "moves in an arc trajectory forward and down-left while tilting up and panning right",
    "Tilt Up and Dolly In": "tilts up and dolly in",
    "Zoom Out and Dolly In": "zooms out and dolly in",
    "Tilt Up and Pan Left": "tilts up and pans left",
    "Tilt Up and Pan Right": "tilts up and pans right",
    "Tilt Down and Pan Left": "tilts down and pans left",
    "Tilt Down and Pan Right": "tilts down and pans right",
}
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
    "背景人很多，倒着走，[UE5 virtual scene dummy, the movements are stiff and fake]"
)

_pipeline = None
_pipeline_lock = threading.Lock()


class Camera:
    def __init__(self, c2w):
        self.c2w_mat = np.asarray(c2w, dtype=np.float32).reshape(4, 4)
        self.w2c_mat = np.linalg.inv(self.c2w_mat)


def _add_dynamic_modules(model):
    """Recreate trajectory modules from OmniCamera architecture version 3.0."""
    dim = model.blocks[0].self_attn.q.weight.shape[0]
    for block_id, block in enumerate(model.blocks):
        block.version = MODEL_VERSION
        if block_id % 5 == 0:
            block.cam_encoder = nn.Linear(12, dim)
            block.projector = nn.Linear(dim, dim)
            nn.init.zeros_(block.cam_encoder.weight)
            nn.init.zeros_(block.cam_encoder.bias)
            nn.init.eye_(block.projector.weight)
            nn.init.zeros_(block.projector.bias)


def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        if not torch.cuda.is_available():
            raise RuntimeError("This Space needs a CUDA GPU for generation.")
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("Add HF_TOKEN as a Space secret to read the private checkpoint.")

        cache_dir = os.getenv("HF_HOME")
        base_dir = snapshot_download(
            repo_id=BASE_MODEL_REPO,
            cache_dir=cache_dir,
            allow_patterns=[
                "models_t5_umt5-xxl-enc-bf16.pth",
                "diffusion_pytorch_model*.safetensors",
                "Wan2.2_VAE.pth",
                "google/umt5-xxl/**",
            ],
        )
        checkpoint_path = hf_hub_download(
            repo_id=OMNICAMERA_REPO,
            filename=CHECKPOINT_NAME,
            token=token,
            cache_dir=cache_dir,
        )
        dit_files = sorted(Path(base_dir).glob("diffusion_pytorch_model*.safetensors"))
        if not dit_files:
            raise RuntimeError("Wan2.2 DiT files were not found after download.")

        pipe = WanVideoPipeline.from_pretrained(
            device="cuda",
            torch_dtype=torch.bfloat16,
            model_configs=[
                ModelConfig(path=str(Path(base_dir) / "models_t5_umt5-xxl-enc-bf16.pth")),
                ModelConfig(path=[str(path) for path in dit_files]),
                ModelConfig(path=str(Path(base_dir) / "Wan2.2_VAE.pth")),
            ],
            tokenizer_config=ModelConfig(path=str(Path(base_dir) / "google" / "umt5-xxl")),
        )
        _add_dynamic_modules(pipe.dit)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True, mmap=True
        )
        model_keys = pipe.dit.state_dict().keys()
        filtered = {
            (key[4:] if key.startswith("dit.") else key): value
            for key, value in checkpoint.items()
            if (key[4:] if key.startswith("dit.") else key) in model_keys
        }
        if len(filtered) != 624:
            raise RuntimeError(
                "Checkpoint/architecture mismatch: expected 624 tensors, "
                f"matched {len(filtered)}."
            )
        result = pipe.dit.load_state_dict(filtered, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"Unexpected checkpoint keys: {result.unexpected_keys[:5]}")
        del checkpoint, filtered

        pipe.to(device="cuda", dtype=torch.bfloat16)
        pipe.eval()
        for parameter in pipe.parameters():
            parameter.requires_grad_(False)
        torch.cuda.empty_cache()
        _pipeline = pipe
        return _pipeline


def _center_crop(image, width, height):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image).astype(np.uint8))
    return ImageOps.fit(
        image.convert("RGB"), (width, height), method=Image.Resampling.BILINEAR
    )


def _video_tensor(path, width, height, num_frames=DEFAULT_NUM_FRAMES, fps=FPS):
    if not path:
        raise gr.Error("Please upload the required video condition.")
    reader = imageio.get_reader(path)
    try:
        metadata = reader.get_meta_data()
        source_fps = float(metadata.get("fps") or fps)
        try:
            total_frames = int(reader.count_frames())
        except Exception:
            total_frames = 0
        if total_frames > 0:
            usable = min(total_frames, max(1, round(source_fps * num_frames / fps)))
            indices = np.linspace(0, usable - 1, min(num_frames, usable)).astype(int)
            arrays = [reader.get_data(int(index)) for index in indices]
        else:
            frames_all = list(reader)
            if not frames_all:
                raise gr.Error("The uploaded video contains no readable frames.")
            usable = min(len(frames_all), max(1, round(source_fps * num_frames / fps)))
            indices = np.linspace(0, usable - 1, min(num_frames, usable)).astype(int)
            arrays = [frames_all[index] for index in indices]
    finally:
        reader.close()
    frames = [_center_crop(Image.fromarray(frame), width, height) for frame in arrays]
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    array = np.stack([np.asarray(frame, dtype=np.float32) for frame in frames[:num_frames]])
    tensor = torch.from_numpy(array).permute(3, 0, 1, 2) / 127.5 - 1.0
    return tensor.unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)


def _parse_matrix(value):
    if isinstance(value, str):
        rows = value.strip().split("] [")
        value = [
            [float(item) for item in row.replace("[", "").replace("]", "").split()]
            for row in rows
        ]
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.size != 16:
        raise gr.Error("Each trajectory pose must contain a 4 x 4 camera matrix.")
    return matrix.reshape(4, 4)


def _relative_pose(first_pose, current_pose):
    first = Camera(first_pose)
    current = Camera(current_pose)
    target_origin = np.eye(4, dtype=np.float32)
    return (target_origin @ first.w2c_mat @ current.c2w_mat)[:3, :]


def _trajectory_tensor(path, camera_id, num_frames):
    if not path:
        raise gr.Error(
            "Trajectory modes require the camera-pose JSON used to render the "
            "trajectory image. A PNG alone does not contain the required 3D poses."
        )
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        poses = [_parse_matrix(item) for item in data]
    elif isinstance(data, dict):
        frame_keys = sorted(
            (key for key in data if key.lower().startswith("frame")),
            key=lambda key: int("".join(ch for ch in key if ch.isdigit()) or 0),
        )
        if not frame_keys:
            raise gr.Error("No frame0, frame1, ... entries were found in the JSON.")
        requested_key = f"cam{int(camera_id):02d}"
        poses = []
        for frame_key in frame_keys:
            frame = data[frame_key]
            if isinstance(frame, dict):
                if requested_key in frame:
                    value = frame[requested_key]
                elif len(frame) == 1:
                    value = next(iter(frame.values()))
                else:
                    available = ", ".join(sorted(frame)[:8])
                    raise gr.Error(
                        f"{requested_key} is missing from {frame_key}. "
                        f"Available cameras: {available}"
                    )
            else:
                value = frame
            poses.append(_parse_matrix(value))
    else:
        raise gr.Error("Unsupported trajectory JSON structure.")
    if len(poses) < 2:
        raise gr.Error("The trajectory JSON must contain at least two poses.")

    if len(poses) < num_frames:
        raise gr.Error(
            f"The trajectory contains {len(poses)} poses, but {num_frames} "
            "source poses are required."
        )
    # Match the training loader exactly: use the first N source poses, then
    # sample every fourth pose to align with the temporal video latent.
    sampled = poses[:num_frames:4]
    converted = []
    for pose in sampled:
        # The training/inference loader transposes the serialized UE matrix first.
        pose = pose.T.copy()[:, [1, 2, 0, 3]]
        pose[:3, 1] *= -1.0
        pose[:3, 3] /= 100.0
        converted.append(pose)
    relative = np.stack([_relative_pose(converted[0], pose) for pose in converted])
    tensor = torch.from_numpy(relative.reshape(len(relative), 12))
    return tensor.unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)


def _prompt_for_mode(scene_prompt, camera_type, camera_motion):
    prompt = scene_prompt.strip()
    if camera_type != "Text motion":
        return prompt
    return (
        f"[Camera Movement: {camera_motion}] "
        f"The camera is {CAMERA_MOTIONS[camera_motion]}. {prompt}"
    )


def _cfg_for_mode(mode):
    """Match the hierarchical-CFG profiles in the original inference scripts."""
    if mode == "r2v_v2v":
        return True, 3.0, 5.0
    if mode.startswith("traj2v_") or mode.startswith("r2v_"):
        return True, 5.0, 3.0
    return False, None, None


def generate_video(
    camera_type,
    content_type,
    camera_motion,
    trajectory_json,
    trajectory_camera_id,
    reference_camera_video,
    scene_prompt,
    content_image,
    content_video,
    seed,
    steps,
    num_frames,
):
    if not scene_prompt or not scene_prompt.strip():
        raise gr.Error("Please enter a content prompt for the generated video.")
    mode = MODE_TABLE[(camera_type, content_type)]
    if content_type == "Image" and content_image is None:
        raise gr.Error("Image-content mode requires an input image.")
    if content_type == "Video" and not content_video:
        raise gr.Error("Video-content mode requires an input video.")
    if camera_type == "Reference camera video" and not reference_camera_video:
        raise gr.Error("Reference-camera mode requires a camera-motion video.")

    try:
        pipe = _load_pipeline()
        num_frames = int(num_frames)
        if num_frames not in (41, 81):
            raise gr.Error("Frame count must be either 41 or 81.")
        use_hierarchical_cfg, text_cfg_scale, camera_cfg_scale = _cfg_for_mode(mode)
        kwargs = {
            "model_mode": mode,
            "prompt": _prompt_for_mode(scene_prompt, camera_type, camera_motion),
            "negative_prompt": NEGATIVE_PROMPT,
            "seed": int(seed),
            "tiled": True,
            "height": HEIGHT,
            "width": WIDTH,
            "num_frames": num_frames,
            "num_inference_steps": int(steps),
            "cfg_scale": 5.0,
            "use_hierarchical_cfg": use_hierarchical_cfg,
            "text_cfg_scale": text_cfg_scale,
            "camera_cfg_scale": camera_cfg_scale,
        }
        if camera_type == "Trajectory":
            kwargs["camera_info"] = _trajectory_tensor(
                trajectory_json, trajectory_camera_id, num_frames
            )
        elif camera_type == "Reference camera video":
            kwargs["camera_cond_video"] = _video_tensor(
                reference_camera_video, CAMERA_WIDTH, CAMERA_HEIGHT, num_frames
            )
        if content_type == "Image":
            kwargs["input_image"] = _center_crop(content_image, WIDTH, HEIGHT)
        elif content_type == "Video":
            kwargs["frame_cond_video"] = _video_tensor(
                content_video, WIDTH, HEIGHT, num_frames
            )

        frames = pipe(**kwargs)
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        imageio.mimsave(
            output_path,
            [np.asarray(frame) for frame in frames],
            fps=FPS,
            codec="libx264",
            quality=5,
        )
        summary = (
            f"Mode: {mode}\nCamera condition: {camera_type}\n"
            f"Content condition: {content_type}\nFrames: {num_frames}\n"
            f"Steps: {int(steps)}\nHierarchical CFG: {use_hierarchical_cfg}\n"
            f"Text CFG: {text_cfg_scale}\nCamera CFG: {camera_cfg_scale}\n"
            f"Prompt: {kwargs['prompt']}"
        )
        return output_path, summary
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc


def update_condition_inputs(camera_type, content_type):
    return (
        gr.update(visible=camera_type == "Text motion"),
        gr.update(visible=camera_type == "Trajectory"),
        gr.update(visible=camera_type == "Trajectory"),
        gr.update(visible=camera_type == "Trajectory"),
        gr.update(visible=camera_type == "Reference camera video"),
        gr.update(visible=content_type == "Image"),
        gr.update(visible=content_type == "Video"),
        f"**Selected task:** `{MODE_TABLE[(camera_type, content_type)]}`",
    )


with gr.Blocks(title="OmniCamera Demo") as demo:
    gr.Markdown(
        """
        # OmniCamera
        **Multi-task video generation with arbitrary camera control**

        Select one of three camera conditions and one of three content conditions.
        All **3 × 3 = 9** combinations use the same joint checkpoint.

        > A trajectory image is a visualization. Trajectory inference requires
        > the corresponding camera-pose JSON because the model consumes 3D poses.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            camera_type = gr.Radio(
                CAMERA_CONDITION_TYPES, value="Text motion", label="Camera condition"
            )
            content_type = gr.Radio(
                CONTENT_CONDITION_TYPES, value="Text", label="Content condition"
            )
            selected_mode = gr.Markdown("**Selected task:** `t2v`")
            camera_motion = gr.Dropdown(
                choices=list(CAMERA_MOTIONS),
                value="Dolly In",
                label="Camera-motion text",
            )
            trajectory_json = gr.File(
                label="Trajectory camera poses (JSON)",
                file_types=[".json"],
                type="filepath",
                visible=False,
            )
            trajectory_preview = gr.Image(
                label="Trajectory visualization (optional; display only)",
                type="pil",
                visible=False,
            )
            trajectory_camera_id = gr.Number(
                value=1,
                precision=0,
                minimum=0,
                label="Camera index inside JSON",
                visible=False,
            )
            reference_camera_video = gr.Video(
                label="Reference camera-motion video", visible=False
            )
            scene_prompt = gr.Textbox(
                value="A cinematic coastal city at sunset, realistic details.",
                label="Content prompt (used by all modes)",
                lines=4,
            )
            content_image = gr.Image(label="Content image", type="pil", visible=False)
            content_video = gr.Video(label="Content video", visible=False)
            seed = gr.Number(value=0, precision=0, label="Seed")
            num_frames = gr.Radio(
                choices=[41, 81], value=41, label="Output frames"
            )
            steps = gr.Slider(10, 50, value=50, step=1, label="Inference steps")
            generate = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            output = gr.Video(label="Generated video", format="mp4")
            run_summary = gr.Textbox(label="Task summary", lines=6)

    condition_outputs = [
        camera_motion,
        trajectory_json,
        trajectory_preview,
        trajectory_camera_id,
        reference_camera_video,
        content_image,
        content_video,
        selected_mode,
    ]
    camera_type.change(
        update_condition_inputs,
        inputs=[camera_type, content_type],
        outputs=condition_outputs,
    )
    content_type.change(
        update_condition_inputs,
        inputs=[camera_type, content_type],
        outputs=condition_outputs,
    )
    generate.click(
        fn=generate_video,
        inputs=[
            camera_type,
            content_type,
            camera_motion,
            trajectory_json,
            trajectory_camera_id,
            reference_camera_video,
            scene_prompt,
            content_image,
            content_video,
            seed,
            steps,
            num_frames,
        ],
        outputs=[output, run_summary],
    )
    gr.Markdown(
        "Generation supports 41 or 81 frames at 16 FPS and uses the trained "
        "1248 × 704 resolution. Camera-controlled modes follow the original "
        "hierarchical-CFG inference profiles.  "
        "[Paper](https://arxiv.org/abs/2604.06010) · "
        "[Project page](https://yukun66.github.io/omnicamera-webdemo/)"
    )

if __name__ == "__main__":
    demo.queue(max_size=5, default_concurrency_limit=1).launch()
