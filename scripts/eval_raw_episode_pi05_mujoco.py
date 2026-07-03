#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from teleop.inference.raw_episode_pi05_eval import evaluate_raw_episode_pi05_mujoco


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 GT raw episode 观测驱动 pi0.5 双臂模型，并在 MuJoCo 中执行模型输出后与 GT actions.pose 比较。",
    )
    parser.add_argument("--dataset-root", required=True, help="raw episode 数据集根目录。")
    parser.add_argument("--episode-index", type=int, required=True, help="episode 索引，例如 107 对应 episode_0107。")
    parser.add_argument("--base-url", required=True, help="HTTP 推理服务地址，例如 http://127.0.0.1:8017")
    parser.add_argument("--prompt", default="", help="发送给 pi0.5 的任务 prompt。")
    parser.add_argument("--output-dir", required=True, help="输出目录，写入 summary.json 和 steps.jsonl。")
    parser.add_argument("--stride", type=int, default=10, help="锚点抽样步长。")
    parser.add_argument("--chunk-size", type=int, default=10, help="每次请求对齐的 GT 动作长度。")
    parser.add_argument("--n-obs-steps", type=int, default=2, help="观测历史长度。")
    parser.add_argument("--start-frame", type=int, default=None, help="可选，起始 frame。")
    parser.add_argument("--end-frame", type=int, default=None, help="可选，结束 frame 的上界。")
    parser.add_argument("--camera-freq", type=float, default=30.0, help="GT 观测频率。")
    parser.add_argument("--action-step-sec", type=float, default=None, help="chunk step 间隔秒数。默认使用 1/camera_freq。")
    parser.add_argument("--response-timeout-sec", type=float, default=30.0, help="HTTP 请求超时时间。")
    parser.add_argument("--max-arm-joint-speed", type=float, default=1.5, help="仅在开启关节限速时使用。")
    parser.add_argument("--apply-speed-limit", action="store_true", help="是否在 MuJoCo 执行时复用关节限速。默认关闭。")
    parser.add_argument("--save-video", default=None, help="可选，评测完成后输出 MuJoCo 执行轨迹 mp4。")
    parser.add_argument("--video-step-repeat", type=int, default=4, help="每个评测 step 在视频中重复写入多少帧。")
    parser.add_argument("--video-width", type=int, default=1280, help="导出视频宽度。")
    parser.add_argument("--video-height", type=int, default=720, help="导出视频高度。")
    parser.add_argument("--video-camera-distance", type=float, default=3.0, help="视频相机 distance。")
    parser.add_argument("--video-camera-azimuth", type=float, default=-135.0, help="视频相机 azimuth。")
    parser.add_argument("--video-camera-elevation", type=float, default=-18.0, help="视频相机 elevation。")
    parser.add_argument(
        "--video-camera-lookat",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.95],
        metavar=("X", "Y", "Z"),
        help="视频相机 lookat。",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary, _, summary_path, steps_path = evaluate_raw_episode_pi05_mujoco(
        dataset_root=args.dataset_root,
        episode_index=args.episode_index,
        base_url=args.base_url,
        prompt=args.prompt,
        output_dir=args.output_dir,
        stride=args.stride,
        chunk_size=args.chunk_size,
        n_obs_steps=args.n_obs_steps,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        camera_freq=args.camera_freq,
        action_step_sec=args.action_step_sec,
        response_timeout_sec=args.response_timeout_sec,
        max_arm_joint_speed=args.max_arm_joint_speed,
        apply_speed_limit=args.apply_speed_limit,
        save_video=args.save_video,
        video_step_repeat=args.video_step_repeat,
        video_width=args.video_width,
        video_height=args.video_height,
        video_camera_distance=args.video_camera_distance,
        video_camera_azimuth=args.video_camera_azimuth,
        video_camera_elevation=args.video_camera_elevation,
        video_camera_lookat=tuple(args.video_camera_lookat),
    )
    print(
        "[RAW_PI05_EVAL] "
        f"summary_path={summary_path} steps_path={steps_path} "
        f"mean_pred_xyz_error_m={summary['mean_pred_xyz_error_m']:.6f} "
        f"mean_exec_xyz_error_m={summary['mean_exec_xyz_error_m']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
