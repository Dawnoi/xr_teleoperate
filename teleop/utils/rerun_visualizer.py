import os
import json
import cv2
import time
import rerun as rr
import rerun.blueprint as rrb
from datetime import datetime
os.environ["RUST_LOG"] = "error"


def canonical_color_key(color_key: str) -> str:
    key = str(color_key or "").strip()
    mapping = {
        "left_wrist": "wrist_left",
        "right_wrist": "wrist_right",
        "wrist_left": "wrist_left",
        "wrist_right": "wrist_right",
        "head": "head",
    }
    return mapping.get(key, key)

class RerunEpisodeReader:
    def __init__(self, task_dir = ".", json_file="data.json"):
        self.task_dir = task_dir
        self.json_file = json_file

    def return_episode_data(self, episode_idx):
        # Load episode data on-demand
        episode_dir = os.path.join(self.task_dir, f"episode_{episode_idx:04d}")
        json_path = os.path.join(episode_dir, self.json_file)

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Episode {episode_idx} data.json not found.")

        with open(json_path, 'r', encoding='utf-8') as jsonf:
            json_file = json.load(jsonf)

        episode_data = []

        # Loop over the data entries and process each one
        for item_data in json_file['data']:
            # Process images and other data
            colors = self._process_images(item_data, 'colors', episode_dir)
            depths = self._process_images(item_data, 'depths', episode_dir)
            audios = self._process_audio(item_data, 'audios', episode_dir)

            # Append the data in the item_data list
            episode_data.append(
                {
                    'idx': item_data.get('idx', 0),
                    'colors': colors,
                    'depths': depths,
                    'states': item_data.get('states', {}),
                    'actions': item_data.get('actions', {}),
                    'tactiles': item_data.get('tactiles', {}),
                    'audios': audios,
                    'timestamps': item_data.get('timestamps', {}),
                }
            )

        return episode_data

    def _process_images(self, item_data, data_type, dir_path):
        images = {}
        data_entries = item_data.get(data_type, {}) or {}
        for key, file_name in data_entries.items():
            if file_name:
                file_path = os.path.join(dir_path, file_name)
                if os.path.exists(file_path):
                    image = cv2.imread(file_path)
                    if image is not None:
                        images[key] = image
        return images

    def _process_audio(self, item_data, data_type, episode_dir):
        audio_data = {}
        dir_path = os.path.join(episode_dir, data_type)
        data_entries = item_data.get(data_type, {}) or {}
        for key, file_name in data_entries.items():
            if file_name:
                file_path = os.path.join(dir_path, file_name)
                if os.path.exists(file_path):
                    pass  # Handle audio data if needed
        return audio_data

class RerunLogger:
    def __init__(self, prefix = "", IdxRangeBoundary = 30, memory_limit = None, rrd_path = None, spawn_viewer = True):
        self.prefix = prefix
        self.IdxRangeBoundary = IdxRangeBoundary
        self.blueprint = None
        self.rrd_path = rrd_path
        self.spawn_viewer = spawn_viewer
        self.application_id = datetime.now().strftime("Runtime_%Y%m%d_%H%M%S_%f")
        self.live_recording = None
        self.file_recording = None

        # Set up blueprint for live visualization
        if self.IdxRangeBoundary:
            self.setup_blueprint()
        else:
            self.blueprint = None

        if self.spawn_viewer:
            self.live_recording = rr.new_recording(self.application_id)
            if memory_limit:
                rr.spawn(
                    recording=self.live_recording,
                    memory_limit=memory_limit,
                    hide_welcome_screen=True,
                    default_blueprint=self.blueprint,
                )
            else:
                rr.spawn(
                    recording=self.live_recording,
                    hide_welcome_screen=True,
                    default_blueprint=self.blueprint,
                )

        if self.rrd_path:
            self.file_recording = rr.new_recording(self.application_id)
            rr.save(self.rrd_path, recording=self.file_recording, default_blueprint=self.blueprint)

    def _recordings(self):
        recordings = []
        if self.live_recording is not None:
            recordings.append(self.live_recording)
        if self.file_recording is not None:
            recordings.append(self.file_recording)
        return recordings

    def _sliding_idx_time_range(self):
        return [
            rrb.VisibleTimeRange(
                "idx",
                start=rrb.TimeRangeBoundary.cursor_relative(seq=-self.IdxRangeBoundary),
                end=rrb.TimeRangeBoundary.cursor_relative(),
            )
        ]

    def setup_blueprint(self):
        joint_curve_views = []

        for plot_path in [
            f"{self.prefix}left_arm",
            f"{self.prefix}right_arm",
            f"{self.prefix}left_ee",
            f"{self.prefix}right_ee",
        ]:
            joint_curve_views.append(
                rrb.TimeSeriesView(
                    origin=plot_path,
                    name=plot_path.split("/")[-1],
                    time_ranges=self._sliding_idx_time_range(),
                    plot_legend=rrb.PlotLegend(visible=True),
                )
            )

        pose_curve_views = []
        for plot_path in [
            f"{self.prefix}left_arm_pose",
            f"{self.prefix}right_arm_pose",
        ]:
            pose_curve_views.append(
                rrb.TimeSeriesView(
                    origin=plot_path,
                    name=plot_path.split("/")[-1],
                    time_ranges=self._sliding_idx_time_range(),
                    plot_legend=rrb.PlotLegend(visible=True),
                )
            )

        head_view = rrb.Spatial2DView(
            origin=f"{self.prefix}colors/head",
            name="head_rgb",
            time_ranges=self._sliding_idx_time_range(),
        )
        left_wrist_view = rrb.Spatial2DView(
            origin=f"{self.prefix}colors/wrist_left",
            name="wrist_left_rgb",
            time_ranges=self._sliding_idx_time_range(),
        )
        right_wrist_view = rrb.Spatial2DView(
            origin=f"{self.prefix}colors/wrist_right",
            name="wrist_right_rgb",
            time_ranges=self._sliding_idx_time_range(),
        )

        curves_tabs = rrb.Tabs(
            contents=[
                rrb.Grid(contents=joint_curve_views, grid_columns=2, name="joint_curves"),
                rrb.Grid(contents=pose_curve_views, grid_columns=2, name="pose_curves"),
            ],
            active_tab=0,
            name="curves",
        )
        camera_tabs = rrb.Tabs(
            contents=[head_view, left_wrist_view, right_wrist_view],
            active_tab="head_rgb",
            name="cameras",
        )
        layout = rrb.Vertical(
            contents=[camera_tabs, curves_tabs],
            row_shares=[2, 2],
            name="teleop_recording",
        )
        self.blueprint = rrb.Blueprint(
            layout,
            rr.blueprint.SelectionPanel(state=rrb.PanelState.Collapsed),
            rr.blueprint.TimePanel(state=rrb.PanelState.Expanded),
            collapse_panels=False,
        )

    @staticmethod
    def _log_pose_series(base_path: str, pose_info: dict, recording):
        position = pose_info.get("position", []) or []
        rpy = pose_info.get("rpy", []) or []
        for axis, value in zip(("x", "y", "z"), position):
            rr.log(f"{base_path}/position/{axis}", rr.Scalar(float(value)), recording=recording)
        for axis, value in zip(("roll", "pitch", "yaw"), rpy):
            rr.log(f"{base_path}/rpy/{axis}", rr.Scalar(float(value)), recording=recording)

    def log_item_data(self, item_data: dict):
        recordings = self._recordings()
        if not recordings:
            return

        for recording in recordings:
            rr.set_time_sequence("idx", item_data.get('idx', 0), recording=recording)
            sample_ts = (((item_data.get("timestamps", {}) or {}).get("sample_monotonic_ns")))
            if sample_ts is not None:
                rr.set_time_nanos("sample_time", int(sample_ts), recording=recording)

            states = item_data.get('states', {}) or {}
            for part, state_info in states.items():
                if part != "body" and state_info:
                    values = state_info.get('qpos', [])
                    for idx, val in enumerate(values):
                        rr.log(f"{self.prefix}{part}/states/qpos/{idx}", rr.Scalar(val), recording=recording)
                    pose_info = state_info.get("pose")
                    if pose_info:
                        self._log_pose_series(f"{self.prefix}{part}_pose/states", pose_info, recording)

            actions = item_data.get('actions', {}) or {}
            for part, action_info in actions.items():
                if part != "body" and action_info:
                    values = action_info.get('qpos', [])
                    for idx, val in enumerate(values):
                        rr.log(f"{self.prefix}{part}/actions/qpos/{idx}", rr.Scalar(val), recording=recording)
                    pose_info = action_info.get("pose")
                    if pose_info:
                        self._log_pose_series(f"{self.prefix}{part}_pose/actions", pose_info, recording)

            colors = item_data.get('colors', {}) or {}
            for color_key, color_val in colors.items():
                if color_val is None:
                    continue
                if isinstance(color_val, str):
                    continue
                canonical_key = canonical_color_key(color_key)
                if hasattr(color_val, "shape"):
                    image = color_val
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    elif len(image.shape) == 3 and image.shape[2] == 4:
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                    rr.log(f"{self.prefix}colors/{canonical_key}", rr.Image(image), recording=recording)

        # # Log depths (images)
        # depths = item_data.get('depths', {}) or {}
        # for depth_key, depth_val in depths.items():
        #     if depth_val is not None:
        #         # rr.log(f"{self.prefix}depths/{depth_key}", rr.Image(depth_val))
        #         pass # Handle depth if needed

        # # Log tactile if needed
        # tactiles = item_data.get('tactiles', {}) or {}
        # for hand, tactile_vals in tactiles.items():
        #     if tactile_vals is not None:
        #         pass # Handle tactile if needed

        # # Log audios if needed
        # audios = item_data.get('audios', {}) or {}
        # for audio_key, audio_val in audios.items():
        #     if audio_val is not None:
        #         pass  # Handle audios if needed

    def log_episode_data(self, episode_data: list):
        for item_data in episode_data:
            self.log_item_data(item_data)

    def save(self, path: str):
        if self.file_recording is None:
            raise RuntimeError("RerunLogger.save() no longer supports exporting past data after logging starts. Configure rrd_path at construction time.")

    def close(self):
        for recording in self._recordings():
            try:
                recording.flush()
            except Exception:
                pass
            try:
                rr.disconnect(recording=recording)
            except Exception:
                pass


if __name__ == "__main__":
    import gdown
    import zipfile
    import os
    import logging_mp
    logger_mp = logging_mp.getLogger(__name__)

    zip_file = "rerun_testdata.zip"
    zip_file_download_url = "https://drive.google.com/file/d/1f5UuFl1z_gaByg_7jDRj1_NxfJZh2evD/view?usp=sharing"
    unzip_file_output_dir = "./testdata"
    if not os.path.exists(os.path.join(unzip_file_output_dir, "episode_0006")):
        if not os.path.exists(zip_file):
            file_id = zip_file_download_url.split('/')[5]
            gdown.download(id=file_id, output=zip_file, quiet=False)
            logger_mp.info("download ok.")
        if not os.path.exists(unzip_file_output_dir):
            os.makedirs(unzip_file_output_dir)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_file_output_dir)
        logger_mp.info("uncompress ok.")
        os.remove(zip_file)
        logger_mp.info("clean file ok.")
    else:
        logger_mp.info("rerun_testdata exits.")


    episode_reader = RerunEpisodeReader(task_dir = unzip_file_output_dir)
    # TEST EXAMPLE 1 : OFFLINE DATA TEST
    user_input = input("Please enter the start signal (enter 'off' or 'on' to start the subsequent program):\n")
    if user_input.lower() == 'off':
        episode_data6 = episode_reader.return_episode_data(6)
        logger_mp.info("Starting offline visualization...")
        offline_logger = RerunLogger(prefix="offline/")
        offline_logger.log_episode_data(episode_data6)
        logger_mp.info("Offline visualization completed.")

    # TEST EXAMPLE 2 : ONLINE DATA TEST, SLIDE WINDOW SIZE IS 60, MEMORY LIMIT IS 50MB
    if user_input.lower() == 'on':
        episode_data8 = episode_reader.return_episode_data(8)
        logger_mp.info("Starting online visualization with fixed idx size...")
        online_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit='50MB')
        for item_data in episode_data8:
            online_logger.log_item_data(item_data)
            time.sleep(0.033) # 30hz
        logger_mp.info("Online visualization completed.")


    # # TEST DATA OF data_dir
    # data_dir = "./data"
    # episode_data_number = 10
    # episode_reader2 = RerunEpisodeReader(task_dir = data_dir)
    # user_input = input("Please enter the start signal (enter 'on' to start the subsequent program):\n")
    # episode_data8 = episode_reader2.return_episode_data(episode_data_number)
    # if user_input.lower() == 'on':
    #     # Example 2: Offline Visualization with Fixed Time Window
    #     logger_mp.info("Starting offline visualization with fixed idx size...")
    #     online_logger = RerunLogger(prefix="offline/", IdxRangeBoundary = 60)
    #     for item_data in episode_data8:
    #         online_logger.log_item_data(item_data)
    #         time.sleep(0.033) # 30hz
    #     logger_mp.info("Offline visualization completed.")
