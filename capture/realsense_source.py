# capture/realsense_source.py
import time
import numpy as np
import pyrealsense2 as rs

from capture.record import Frame


class RealSenseFrameSource:
    def __init__(self, width=1280, height=720, fps=30, lock_exposure_us: int | None = 8000):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(config)

        color_sensor = profile.get_device().first_color_sensor()
        exposure_mode = "auto"
        if lock_exposure_us is not None:
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
            color_sensor.set_option(rs.option.exposure, lock_exposure_us)
            exposure_mode = "locked"

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        depth_sensor = profile.get_device().first_depth_sensor()
        self.intrinsics = {
            "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
            "width": intr.width, "height": intr.height,
            "depth_scale": depth_sensor.get_depth_scale(),
            "exposure_mode": exposure_mode,
        }
        self._align = rs.align(rs.stream.color)

    def frames(self):
        index = 0
        start = time.monotonic()
        try:
            while True:
                frameset = self.pipeline.wait_for_frames()
                frameset = self._align.process(frameset)
                color_frame = frameset.get_color_frame()
                depth_frame = frameset.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                color = np.asanyarray(color_frame.get_data())[:, :, ::-1]  # BGR -> RGB
                depth = np.asanyarray(depth_frame.get_data())
                yield Frame(index=index, timestamp=time.monotonic() - start, color=color, depth=depth)
                index += 1
        finally:
            self.pipeline.stop()
