from .orbbec_session_runtime import OrbbecSession, SensorFrustumConfig


class Gemini305(OrbbecSession):
    """Gemini305 专用会话，集中维护该型号默认视锥参数。"""

    @property
    def sensor_frustum(self) -> SensorFrustumConfig:
        """视锥参数"""
        return SensorFrustumConfig(
            min_depth_mm=70.0,
            max_depth_mm=430.0,
            near_width_mm=117.0,
            near_height_mm=89.0,
            far_width_mm=839.0,
            far_height_mm=637.0,
        )


class Gemini336(OrbbecSession):
    """Gemini336 专用会话，集中维护该型号默认视锥参数。"""

    base_line_length_mm: float = 50.0
    "基线长度，单位 mm"

    @property
    def sensor_frustum(self) -> SensorFrustumConfig:
        """视锥参数"""
        return SensorFrustumConfig(
            min_depth_mm=70.0,
            max_depth_mm=430.0,
            near_width_mm=117.0,
            near_height_mm=89.0,
            far_width_mm=839.0,
            far_height_mm=637.0,
        )


class Gemini336L(OrbbecSession):
    """Gemini336L 专用会话，集中维护该型号视锥参数。"""

    base_line_length_mm: float = 95.0
    "基线长度，单位 mm"

    @property
    def sensor_frustum(self) -> SensorFrustumConfig:
        """视锥参数"""
        return SensorFrustumConfig(
            min_depth_mm=70.0,
            max_depth_mm=430.0,
            near_width_mm=117.0,
            near_height_mm=89.0,
            far_width_mm=839.0,
            far_height_mm=637.0,
        )
