from __future__ import annotations

_EXPORTS = {
    "AdvancedToolBox": ".advanced_tool_box",
    "SectionContentMode": ".advanced_tool_box",
    "CasiaIndicatorLight": ".casia_indicator_light",
    "CasiaInputWidgetDouble": ".casia_input_widget_double",
    "DegreeValueConverter": ".casia_degree_value_converter",
    "PointCloudNode": ".casia_tree",
    "PointCloudTreeModel": ".casia_tree",
    "CallableValueConverter": ".casia_value_converter",
    "CasiaValueConverter": ".casia_value_converter",
    "IntValueConverter": ".casia_value_converter",
    "CasiaValueSlider": ".casia_value_slider",
    "O3DViewControlWidget": ".open3d_view_control_widget",
    "O3DViewerWidget": ".open3d_widget",
    "PointCloudColorizeMode": ".point_cloud_control_widget",
    "PointCloudControlWidget": ".point_cloud_control_widget",
    "PointCloudDisplayStyle": ".point_cloud_control_widget",
    "PointCloudInfoWidget": ".point_cloud_info_widget",
}

def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(module_name, package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
