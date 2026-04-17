from scipy.spatial.transform import Rotation as R

from src.utils.Datas import Transform


def SE3_string(mat):
    t = Transform.from_SE3(mat)
    x, y, z = t.translation.to_list()

    rz, ry, rx = R.from_quat(t.rotation.to_list(), scalar_first=True).as_euler("zyx", degrees=True)
    return f"x:{x:+.3f}, y:{y:+.3f}, z:{z:+.3f}, rz:{rz:+.4f}, ry:{ry:+.4f}, rx:{rx:+.4f}"


def SE3_2_xyzr(mat):
    t = Transform.from_SE3(mat)
    x, y, z = t.translation.to_list()

    rz, ry, rx = R.from_quat(t.rotation.to_list(), scalar_first=True).as_euler("zyx", degrees=True)
    return x, y, z, rz, ry, rx
