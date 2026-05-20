from __future__ import annotations

import copy
from pathlib import Path
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[2]
TWIST2_ROOT = REPO_ROOT.parent / "TWIST2"
SRC_G1D_COMPILED_XML = TWIST2_ROOT / "assets" / "g1_d" / "g1_d_compiled.xml"
SRC_G1D_DIR = SRC_G1D_COMPILED_XML.parent
GENERATED_DIR = REPO_ROOT / "assets" / ".generated"
GENERATED_G1D_MOBILE_XML = GENERATED_DIR / "g1_d_mobile_scene.xml"

WHEEL_RADIUS = 0.0848
WHEEL_HALF_WIDTH = 0.0341
WHEEL_TRACK = 0.4062
ROOT_HEIGHT = 0.1115


def _indent(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def _find_body_by_name(root: ET.Element, name: str) -> ET.Element | None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        return None
    for body in worldbody.iter("body"):
        if body.attrib.get("name") == name:
            return body
    return None


def _disable_mesh_collisions(body: ET.Element) -> None:
    for geom in body.findall("geom"):
        geom.attrib["contype"] = "0"
        geom.attrib["conaffinity"] = "0"
        geom.attrib["group"] = "1"
        geom.attrib["density"] = "0"


def prepare_g1d_mobile_scene(force_rebuild: bool = False) -> Path:
    if not SRC_G1D_COMPILED_XML.exists():
        raise FileNotFoundError(f"G1D compiled XML not found: {SRC_G1D_COMPILED_XML}")

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    if (
        GENERATED_G1D_MOBILE_XML.exists()
        and not force_rebuild
        and GENERATED_G1D_MOBILE_XML.stat().st_mtime >= SRC_G1D_COMPILED_XML.stat().st_mtime
    ):
        return GENERATED_G1D_MOBILE_XML

    src_root = ET.parse(SRC_G1D_COMPILED_XML).getroot()

    new_root = ET.Element("mujoco", {"model": "g1_d_mobile_scene"})

    compiler = src_root.find("compiler")
    if compiler is not None:
        compiler_new = copy.deepcopy(compiler)
        compiler_new.attrib["meshdir"] = str(SRC_G1D_DIR)
        new_root.append(compiler_new)

    ET.SubElement(
        new_root,
        "option",
        {
            "timestep": "0.002",
            "gravity": "0 0 -9.81",
            "integrator": "implicitfast",
            "iterations": "100",
            "noslip_iterations": "8",
        },
    )
    ET.SubElement(
        new_root,
        "statistic",
        {
            "center": "0 0 0.9",
            "extent": "1.4",
        },
    )

    visual = ET.SubElement(new_root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {"diffuse": "0.6 0.6 0.6", "ambient": "0.2 0.2 0.2", "specular": "0.9 0.9 0.9"},
    )
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(visual, "global", {"azimuth": "-140", "elevation": "-20"})

    default = ET.SubElement(new_root, "default")
    ET.SubElement(default, "joint", {"damping": "0.5", "armature": "0.01"})
    ET.SubElement(default, "geom", {"friction": "0.9 0.02 0.002"})

    asset = src_root.find("asset")
    if asset is not None:
        new_root.append(copy.deepcopy(asset))
    asset_new = new_root.find("asset")
    if asset_new is None:
        asset_new = ET.SubElement(new_root, "asset")
    ET.SubElement(asset_new, "texture", {
        "type": "skybox",
        "builtin": "flat",
        "rgb1": "0 0 0",
        "rgb2": "0 0 0",
        "width": "512",
        "height": "3072",
    })
    ET.SubElement(asset_new, "texture", {
        "type": "2d",
        "name": "groundplane",
        "builtin": "checker",
        "mark": "edge",
        "rgb1": "0.2 0.3 0.4",
        "rgb2": "0.1 0.2 0.3",
        "markrgb": "0.8 0.8 0.8",
        "width": "300",
        "height": "300",
    })
    ET.SubElement(asset_new, "material", {
        "name": "groundplane",
        "texture": "groundplane",
        "texuniform": "true",
        "texrepeat": "5 5",
        "reflectance": "0.2",
    })

    worldbody = ET.SubElement(new_root, "worldbody")
    ET.SubElement(worldbody, "light", {"pos": "1 0 3.5", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(worldbody, "geom", {
        "name": "floor",
        "type": "plane",
        "size": "0 0 0.05",
        "material": "groundplane",
        "friction": "1.2 0.02 0.002",
    })

    root_body = ET.SubElement(worldbody, "body", {"name": "g1d_root", "pos": f"0 0 {ROOT_HEIGHT:.4f}"})
    ET.SubElement(root_body, "freejoint", {"name": "g1d_freejoint"})

    src_worldbody = src_root.find("worldbody")
    if src_worldbody is None:
        raise ValueError(f"Malformed G1D XML: missing worldbody in {SRC_G1D_COMPILED_XML}")
    for child in list(src_worldbody):
        root_body.append(copy.deepcopy(child))

    left_wheel = _find_body_by_name(new_root, "Left_Wheel_Link")
    right_wheel = _find_body_by_name(new_root, "RIght_Wheel_Link")
    for wheel_body, sign in ((left_wheel, 1), (right_wheel, -1)):
        if wheel_body is None:
            continue
        _disable_mesh_collisions(wheel_body)
        ET.SubElement(
            wheel_body,
            "geom",
            {
                "name": f"{wheel_body.attrib['name']}_collision",
                "type": "cylinder",
                "size": f"{WHEEL_RADIUS:.4f} {WHEEL_HALF_WIDTH:.4f}",
                "quat": "0.707107 0.707107 0 0",
                "friction": "1.5 0.03 0.002",
                "condim": "6",
                "rgba": "0.15 0.15 0.15 0.35",
                "mass": "0.05",
            },
        )
        ET.SubElement(
            wheel_body,
            "site",
            {
                "name": f"{wheel_body.attrib['name']}_site",
                "type": "cylinder",
                "size": "0.003 0.003",
                "quat": "0.707107 0.707107 0 0",
                "rgba": "1 0.2 0.2 1" if sign > 0 else "0.2 0.6 1 1",
            },
        )

    actuator = ET.SubElement(new_root, "actuator")
    ET.SubElement(
        actuator,
        "velocity",
        {
            "name": "left_wheel_drive",
            "joint": "Left_Wheel_Joint",
            "kv": "25.0",
            "ctrlrange": "-30 30",
            "forcerange": "-80 80",
        },
    )
    ET.SubElement(
        actuator,
        "velocity",
        {
            "name": "right_wheel_drive",
            "joint": "Right_Wheel_Joint",
            "kv": "25.0",
            "ctrlrange": "-30 30",
            "forcerange": "-80 80",
        },
    )

    sensor = ET.SubElement(new_root, "sensor")
    ET.SubElement(sensor, "framepos", {"name": "g1d_base_pos", "objtype": "body", "objname": "g1d_root"})
    ET.SubElement(sensor, "framequat", {"name": "g1d_base_quat", "objtype": "body", "objname": "g1d_root"})
    ET.SubElement(sensor, "framelinvel", {"name": "g1d_base_linvel", "objtype": "body", "objname": "g1d_root"})
    ET.SubElement(sensor, "frameangvel", {"name": "g1d_base_angvel", "objtype": "body", "objname": "g1d_root"})

    _indent(new_root)
    GENERATED_G1D_MOBILE_XML.write_text(
        ET.tostring(new_root, encoding="unicode"),
        encoding="utf-8",
    )
    return GENERATED_G1D_MOBILE_XML


__all__ = [
    "prepare_g1d_mobile_scene",
    "GENERATED_G1D_MOBILE_XML",
    "WHEEL_RADIUS",
    "WHEEL_TRACK",
]
