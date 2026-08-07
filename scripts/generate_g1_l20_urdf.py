#!/usr/bin/env python3
"""Build the G1 arm model with the checked-in LinkerHand L20 models attached."""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_URDF = REPO_ROOT / "assets/g1/g1_body29_hand14.urdf"
OUTPUT_URDF = REPO_ROOT / "assets/g1/g1_body29_l20.urdf"
TARGET_L20_MASS_KG = 1.2
# Gravity-only validation offset for the physical right-hand center of mass.
# The IK model remains unchanged; this moves the attached hand in the gravity URDF.
RIGHT_GRAVITY_COM_FORWARD_OFFSET_M = 0.03

L20_SIDES = (
    {
        "side": "left",
        "source": REPO_ROOT / "assets/l20/left/linkerhand_l20_left.urdf",
        "prefix": "l20_left_",
        "root": "hand_base_link",
        "wrist": "left_wrist_yaw_link",
        "origin": "0.0415 0.003 0",
        "rpy": "0 1.5707963267948966 0",
        "mesh_dir": "../l20/left/meshes",
    },
    {
        "side": "right",
        "source": REPO_ROOT / "assets/l20/right/linkerhand_l20_right.urdf",
        "prefix": "l20_right_",
        "root": "base_link",
        "wrist": "right_wrist_yaw_link",
        "origin": f"{0.0415 + RIGHT_GRAVITY_COM_FORWARD_OFFSET_M:.4f} -0.003 0",
        "rpy": "0 1.5707963267948966 0",
        "mesh_dir": "../l20/right/meshes",
    },
)


def _link_subtree(root: ET.Element, root_link: str) -> set[str]:
    joints = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and child is not None:
            joints.append((parent.get("link"), child.get("link")))

    links = {root_link}
    changed = True
    while changed:
        changed = False
        for parent, child in joints:
            if parent in links and child not in links:
                links.add(child)
                changed = True
    return links


def _remove_default_hand(root: ET.Element, side: str) -> None:
    old_root = f"{side}_hand_palm_link"
    removed_links = _link_subtree(root, old_root)
    for element in list(root):
        if element.tag == "link" and element.get("name") in removed_links:
            root.remove(element)
        elif element.tag == "joint":
            parent = element.find("parent")
            child = element.find("child")
            parent_name = None if parent is None else parent.get("link")
            child_name = None if child is None else child.get("link")
            if parent_name in removed_links or child_name in removed_links:
                root.remove(element)


def _rename_link_tree(element: ET.Element, prefix: str) -> None:
    if element.tag == "link":
        element.set("name", prefix + element.get("name", ""))
        for mesh in element.findall(".//mesh"):
            filename = mesh.get("filename")
            if filename:
                mesh.set("filename", filename.replace("meshes/", ""))
    elif element.tag == "joint":
        element.set("name", prefix + element.get("name", ""))
        for relation in ("parent", "child"):
            node = element.find(relation)
            if node is not None:
                node.set("link", prefix + node.get("link", ""))
        mimic = element.find("mimic")
        if mimic is not None and mimic.get("joint"):
            mimic.set("joint", prefix + mimic.get("joint"))


def _append_l20(root: ET.Element, spec: dict[str, str]) -> None:
    source_root = ET.parse(spec["source"]).getroot()
    source_mass = sum(
        float(mass.get("value"))
        for mass in source_root.findall("link/inertial/mass")
    )
    mass_scale = TARGET_L20_MASS_KG / source_mass

    for material in source_root.findall("material"):
        if root.find(f"material[@name='{material.get('name')}']") is None:
            root.append(deepcopy(material))

    for element in source_root:
        if element.tag not in {"link", "joint"}:
            continue
        copied = deepcopy(element)
        _rename_link_tree(copied, spec["prefix"])
        if copied.tag == "link":
            inertial = copied.find("inertial")
            if inertial is not None:
                mass = inertial.find("mass")
                if mass is not None:
                    mass.set("value", str(float(mass.get("value")) * mass_scale))
                inertia = inertial.find("inertia")
                if inertia is not None:
                    for attribute in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                        if inertia.get(attribute) is not None:
                            inertia.set(attribute, str(float(inertia.get(attribute)) * mass_scale))
        for mesh in copied.findall(".//mesh"):
            filename = mesh.get("filename")
            if filename:
                mesh.set("filename", f"{spec['mesh_dir']}/{filename}")
        root.append(copied)

    mount = ET.Element("joint", {"name": f"{spec['side']}_l20_mount_joint", "type": "fixed"})
    ET.SubElement(mount, "origin", {"xyz": spec["origin"], "rpy": spec["rpy"]})
    ET.SubElement(mount, "parent", {"link": spec["wrist"]})
    ET.SubElement(mount, "child", {"link": spec["prefix"] + spec["root"]})
    root.append(mount)


def main() -> None:
    tree = ET.parse(BASE_URDF)
    root = tree.getroot()
    root.set("name", "g1_body29_l20")

    _remove_default_hand(root, "left")
    _remove_default_hand(root, "right")
    for spec in L20_SIDES:
        _append_l20(root, spec)

    ET.indent(tree, space="  ")
    tree.write(OUTPUT_URDF, encoding="utf-8", xml_declaration=True)
    print(f"wrote {OUTPUT_URDF}")


if __name__ == "__main__":
    main()
