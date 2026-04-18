import sys
sys.path.append('../DataReader')
from read import DataReader
from util import zRotMatrix

reader = DataReader()

import os
from random import choice
import numpy as np
import plotly.graph_objs as go

import argparse, tqdm
from tqdm import tqdm
from IO import readOBJ

# Triangulated faces
def quads2tris(F):
    out = []
    for f in F:
        if len(f) == 3: out += [f]
        elif len(f) == 4: out += [[f[0],f[1],f[2]],
                                [f[0],f[2],f[3]]]
        else: print("This should not happen...")
    return np.array(out, np.int32)

# Display mesh
def display(V, F, C):
    if F.shape[1] != 3: F = quads2tris(F)
    fig = go.Figure(data=[
        go.Mesh3d(
            x=V[:,0],
            y=V[:,1],
            z=V[:,2],
            # i, j and k give the vertices of triangles
            i = F[:,0],
            j = F[:,1],
            k = F[:,2],
            vertexcolor = C,
            showscale=True
        )
    ])
    fig.show()
    
def get_num_frames(sample):
    info = reader.read_info(sample)
    poses = np.asarray(info["poses"])
    return 1 if poses.ndim == 1 else poses.shape[1]

def read_garment_rest_vertices(sample, garment):
    obj_path = os.path.join(reader.SRC, sample, garment + '.obj')
    V, _, __, ___ = readOBJ(obj_path)
    return V
    # # CLOTH3D stores a per-sequence z-axis world rotation in info["zrot"].
    # # The animated garment and human readers already apply this rotation, so
    # # the rest garment must be rotated as well to keep stage.obj consistent
    # # with the animated cloth/body world frame.
    # info = reader.read_info(sample)
    # zRot = zRotMatrix(info["zrot"])
    # return zRot.dot(V.T).T

def faces_to_edges(F):
    E = np.concatenate((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]), axis=0)
    E = np.sort(E, axis=1)
    E = np.unique(E, axis=0)
    return E
    
def extract_sample_single_frame(sample, frame, use_uv_map=False, show_display=False):
    """
    Extract one frame and return a flat dictionary of mesh arrays.

    Output schema (output):
    - "sample": `str`，样本编码，如 "00016"
    - "frame": `int`，帧索引
    - "use_uv_map": `bool`，是否使用了 UV 映射，False表示使用颜色渐变，True表示使用 UV 映射
    - "merged_V": `(N_merged, 3)` float array，人体+所有衣物mesh的3D点坐标
    - "merged_F": `(M_merged, 3)` int array，人体+所有衣物mesh的面的三点索引
    - "merged_C": `(N_merged, 3)` uint8 array，人体+所有衣物mesh的点颜色
    - "merged_E": `(#E_merged, 2)` int array，人体+所有衣物mesh的边的两点索引
    - "human_V": `(N_human, 3)` float array，人体mesh的3D点坐标
    - "human_F": `(M_human, 3)` int array，人体mesh的面的三点索引
    - "human_C": `(N_human, 3)` uint8 array，人体mesh的点颜色
    - "human_E": `(#E_human, 2)` int array，人体mesh的边的两点索引
    - "garments_merged_V": `(N_garments, 3)` float array，所有衣物mesh的3D点坐标
    - "garments_merged_V_rest": `(N_garments, 3)` float array，所有衣物mesh在自然状态下的原始3D点坐标
    - "garments_merged_F": `(M_garments, 3)` int array，所有衣物mesh的面的三点索引
    - "garments_merged_C": `(N_garments, 3)` uint8 array，所有衣物mesh的点颜色
    - "garments_merged_E": `(#E_garments, 2)` int array，所有衣物mesh的边的两点索引
    - "garment_names": `(G,)` string array, where `G` is the number of garments，衣物的名称列表

    Per-garment entries are also included using the garment name:
    - "garment_<name>_V": `(N_garment, 3)` float array，单件衣物的3D点坐标
    - "garment_<name>_F": `(M_garment, 3)` int array，单件衣物的面的三点索引
    - "garment_<name>_C": `(N_garment, 3)` uint8 array，单件衣物的点颜色
    - "garment_<name>_V_rest": `(N_garment, 3)` float array，单件衣物在自然状态下的原始3D点坐标
    - "garment_<name>_E": `(#E_garment, 2)` int array，单件衣物的边的两点索引
    - "garment_<name>_fabric": `str` fabric type from CLOTH3D (e.g. cotton, silk)，布料类型，用于C-IPC材质映射

    Notes:
    - `merged_*` contains human + all garments.
    - `garments_merged_*` contains all garments only, without the human mesh.
    - Per-garment `F` uses local vertex indexing for that garment alone.
    - Color arrays are vertex colors. With `use_uv_map=False` they are synthetic
      gradient colors; with `use_uv_map=True` they come from the garment texture.
    - When loading from `.npz`, convert scalar metadata keys with `.item()`:
      "sample", "frame", and "use_uv_map".
    """
    
    individual_garments = []
    
    info = reader.read_info(sample)
    
    """ Human Meshes """
    V_human, F_human = reader.read_human(sample, frame)
    V_human = np.array(V_human)
    F_human = np.array(F_human, np.int32)
    C_human = np.array([[255, 255, 255]] * V_human.shape[0], np.uint8) # white color for human mesh
    E_human = faces_to_edges(F_human)

    """ Garment Meshes """
    garments = list(info["outfit"].keys())
    num_garments = max(len(garments) - 1, 1)

    V_garments_list = []
    F_garments_list = []
    C_garments_list = []
    V_rest_garments_list = []
    E_garments_list = []

    vertex_offset = 0
    for i, garment in enumerate(garments):
        _V = reader.read_garment_vertices(sample, garment, frame)
        _F = reader.read_garment_topology(sample, garment)
        _V_rest = read_garment_rest_vertices(sample, garment)
        
        if use_uv_map:
            # Read garment vertex colors
            Vt, Ft = reader.read_garment_UVMap(sample, garment) # UV map required to estimate vertex color
            _C = reader.read_garment_vertex_colors(sample, garment, _F, Vt, Ft)
            if _C.ndim == 1: _C = np.stack([_C]*_V.shape[0], 0) # Plain RGB color
        else:
            r = int(255 * i / num_garments)
            b = int(255 * (1 - i / num_garments))
            _C = np.array([[r, 0, b]] * _V.shape[0], np.uint8)
        
        _F = quads2tris(_F)
        _E = faces_to_edges(_F)
        # Fabric type from info (e.g. cotton, silk) for C-IPC material mapping
        _fabric = info["outfit"][garment].get("fabric", "")

        individual_garments.append({
            "name": garment,
            "V": _V,
            "F": _F,
            "C": _C,
            "V_rest": _V_rest,
            "E": _E,
            "fabric": _fabric,
        })

        V_garments_list.append(_V)
        F_garments_list.append(_F + vertex_offset)
        C_garments_list.append(_C)
        V_rest_garments_list.append(_V_rest)
        E_garments_list.append(_E + vertex_offset)
        
        vertex_offset += _V.shape[0]

    if len(V_garments_list) > 0:
        V_garments = np.concatenate(V_garments_list, axis=0)
        V_rest_garments = np.concatenate(V_rest_garments_list, axis=0)
        F_garments = np.concatenate(F_garments_list, axis=0)
        C_garments = np.concatenate(C_garments_list, axis=0)
        E_garments = np.concatenate(E_garments_list, axis=0)
    else:
        V_garments = np.zeros((0, 3))
        V_rest_garments = np.zeros((0, 3))
        F_garments = np.zeros((0, 3), dtype=int)
        C_garments = np.zeros((0, 3), dtype=np.uint8)
        E_garments = np.zeros((0, 2), dtype=int)

    """ Merged Meshes"""
    V_merged = np.concatenate((V_human, V_garments), axis=0)
    F_merged = np.concatenate((F_human, F_garments + V_human.shape[0]), axis=0) # offset the garment vertices by the number of human vertices
    C_merged = np.concatenate((C_human, C_garments), axis=0)
    E_merged = np.concatenate((E_human, E_garments + V_human.shape[0]), axis=0)

    """ DISPLAY """
    if show_display:
        display(V_merged, F_merged, C_merged)
    
    output = {
        "sample": sample,
        "frame": frame,
        "use_uv_map": use_uv_map,
        "merged_V": V_merged,
        "merged_F": F_merged,
        "merged_C": C_merged,
        "merged_E": E_merged,
        "human_V": V_human,
        "human_F": F_human,
        "human_C": C_human,
        "human_E": E_human,
        "garments_merged_V": V_garments,
        "garments_merged_V_rest": V_rest_garments,
        "garments_merged_F": F_garments,
        "garments_merged_C": C_garments,
        "garments_merged_E": E_garments,
        "garment_names": np.array(garments),
    }
    
    for g in individual_garments:
        garment_key = f"garment_{g['name']}"
        if f"{garment_key}_V" in output or f"{garment_key}_F" in output or f"{garment_key}_C" in output:
            raise ValueError(f"Key {garment_key} already exists in output, should consider naming like garment_0_V plus garment_name.")
        output[f"{garment_key}_V"] = g["V"]
        output[f"{garment_key}_F"] = g["F"]
        output[f"{garment_key}_C"] = g["C"]
        output[f"{garment_key}_V_rest"] = g["V_rest"]
        output[f"{garment_key}_E"] = g["E"]
        output[f"{garment_key}_fabric"] = g["fabric"]

    return output

def extract_sample_all_frames(sample, use_uv_map=False):
    """
    Extract all frames of one sample and return a flat dictionary of static data
    plus per-frame vertex sequences.

    Output schema (final_output):
    - "sample": `str` 样本编码，如 "00016"
    - "num_frames": `int`, number of frames `T`，这个样本的帧数
    - "frame_dt": `float`, frame time step (1/30 s = 30 fps)，帧时间步长
    - "length_unit": `str`, length unit ("m" = meters, inferred from body height ~1.7–1.9m)，长度单位
    - "up_axis": `str`, vertical axis ("z" in CLOTH3D, largest span for standing pose)，竖直轴
    - "use_uv_map": `bool` 是否使用了 UV 映射，False表示使用颜色渐变，True表示使用 UV 映射
    - "merged_V_seq": `(T, N_merged, 3)` float array，人体+所有衣物mesh的3D点坐标序列，T是帧数，N_merged是人3D点的数量
    - "merged_F": `(M_merged, 3)` int array，人体+所有衣物mesh的面的三点索引
    - "merged_C": `(N_merged, 3)` uint8 array，人体+所有衣物mesh的点颜色
    - "merged_E": `(#E_merged, 2)` int array，人体+所有衣物mesh的边的两点索引
    - "human_V_seq": `(T, N_human, 3)` float array，人体mesh的3D点坐标序列，T是帧数，N_human是人3D点的数量
    - "human_F": `(M_human, 3)` int array，人体mesh的面的三点索引
    - "human_C": `(N_human, 3)` uint8 array，人体mesh的点颜色
    - "human_E": `(#E_human, 2)` int array，人体mesh的边的两点索引
    - "garments_merged_V_seq": `(T, N_garments, 3)` float array，所有衣物mesh的3D点坐标序列，T是帧数，N_garments是所有衣物mesh的3D点的数量
    - "garments_merged_V_rest": `(N_garments, 3)` float array，所有衣物mesh在自然状态下的原始3D点坐标
    - "garments_merged_F": `(M_garments, 3)` int array，所有衣物mesh的面的三点索引
    - "garments_merged_C": `(N_garments, 3)` uint8 array，所有衣物mesh的点颜色
    - "garments_merged_E": `(#E_garments, 2)` int array，所有衣物mesh的边的两点索引
    - "garment_names": `(G,)` string array, where `G` is the number of garments，衣物的名称列表

    Per-garment entries are also included using the garment name:
    - "garment_<name>_V_seq": `(T, N_garment, 3)` float array，单件衣物的3D点坐标序列，T是帧数，N_garment是单件衣物的3D点的数量
    - "garment_<name>_F": `(M_garment, 3)` int array，单件衣物的面的三点索引
    - "garment_<name>_C": `(N_garment, 3)` uint8 array，单件衣物的点颜色
    - "garment_<name>_V_rest": `(N_garment, 3)` float array，单件衣物在自然状态下的原始3D点坐标
    - "garment_<name>_E": `(#E_garment, 2)` int array，单件衣物的边的两点索引
    - "garment_<name>_fabric": `str` fabric type from CLOTH3D (e.g. cotton, silk)，布料类型，用于C-IPC材质映射
    
    Notes:
    - Keys ending with `_seq` are frame sequences stacked along axis 0.
    - `F` and `C` are stored once because they are static for this dataset.
    - Per-garment `F` uses local vertex indexing for that garment alone.
    - The merged sequence can be reconstructed from `human_V_seq` and
      `garments_merged_V_seq`, but it is included here for convenience.
    - When loading from `.npz`, convert scalar metadata keys with `.item()`:
      "sample", "num_frames", "frame_dt", "length_unit", "up_axis", "use_uv_map",
      and "garment_<name>_fabric" for each garment.
    """
    
    all_frames_output = []
    for frame in tqdm(range(get_num_frames(sample))):
        frame_output = extract_sample_single_frame(sample, frame, use_uv_map, False)
        all_frames_output.append(frame_output)
    final_output = {
        "sample": all_frames_output[0]["sample"],
        "num_frames": len(all_frames_output),
        "frame_dt": 1.0 / 30,  # 30 fps
        "length_unit": "m",  # meters (CLOTH3D body height ~1.7–1.9m)
        "up_axis": "z",
        "use_uv_map": all_frames_output[0]["use_uv_map"],
        "merged_V_seq": np.stack([frame["merged_V"] for frame in all_frames_output], axis=0),
        "merged_F": all_frames_output[0]["merged_F"],
        "merged_C": all_frames_output[0]["merged_C"],
        "merged_E": all_frames_output[0]["merged_E"],
        "human_V_seq": np.stack([frame["human_V"] for frame in all_frames_output], axis=0),
        "human_F": all_frames_output[0]["human_F"],
        "human_C": all_frames_output[0]["human_C"],
        "human_E": all_frames_output[0]["human_E"],
        "garments_merged_V_seq": np.stack([frame["garments_merged_V"] for frame in all_frames_output], axis=0),
        "garments_merged_V_rest": all_frames_output[0]["garments_merged_V_rest"],
        "garments_merged_F": all_frames_output[0]["garments_merged_F"],
        "garments_merged_C": all_frames_output[0]["garments_merged_C"],
        "garments_merged_E": all_frames_output[0]["garments_merged_E"],
        "garment_names": all_frames_output[0]["garment_names"],
    }
    for g in all_frames_output[0]["garment_names"]:
        garment_key = f"garment_{g}"
        if f"{garment_key}_V_seq" in final_output or f"{garment_key}_F" in final_output or f"{garment_key}_C" in final_output:
            raise ValueError(f"Key {garment_key} already exists in output, should consider naming like garment_0_V_seq plus garment_name.")
        final_output[f"{garment_key}_V_seq"] = np.stack([frame[f"{garment_key}_V"] for frame in all_frames_output], axis=0)
        final_output[f"{garment_key}_F"] = all_frames_output[0][f"{garment_key}_F"]
        final_output[f"{garment_key}_C"] = all_frames_output[0][f"{garment_key}_C"]
        final_output[f"{garment_key}_V_rest"] = all_frames_output[0][f"{garment_key}_V_rest"]
        final_output[f"{garment_key}_E"] = all_frames_output[0][f"{garment_key}_E"]
        final_output[f"{garment_key}_fabric"] = all_frames_output[0][f"{garment_key}_fabric"]  # static per garment

    return final_output

def save_sample(output_dict, output_path):
    os.makedirs(output_path, exist_ok=True)
    np.savez_compressed(os.path.join(output_path, f"{output_dict['sample']}.npz"), **output_dict)
    
def load_sample(input_path, sample):
    with np.load(os.path.join(input_path, f"{sample}.npz"), allow_pickle=False) as data:
        loaded = {k: data[k] for k in data.files}
    # Convert numpy scalars to Python types
    for key in ["sample", "frame", "num_frames", "frame_dt", "length_unit", "up_axis", "use_uv_map"]:
        if key in loaded:
            loaded[key] = loaded[key].item()
    for key in list(loaded.keys()):
        if key.endswith("_fabric"):
            loaded[key] = loaded[key].item()
    return loaded

# Note: This file must be placed in the root directory of any folder in StarterKit,
# and the source data of samples must be stored in the StarterKit/Samples folder with
# sample code folder name. Otherwise, it cannot run unless the DataReader source code is modified.
# 备注：在不改变DataReader源码的情况下，本文件必须存放在StarterKit内任意文件夹根目录下，
# 且样本源数据必须存放在StarterKit/Samples文件夹下，按样本编码分文件夹存储，否则无法运行。

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, default="00016") # 样本编码，如 "00016"
    parser.add_argument("--frame", type=int, default=-1) # 帧索引，-1表示提取所有帧，否则提取指定帧
    parser.add_argument("--use_uv_map", default=False, action="store_true") # 是否使用 UV 映射，False表示使用颜色渐变，True表示使用 UV 映射
    parser.add_argument("--show_display", default=False, action="store_true") # 对于单帧提取，是否显示渲染结果
    parser.add_argument("--output_path", type=str, default="../../Codim-IPC/Projects/FEMShell/cloth3d_data/") # 输出npz文件路径
    args = parser.parse_args()
    
    if args.frame == -1:
        output = extract_sample_all_frames(args.sample, args.use_uv_map)
    else:
        output = extract_sample_single_frame(args.sample, args.frame, args.use_uv_map, args.show_display)
        print("WARNING: to run the 5a_cloth3d.py, you need to extract all frames by setting frame to -1")
    save_sample(output, args.output_path)
    loaded = load_sample(args.output_path, args.sample)
    for key in loaded.keys():
        print(key)
    for garment in loaded["garment_names"]:
        print(f"garment {garment} fabric: {loaded[f'garment_{garment}_fabric']}")
