# -*- coding: utf-8 -*-
"""
批量处理所有bbox的LST/NDVI剪切程序
功能：读取JSON文件中所有的bbox坐标，使用gdalwarp工具批量剪切对应的LST和NDVI图像区域，
      保留完整地理信息，支持多bbox处理
依赖：需安装GDAL并配置环境变量（确保命令行可运行gdalwarp）
"""

import os
import json
import subprocess
import platform
from typing import Tuple, List, Dict


def extract_all_geo_bboxes(json_path: str) -> List[Tuple[float, float, float, float]]:
    """
    从JSON结果文件中提取所有地理坐标边界框（bbox）
    :param json_path: JSON文件路径
    :return: 地理坐标列表，每个元素为 (xmin, ymin, xmax, ymax) 元组
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON文件不存在: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_bboxes = []
        
        # 处理多bbox结构（优先）
        if "bbox_process_results" in data:
            for item in data["bbox_process_results"]:
                if item.get("processing_status") == "success" and "crop_result" in item:
                    corners = item["crop_result"]["corner_coordinates"]
                    # 提取坐标并标准化
                    top_left = corners["top_left"]
                    bottom_right = corners["bottom_right"]
                    
                    xmin = min(top_left["x"], bottom_right["x"])
                    xmax = max(top_left["x"], bottom_right["x"])
                    ymin = min(top_left["y"], bottom_right["y"])
                    ymax = max(top_left["y"], bottom_right["y"])
                    
                    # 验证有效性
                    if (xmax - xmin) >= 0.1 and (ymax - ymin) >= 0.1:
                        all_bboxes.append((xmin, ymin, xmax, ymax))
        
        # 处理单bbox结构
        elif "output_info" in data and "corner_coordinates" in data["output_info"]:
            corners = data["output_info"]["corner_coordinates"]
            top_left = corners["top_left"]
            bottom_right = corners["bottom_right"]
            
            xmin = min(top_left["x"], bottom_right["x"])
            xmax = max(top_left["x"], bottom_right["x"])
            ymin = min(top_left["y"], bottom_right["y"])
            ymax = max(top_left["y"], bottom_right["y"])
            
            if (xmax - xmin) >= 0.1 and (ymax - ymin) >= 0.1:
                all_bboxes.append((xmin, ymin, xmax, ymax))
        
        # 验证是否提取到有效bbox
        if not all_bboxes:
            raise ValueError(f"JSON文件中未找到有效bbox记录: {json_path}")
            
        print(f"✅ 从JSON提取到 {len(all_bboxes)} 个有效bbox")
        return all_bboxes

    except json.JSONDecodeError:
        raise ValueError(f"JSON文件格式错误: {json_path}")
    except Exception as e:
        raise RuntimeError(f"提取地理bbox失败: {str(e)} (文件: {json_path})")


def gdalwarp_crop(input_raster: str, output_raster: str, bbox: Tuple[float, float, float, float], epsg: str) -> None:
    """
    使用gdalwarp工具按地理坐标剪切栅格图像
    :param input_raster: 输入图像路径
    :param output_raster: 输出图像路径
    :param bbox: 地理坐标边界框 (xmin, ymin, xmax, ymax)
    :param epsg: EPSG坐标系编码
    """
    # 验证gdalwarp是否可用
    try:
        subprocess.run(
            ["gdalwarp", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
    except FileNotFoundError:
        raise EnvironmentError("未找到gdalwarp工具，请确保GDAL已添加到系统环境变量")
    except subprocess.CalledProcessError:
        raise EnvironmentError("gdalwarp工具运行异常，请检查GDAL安装")

    xmin, ymin, xmax, ymax = bbox
    
    # 构建gdalwarp命令
    cmd = [
        "gdalwarp",
        "-overwrite",                  # 覆盖现有文件
        "-te", str(xmin), str(ymin), str(xmax), str(ymax),  # 地理裁剪范围
        "-t_srs", epsg,                # 输出坐标系
        "-r", "near",                  # 重采样方法（近邻法，适合指数数据）
        "-co", "COMPRESS=LZW",         # 压缩输出
        "-co", "TILED=YES",            # 分块存储，提高读取效率
        input_raster,
        output_raster
    ]

    # 执行命令
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # 验证输出文件
        if not os.path.exists(output_raster) or os.path.getsize(output_raster) == 0:
            raise RuntimeError("gdalwarp执行成功但未生成有效输出文件")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gdalwarp执行失败: {e.stderr}")


def batch_process(json_dir: str, lst_path: str, ndvi_path: str, output_root: str, epsg_code: str) -> None:
    """
    批量处理所有JSON文件中的所有bbox，剪切对应的LST和NDVI图像
    :param json_dir: JSON文件目录
    :param lst_path: LST图像路径
    :param ndvi_path: NDVI图像路径
    :param output_root: 输出根目录
    :param epsg_code: 手动指定的EPSG坐标系编码
    """
    # 1. 验证输入
    if not os.path.isdir(json_dir):
        raise NotADirectoryError(f"JSON目录不存在: {json_dir}")
    for path in [lst_path, ndvi_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"输入图像不存在: {path}")

    # 2. 创建输出目录
    lst_out_dir = os.path.join(output_root, "cropped_lst")
    ndvi_out_dir = os.path.join(output_root, "cropped_ndvi")
    os.makedirs(lst_out_dir, exist_ok=True)
    os.makedirs(ndvi_out_dir, exist_ok=True)

    # 3. 获取所有JSON文件
    json_files = [f for f in os.listdir(json_dir) 
                 if f.endswith("_result.json") and (f.startswith("patch_") or "bbox" in f)]
    
    if not json_files:
        raise ValueError(f"未找到符合条件的JSON文件（需以_result.json结尾）: {json_dir}")

    # 4. 批量处理
    total_json = len(json_files)
    total_bboxes = 0
    success_count = 0
    fail_records: List[Dict] = []

    print("="*80)
    print(f"开始批量剪切（共{total_json}个JSON文件）")
    print(f"使用坐标系: {epsg_code}")
    print(f"输出目录: {output_root}")
    print("="*80 + "\n")

    for json_idx, json_file in enumerate(json_files, 1):
        json_path = os.path.join(json_dir, json_file)
        base_name = os.path.splitext(json_file)[0].replace("_all_bboxes", "").replace("_result", "")
        
        print(f"[{json_idx}/{total_json}] 处理JSON文件: {json_file}")
        
        try:
            # 提取当前JSON中的所有bbox
            all_bboxes = extract_all_geo_bboxes(json_path)
            total_bboxes += len(all_bboxes)
            
            # 逐个处理每个bbox
            for bbox_idx, bbox in enumerate(all_bboxes, 1):
                print(f"  处理第 {bbox_idx}/{len(all_bboxes)} 个bbox: {[round(x, 3) for x in bbox]}")
                
                # 生成带bbox索引的输出文件名
                bbox_suffix = f"_bbox_{bbox_idx:03d}"  # 三位数索引，如_bbox_001
                lst_out = os.path.join(lst_out_dir, f"{base_name}{bbox_suffix}_lst.tif")
                ndvi_out = os.path.join(ndvi_out_dir, f"{base_name}{bbox_suffix}_ndvi.tif")
                
                try:
                    # 剪切LST
                    gdalwarp_crop(lst_path, lst_out, bbox, epsg_code)
                    
                    # 剪切NDVI
                    gdalwarp_crop(ndvi_path, ndvi_out, bbox, epsg_code)
                    
                    success_count += 1
                    print(f"    ✅ 成功生成: {os.path.basename(lst_out)} 和 {os.path.basename(ndvi_out)}")
                
                except Exception as e:
                    error_msg = str(e)
                    print(f"    ❌ 处理失败: {error_msg}")
                    fail_records.append({
                        "json_file": json_file,
                        "bbox_index": bbox_idx,
                        "bbox_coords": bbox,
                        "error": error_msg,
                        "timestamp": os.popen('date /t').read().strip() + " " + os.popen('time /t').read().strip()
                    })
            
            print(f"  🔍 该JSON文件处理完成，共{len(all_bboxes)}个bbox\n")
        
        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ JSON文件处理失败: {error_msg}\n")
            fail_records.append({
                "json_file": json_file,
                "error": f"JSON解析失败: {error_msg}",
                "timestamp": os.popen('date /t').read().strip() + " " + os.popen('time /t').read().strip()
            })

    # 5. 输出总结
    print("="*80)
    print("批量处理完成")
    print("="*80)
    print(f"总JSON文件数: {total_json}")
    print(f"总bbox数: {total_bboxes}")
    print(f"成功处理: {success_count}")
    print(f"处理失败: {len(fail_records)}")
    print(f"LST输出目录: {lst_out_dir}")
    print(f"NDVI输出目录: {ndvi_out_dir}")

    # 保存失败记录
    if fail_records:
        fail_log = os.path.join(output_root, "crop_fail_log.json")
        with open(fail_log, 'w', encoding='utf-8') as f:
            json.dump(fail_records, f, ensure_ascii=False, indent=4)
        print(f"\n⚠️  失败详情已保存至: {fail_log}")
    print("="*80)


if __name__ == "__main__":
    # ========================== 请修改为实际路径和参数 ==========================
    JSON_DIRECTORY = r"D:\Remote Sensing\Shenzhen\cropped_patches\all_bboxes_processing_results"          # 存放JSON结果的目录
    LST_IMAGE_PATH = r"D:\Remote Sensing\LST_23.tif"       # LST图像完整路径
    NDVI_IMAGE_PATH = r"D:\Remote Sensing\NDVI_23.tif"     # NDVI图像完整路径
    OUTPUT_ROOT_DIR = r"D:\Remote Sensing\Shenzhen\lst_ndvi"          # 结果输出根目录
    MANUAL_EPSG_CODE = "EPSG:32650"                         # 手动指定EPSG编码（根据实际坐标系修改）

    # 执行批量处理
    try:
        batch_process(JSON_DIRECTORY, LST_IMAGE_PATH, NDVI_IMAGE_PATH, OUTPUT_ROOT_DIR, MANUAL_EPSG_CODE)
    except Exception as e:
        print(f"\n❌ 程序执行失败: {str(e)}")
    

