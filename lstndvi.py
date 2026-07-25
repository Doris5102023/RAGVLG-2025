# -*- coding: utf-8 -*-
"""
LST和NDVI patch平均值计算程序（修正bbox坐标记录）
功能：批量计算LST和NDVI patch的平均值，按原始JSON中的top_left和bottom_right记录坐标，
      确保y值方向正确（top_left y > bottom_right y）
依赖：GDAL库、numpy库
"""

import os
import json
import re
import numpy as np
from osgeo import gdal
from typing import Dict, List


def calculate_raster_mean(raster_path: str) -> float:
    """计算单张栅格图像的平均值（忽略NoData值）"""
    # 打开栅格文件
    dataset = gdal.Open(raster_path)
    if not dataset:
        raise FileNotFoundError(f"无法打开栅格文件: {raster_path}")
    
    try:
        # 获取第一波段数据
        band = dataset.GetRasterBand(1)
        if not band:
            raise ValueError(f"栅格文件无有效波段: {raster_path}")
        
        # 读取数据和NoData值
        data = band.ReadAsArray()
        nodata = band.GetNoDataValue()
        
        # 处理NoData值
        if nodata is not None:
            if np.issubdtype(data.dtype, np.floating):
                data = np.where(np.isclose(data, nodata), np.nan, data)
            else:
                data = np.where(data == nodata, np.nan, data)
        
        # 计算有效平均值
        mean_value = np.nanmean(data)
        
        if np.isnan(mean_value) or mean_value == 0:
            raise ValueError(f"无法计算有效平均值（可能全为NoData）: {raster_path}")
        
        return round(float(mean_value), 4)
    
    finally:
        dataset = None


def find_matching_ndvi(lst_filename: str, ndvi_dir: str) -> str:
    """根据LST文件名找到对应的NDVI文件"""
    # 替换LST标识为NDVI标识
    ndvi_filename = lst_filename.replace("_lst.tif", "_ndvi.tif")
    ndvi_path = os.path.join(ndvi_dir, ndvi_filename)
    
    if os.path.exists(ndvi_path):
        return ndvi_path
    
    # 尝试其他命名格式
    alternative_patterns = [
        (r"_lst\.tif$", "_ndvi.tif"),
        (r"_LST\.tif$", "_NDVI.tif"),
        (r"lst_", "ndvi_")
    ]
    
    for pattern, replacement in alternative_patterns:
        ndvi_filename = re.sub(pattern, replacement, lst_filename)
        ndvi_path = os.path.join(ndvi_dir, ndvi_filename)
        if os.path.exists(ndvi_path):
            return ndvi_path
    
    raise FileNotFoundError(f"未找到对应的NDVI文件: {lst_filename} -> {ndvi_filename}")


def extract_bbox_info_from_filename(filename: str) -> Dict:
    """从文件名中提取patch名和bbox索引"""
    match = re.match(r"^(.*?)_bbox_(\d+)_.*?\.tif$", filename)
    if match:
        return {
            "original_patch": match.group(1),
            "bbox_index": int(match.group(2)),
            "filename": filename
        }
    return {
        "original_patch": "unknown",
        "bbox_index": -1,
        "filename": filename
    }


def load_bbox_coordinates_from_json(json_dir: str) -> Dict:
    """
    预加载所有JSON文件中的bbox坐标信息，按top_left和bottom_right原始坐标记录
    确保top_left的y值大于bottom_right的y值（符合地理图像坐标规范）
    """
    bbox_coords = {}
    
    if not os.path.isdir(json_dir):
        raise NotADirectoryError(f"JSON目录不存在: {json_dir}")
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(json_dir)
                 if f.endswith("_result.json") and (f.startswith("patch_") or "bbox" in f)]
    
    if not json_files:
        raise ValueError(f"未找到JSON文件: {json_dir}")
    
    # 逐个解析JSON文件
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        base_name = os.path.splitext(json_file)[0].replace("_all_bboxes", "").replace("_result", "")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 初始化当前patch的bbox存储
            bbox_coords[base_name] = {}
            
            # 处理多bbox结构
            if "bbox_process_results" in data:
                for idx, item in enumerate(data["bbox_process_results"], 1):
                    if item.get("processing_status") == "success" and "crop_result" in item:
                        # 直接使用原始的top_left和bottom_right坐标（不做min/max处理）
                        corners = item["crop_result"]["corner_coordinates"]
                        top_left = corners["top_left"]
                        bottom_right = corners["bottom_right"]
                        
                        # 验证坐标方向（top_left y应大于bottom_right y）
                        if top_left["y"] < bottom_right["y"]:
                            print(f"⚠️  JSON文件{json_file}中bbox {idx}的top_left y值小于bottom_right y值，已按原始记录")
                        
                        # 按原始坐标存储（保留6位小数）
                        bbox_coords[base_name][idx] = {
                            "top_left": {
                                "x": round(top_left["x"], 6),
                                "y": round(top_left["y"], 6)
                            },
                            "bottom_right": {
                                "x": round(bottom_right["x"], 6),
                                "y": round(bottom_right["y"], 6)
                            }
                        }
            
            # 处理单bbox结构
            elif "output_info" in data and "corner_coordinates" in data["output_info"]:
                corners = data["output_info"]["corner_coordinates"]
                top_left = corners["top_left"]
                bottom_right = corners["bottom_right"]
                
                # 按原始坐标存储
                bbox_coords[base_name][1] = {  # 单bbox默认索引为1
                    "top_left": {
                        "x": round(top_left["x"], 6),
                        "y": round(top_left["y"], 6)
                    },
                    "bottom_right": {
                        "x": round(bottom_right["x"], 6),
                        "y": round(bottom_right["y"], 6)
                    }
                }
        
        except Exception as e:
            print(f"⚠️  解析JSON文件时警告: {json_file}，错误: {str(e)}")
            continue
    
    return bbox_coords


def batch_calculate_means(lst_dir: str, ndvi_dir: str, json_dir: str, 
                         output_dir: str, fail_log_path: str = None) -> None:
    """
    批量计算所有LST和NDVI patch的平均值，按top_left和bottom_right记录原始坐标
    """
    # 验证输入目录
    for dir_path in [lst_dir, ndvi_dir, json_dir]:
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"目录不存在: {dir_path}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置失败日志路径
    if not fail_log_path:
        fail_log_path = os.path.join(output_dir, "mean_calculation_fail_log.json")
    
    # 预加载所有JSON中的bbox坐标信息（按top_left和bottom_right）
    print("="*80)
    print("开始加载JSON中的bbox坐标信息...")
    bbox_coords = load_bbox_coordinates_from_json(json_dir)
    print(f"成功加载 {len(bbox_coords)} 个patch的bbox坐标信息")
    print("="*80 + "\n")
    
    # 获取所有LST文件
    lst_files = [f for f in os.listdir(lst_dir) 
                if f.endswith((".tif", ".TIF")) and ("lst" in f.lower() or "LST" in f)]
    
    if not lst_files:
        raise ValueError(f"未找到LST文件: {lst_dir}")
    
    # 初始化统计变量
    total_files = len(lst_files)
    success_count = 0
    results: List[Dict] = []
    fail_records: List[Dict] = []
    
    print("="*80)
    print(f"开始计算LST和NDVI平均值（共{total_files}个patch）")
    print(f"LST目录: {lst_dir}")
    print(f"NDVI目录: {ndvi_dir}")
    print(f"JSON目录: {json_dir}")
    print(f"结果输出: {output_dir}")
    print("="*80 + "\n")
    
    # 批量处理每个LST-NDVI对
    for idx, lst_filename in enumerate(lst_files, 1):
        lst_path = os.path.join(lst_dir, lst_filename)
        print(f"[{idx}/{total_files}] 处理: {lst_filename}")
        
        try:
            # 1. 提取bbox信息
            bbox_info = extract_bbox_info_from_filename(lst_filename)
            original_patch = bbox_info["original_patch"]
            bbox_index = bbox_info["bbox_index"]
            
            # 2. 获取对应的bbox坐标（top_left和bottom_right）
            if original_patch not in bbox_coords:
                raise ValueError(f"未找到对应的JSON信息: {original_patch}")
            
            if bbox_index not in bbox_coords[original_patch]:
                raise ValueError(f"未找到索引为 {bbox_index} 的bbox坐标: {original_patch}")
            
            bbox = bbox_coords[original_patch][bbox_index]
            print(f"   找到bbox坐标: "
                  f"top_left=({bbox['top_left']['x']}, {bbox['top_left']['y']}), "
                  f"bottom_right=({bbox['bottom_right']['x']}, {bbox['bottom_right']['y']})")
            
            # 3. 找到对应的NDVI文件
            ndvi_path = find_matching_ndvi(lst_filename, ndvi_dir)
            print(f"   对应NDVI文件: {os.path.basename(ndvi_path)}")
            
            # 4. 计算LST平均值
            lst_mean = calculate_raster_mean(lst_path)
            
            # 5. 计算NDVI平均值
            ndvi_mean = calculate_raster_mean(ndvi_path)
            
            # 6. 记录结果（包含原始top_left和bottom_right坐标）
            result = {
                **bbox_info,
                "bbox_coordinates": bbox,  # 按top_left和bottom_right存储
                "lst_path": lst_path,
                "ndvi_path": ndvi_path,
                "lst_mean": lst_mean,
                "ndvi_mean": ndvi_mean,
                "processing_status": "success"
            }
            results.append(result)
            
            success_count += 1
            print(f"   ✅ 平均值计算完成: LST={lst_mean}, NDVI={ndvi_mean}\n")
        
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ 处理失败: {error_msg}\n")
            fail_records.append({
                "filename": lst_filename,
                "error": error_msg,
                "timestamp": os.popen('date /t').read().strip() + " " + os.popen('time /t').read().strip()
            })
    
    # 保存统计结果
    output_path = os.path.join(output_dir, "lst_ndvi_mean_with_bbox_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # 保存CSV格式结果（拆分top_left和bottom_right坐标）
    csv_path = os.path.join(output_dir, "lst_ndvi_mean_with_bbox_results.csv")
    with open(csv_path, 'w', encoding='utf-8-sig') as f:
        # 写入表头
        f.write("original_patch,bbox_index,lst_mean,ndvi_mean,"
                "top_left_x,top_left_y,bottom_right_x,bottom_right_y,"
                "lst_path,ndvi_path\n")
        # 写入数据
        for res in results:
            bbox = res["bbox_coordinates"]
            f.write(
                f"{res['original_patch']},{res['bbox_index']},{res['lst_mean']},{res['ndvi_mean']},"
                f"{bbox['top_left']['x']},{bbox['top_left']['y']},"
                f"{bbox['bottom_right']['x']},{bbox['bottom_right']['y']},"
                f"\"{res['lst_path']}\",\"{res['ndvi_path']}\"\n"
            )
    
    # 输出总结
    print("="*80)
    print("平均值计算完成")
    print("="*80)
    print(f"总处理patch数: {total_files}")
    print(f"成功计算: {success_count}")
    print(f"计算失败: {len(fail_records)}")
    print(f"JSON结果文件: {output_path}")
    print(f"CSV结果文件: {csv_path}")
    
    # 保存失败记录
    if fail_records:
        with open(fail_log_path, 'w', encoding='utf-8') as f:
            json.dump(fail_records, f, ensure_ascii=False, indent=4)
        print(f"\n⚠️  失败记录: {fail_log_path}")
    print("="*80)


if __name__ == "__main__":
    # ========================== 请修改为实际路径 ==========================
    LST_PATCH_DIR = r"D:\Remote Sensing\Shenzhen\lst_ndvi\cropped_lst"       # LST patch所在目录
    NDVI_PATCH_DIR = r"D:\Remote Sensing\Shenzhen\lst_ndvi\cropped_ndvi"     # NDVI patch所在目录
    JSON_DIR = r"D:\Remote Sensing\Shenzhen\cropped_patches\all_bboxes_processing_results"                  # 原始JSON文件所在目录（含bbox坐标）
    OUTPUT_RESULT_DIR = r"D:\Remote Sensing\Shenzhen\mean_results\mean_results_withbbox"  # 结果输出目录
    
    # 执行批量计算
    try:
        batch_calculate_means(LST_PATCH_DIR, NDVI_PATCH_DIR, JSON_DIR, OUTPUT_RESULT_DIR)
    except Exception as e:
        print(f"\n❌ 程序执行失败: {str(e)}")
    
    
    