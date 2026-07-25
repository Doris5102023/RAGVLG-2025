# -*- coding: utf-8 -*-
"""
批量处理遥感图像Patch程序（提取所有bbox）
功能：自动匹配patch_xxxx_xxxx.tif与对应的patch_xxxx_xxxx.tif.json文件，
      提取JSON中所有bbox_2d坐标，为每个bbox生成独立的带地理信息的新Patch，
      并记录每个新Patch的左上角和右下角坐标
"""

import os
import json
import re
import math
from osgeo import gdal, osr


def crop_bbox_from_patch(patch_path, bbox, output_path):
    """从单张Patch中剪切指定bbox区域，生成带地理信息的新Patch"""
    dataset = gdal.Open(patch_path)
    if not dataset:
        raise FileNotFoundError(f"无法打开Patch图像: {patch_path}")
    
    try:
        # 获取原始地理变换参数
        geotransform = dataset.GetGeoTransform()
        if not geotransform:
            raise ValueError(f"Patch {os.path.basename(patch_path)} 无地理变换信息")
        
        # 获取图像基本信息
        bands = dataset.RasterCount
        patch_width = dataset.RasterXSize
        patch_height = dataset.RasterYSize
        data_type = dataset.GetRasterBand(1).DataType
        
        # 处理bbox参数（转换为整数像素索引）
        xmin, ymin, xmax, ymax = bbox
        start_x, start_y = int(round(xmin)), int(round(ymin))
        end_x, end_y = int(round(xmax)), int(round(ymax))
        
        # 验证bbox有效性
        if start_x < 0 or start_y < 0 or end_x > patch_width or end_y > patch_height:
            raise ValueError(
                f"bbox超出Patch范围: \n"
                f"Patch尺寸: {patch_width}x{patch_height} \n"
                f"bbox像素范围: ({start_x}, {start_y}, {end_x}, {end_y})"
            )
        if start_x >= end_x or start_y >= end_y:
            raise ValueError(f"bbox参数无效: {bbox} (xmin >= xmax 或 ymin >= ymax)")
        
        # 计算新Patch尺寸
        new_width = end_x - start_x
        new_height = end_y - start_y
        
        # 读取bbox区域数据（逐波段）
        patch_data = []
        for band_idx in range(1, bands + 1):
            band = dataset.GetRasterBand(band_idx)
            data = band.ReadAsArray(start_x, start_y, new_width, new_height)
            if data is None:
                raise RuntimeError(f"无法读取第{band_idx}波段的bbox区域数据")
            patch_data.append(data)
        
        # 创建输出数据集（带地理参考）
        driver = gdal.GetDriverByName("GTiff")
        if not driver:
            raise RuntimeError("无法获取GTiff驱动")
        
        out_dataset = driver.Create(
            output_path,
            new_width,
            new_height,
            bands,
            data_type,
            options=["PROFILE=GeoTIFF", "TFW=YES", "COMPRESS=LZW"]
        )
        if not out_dataset:
            raise RuntimeError(f"无法创建输出文件: {output_path}")
        
        # 计算并设置新的地理变换
        new_geotransform = list(geotransform)
        new_geotransform[0] = geotransform[0] + start_x * geotransform[1]  # 新左上角x
        new_geotransform[3] = geotransform[3] + start_y * geotransform[5]  # 新左上角y
        out_dataset.SetGeoTransform(new_geotransform)
        
        # 设置坐标系并解析EPSG编码
        projection = dataset.GetProjection()
        epsg_code = "未知"
        if projection:
            out_dataset.SetProjection(projection)
            srs = osr.SpatialReference(wkt=projection)
            if srs.ImportFromWkt(projection) == 0:
                srs.AutoIdentifyEPSG()
                epsg_code = srs.GetAttrValue("AUTHORITY", 1) or "未知"
        
        # 写入波段数据并复制元数据
        for band_idx in range(bands):
            out_band = out_dataset.GetRasterBand(band_idx + 1)
            out_band.WriteArray(patch_data[band_idx])
            
            original_band = dataset.GetRasterBand(band_idx + 1)
            out_band.SetMetadata(original_band.GetMetadata())
            
            nodata = original_band.GetNoDataValue()
            if nodata is not None:
                try:
                    out_band.SetNoDataValue(float(nodata))
                except:
                    pass  # 忽略无法转换的NoData值
            
            out_band.FlushCache()
        
        out_dataset.FlushCache()
        
        # 计算新Patch的左上角和右下角坐标
        top_left_x = new_geotransform[0]
        top_left_y = new_geotransform[3]
        bottom_right_x = top_left_x + new_width * new_geotransform[1]
        bottom_right_y = top_left_y + new_height * new_geotransform[5]
        
        return {
            "new_patch_path": output_path,
            "geotransform": new_geotransform,
            "epsg_code": epsg_code,
            "size_pixel": {"width": new_width, "height": new_height},
            "corner_coordinates": {
                "top_left": {"x": round(top_left_x, 6), "y": round(top_left_y, 6)},
                "bottom_right": {"x": round(bottom_right_x, 6), "y": round(bottom_right_y, 6)}
            },
            "source_bbox": bbox
        }
    
    finally:
        # 释放资源
        dataset = None
        if 'out_dataset' in locals() and out_dataset:
            out_dataset = None


def extract_all_bboxes_from_json(json_path):
    """从JSON文件中提取所有bbox_2d坐标"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON文件不存在: {json_path}")
    
    try:
        # 读取JSON文件内容
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 提取raw_response字段
        raw_response = json_data.get('raw_response', '')
        if not raw_response:
            raise ValueError("JSON文件中没有'raw_response'字段")
        
        # 提取```json```代码块中的内容
        json_match = re.search(r'```json(.*?)```', raw_response, re.DOTALL)
        if not json_match:
            raise ValueError("raw_response中没有找到```json```代码块")
        
        # 清理内容
        json_content = json_match.group(1).strip()
        json_content = re.sub(r'<\|eot_id\|>.*$', '', json_content)
        
        # 解析JSON数组
        bbox_list = json.loads(json_content)
        if not isinstance(bbox_list, list) or len(bbox_list) == 0:
            raise ValueError("JSON代码块中没有有效的bbox数组")
        
        # 提取所有含bbox_2d的有效坐标
        all_bboxes = []
        for item in bbox_list:
            if isinstance(item, dict) and 'bbox_2d' in item:
                bbox = item['bbox_2d']
                if isinstance(bbox, list) and len(bbox) == 4:
                    valid_bbox = tuple(map(float, bbox))
                    all_bboxes.append(valid_bbox)
        
        # 验证是否提取到有效bbox
        if len(all_bboxes) == 0:
            raise ValueError("JSON代码块中没有有效的'bbox_2d'数据")
        
        return all_bboxes
    
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON格式错误: {str(e)} 在位置 {e.pos}（文件：{json_path}）")
    except Exception as e:
        raise ValueError(f"提取bbox失败: {str(e)}（文件：{json_path}）")


def batch_process_all_bboxes(patch_dir, json_dir, output_root_dir):
    """批量处理所有Patch，提取每个JSON中的所有bbox并生成新Patch"""
    # 创建输出目录结构
    output_patch_root = os.path.join(output_root_dir, "cropped_patches_by_bbox")
    output_result_dir = os.path.join(output_root_dir, "all_bboxes_processing_results")
    os.makedirs(output_patch_root, exist_ok=True)
    os.makedirs(output_result_dir, exist_ok=True)
    
    # 获取所有符合格式的Patch文件
    patch_files = [f for f in os.listdir(patch_dir)
                  if f.startswith("patch_") and f.endswith(".tif")
                  and re.match(r'patch_\d{4}_\d{4}\.tif', f)]
    
    if not patch_files:
        print(f"未在 {patch_dir} 中找到符合格式的Patch文件 (patch_xxxx_xxxx.tif)")
        return
    
    # 初始化统计变量
    total_patch = len(patch_files)
    total_bbox_processed = 0
    total_bbox_success = 0
    total_bbox_fail = 0
    fail_records = []
    
    # 打印配置信息
    print("="*80)
    print("开始批量处理（提取所有bbox）")
    print("="*80)
    print(f"原始Patch文件夹：{patch_dir}")
    print(f"JSON文件夹：{json_dir}")
    print(f"新Patch输出根目录：{output_patch_root}")
    print(f"结果记录输出目录：{output_result_dir}")
    print(f"待处理原始Patch数：{total_patch}")
    print("="*80 + "\n")
    
    # 批量处理每个原始Patch
    for patch_idx, patch_filename in enumerate(patch_files, 1):
        patch_basename = os.path.splitext(patch_filename)[0]
        json_filename = f"{patch_basename}.tif.json"
        patch_path = os.path.join(patch_dir, patch_filename)
        json_path = os.path.join(json_dir, json_filename)
        
        print(f"[{patch_idx}/{total_patch}] 处理原始Patch：{patch_filename}")
        print(f"  对应JSON文件：{json_filename}")
        
        # 检查JSON文件是否存在
        if not os.path.exists(json_path):
            msg = "对应的JSON文件不存在"
            print(f"  ❌ {msg}，跳过该Patch\n")
            fail_records.append({
                "type": "patch",
                "patch_filename": patch_filename,
                "reason": msg,
                "json_filename": json_filename
            })
            continue
        
        try:
            # 提取所有bbox
            all_bboxes = extract_all_bboxes_from_json(json_path)
            bbox_count = len(all_bboxes)
            print(f"  ✅ 成功提取到 {bbox_count} 个有效bbox")
            
            # 创建当前Patch的专属输出目录
            patch_output_dir = os.path.join(output_patch_root, patch_basename)
            os.makedirs(patch_output_dir, exist_ok=True)
            
            # 记录当前Patch的所有处理结果
            patch_result = {
                "processing_time": os.popen('date /t').read().strip() + " " + os.popen('time /t').read().strip(),
                "input_info": {
                    "original_patch_path": patch_path,
                    "matched_json_path": json_path,
                    "total_extracted_bboxes": bbox_count,
                    "all_extracted_bboxes": all_bboxes
                },
                "bbox_process_results": []
            }
            
            # 逐个处理每个bbox
            for bbox_idx, bbox in enumerate(all_bboxes, 1):
                print(f"  处理第 {bbox_idx}/{bbox_count} 个bbox：{bbox}")
                
                try:
                    # 生成新Patch文件名
                    new_patch_filename = f"{patch_basename}_bbox_{bbox_idx:02d}.tif"
                    new_patch_path = os.path.join(patch_output_dir, new_patch_filename)
                    
                    # 剪切并获取结果
                    bbox_crop_result = crop_bbox_from_patch(patch_path, bbox, new_patch_path)
                    
                    # 记录结果
                    bbox_result = {
                        "bbox_index": bbox_idx,
                        "source_bbox": bbox,
                        "new_patch_path": new_patch_path,
                        "processing_status": "success",
                        "crop_result": bbox_crop_result
                    }
                    patch_result["bbox_process_results"].append(bbox_result)
                    
                    print(f"    ✅ 成功生成新Patch：{new_patch_filename}")
                    print(f"    📍 左上角坐标：x={bbox_crop_result['corner_coordinates']['top_left']['x']:.2f}, y={bbox_crop_result['corner_coordinates']['top_left']['y']:.2f}")
                    print(f"    📍 右下角坐标：x={bbox_crop_result['corner_coordinates']['bottom_right']['x']:.2f}, y={bbox_crop_result['corner_coordinates']['bottom_right']['y']:.2f}")
                    
                    total_bbox_success += 1
                
                except Exception as e:
                    bbox_fail_msg = str(e)
                    bbox_result = {
                        "bbox_index": bbox_idx,
                        "source_bbox": bbox,
                        "processing_status": "fail",
                        "fail_reason": bbox_fail_msg
                    }
                    patch_result["bbox_process_results"].append(bbox_result)
                    
                    print(f"    ❌ 处理失败：{bbox_fail_msg}")
                    
                    total_bbox_fail += 1
                    fail_records.append({
                        "type": "bbox",
                        "patch_filename": patch_filename,
                        "bbox_index": bbox_idx,
                        "source_bbox": bbox,
                        "reason": bbox_fail_msg
                    })
            
            # 保存当前Patch的处理结果
            patch_result_filename = f"{patch_basename}_all_bboxes_result.json"
            patch_result_path = os.path.join(output_result_dir, patch_result_filename)
            with open(patch_result_path, 'w', encoding='utf-8') as f:
                json.dump(patch_result, f, ensure_ascii=False, indent=4)
            
            # 打印当前Patch处理总结
            total_bbox_processed += bbox_count
            success_in_patch = len([r for r in patch_result['bbox_process_results'] if r['processing_status']=='success'])
            fail_in_patch = len([r for r in patch_result['bbox_process_results'] if r['processing_status']=='fail'])
            print(f"\n  📊 该Patch处理总结：共{bbox_count}个bbox，成功{success_in_patch}个，失败{fail_in_patch}个")
            print(f"  📄 结果记录已保存至：{patch_result_filename}\n")
        
        except Exception as e:
            patch_fail_msg = str(e)
            print(f"  ❌ 提取bbox失败：{patch_fail_msg}，跳过该Patch\n")
            fail_records.append({
                "type": "patch",
                "patch_filename": patch_filename,
                "reason": f"提取bbox失败：{patch_fail_msg}",
                "json_filename": json_filename
            })
            continue
    
    # 打印最终处理总结
    print("="*80)
    print("批量处理完成（提取所有bbox）")
    print("="*80)
    print(f"原始Patch处理情况：共{total_patch}个，成功处理{total_patch - len([r for r in fail_records if r['type']=='patch'])}个，失败{len([r for r in fail_records if r['type']=='patch'])}个")
    print(f"bbox处理情况：共处理{total_bbox_processed}个，成功{total_bbox_success}个，失败{total_bbox_fail}个")
    print(f"新Patch保存目录：{output_patch_root}")
    print(f"结果记录保存目录：{output_result_dir}")
    
    # 保存全局失败记录
    if fail_records:
        global_fail_log_path = os.path.join(output_root_dir, "all_bboxes_process_fail_log.json")
        with open(global_fail_log_path, 'w', encoding='utf-8') as f:
            json.dump(fail_records, f, ensure_ascii=False, indent=4)
        print(f"\n⚠️  全局失败记录已保存至：{global_fail_log_path}")
    print("="*80)


if __name__ == "__main__":
    # 配置参数 - 请根据实际情况修改以下路径
    PATCH_DIR = r"D:\Remote Sensing\Shenzhen\RSI ( final version )"  # 存放patch_xxxx_xxxx.tif的文件夹
    JSON_DIR = r"D:\Remote Sensing\Shenzhen\250910Qwen2.5-VL-72B-Instruct\home\yuling\test\test_result\250910Qwen2.5-VL-72B-Instruct"  # 存放对应的json文件的文件夹
    OUTPUT_ROOT_DIR = r"D:\Remote Sensing\Shenzhen\cropped_patches"  # 输出结果的根目录
    
    # 执行批量处理
    batch_process_all_bboxes(PATCH_DIR, JSON_DIR, OUTPUT_ROOT_DIR)