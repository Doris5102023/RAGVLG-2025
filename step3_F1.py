import json
import os
import re
import csv
from collections import defaultdict

def calculate_iou(gt_bbox, pred_bbox):
    """计算两个边界框的交并比（IoU）"""
    x1 = max(gt_bbox[0], pred_bbox[0])
    y1 = max(gt_bbox[1], pred_bbox[1])
    x2 = min(gt_bbox[2], pred_bbox[2])
    y2 = min(gt_bbox[3], pred_bbox[3])
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    union_area = gt_area + pred_area - intersection_area
    
    return intersection_area / union_area if union_area != 0 else 0

def get_number_id(filename, prefix):
    """从文件名中提取数字编号"""
    pattern = re.compile(f'{prefix}_(\\d+)\\.json')
    match = pattern.match(filename)
    if match:
        return match.group(1)
    return None

def validate_bbox_coords(bbox, source, file_id, strict_mode=True):
    """验证边界框坐标是否有效，返回验证结果和边界框"""
    result = {
        'valid': True,
        'bbox': bbox,
        'error': None,
        'source': source,
        'file_id': file_id
    }
    
    # 检查是否包含4个坐标值
    if len(bbox) != 4:
        result['valid'] = False
        result['error'] = f"边界框格式错误，必须包含4个坐标值，实际有{len(bbox)}个"
        return result
    
    # 检查坐标是否为数字
    try:
        bbox = [float(coord) for coord in bbox]
        result['bbox'] = bbox
    except (ValueError, TypeError):
        result['valid'] = False
        result['error'] = "边界框坐标必须为数字"
        return result
    
    # 严格模式下检查范围和顺序
    if strict_mode:
        # 检查坐标范围是否有效（0-512或根据实际情况调整）
        for i, coord in enumerate(bbox):
            if coord < 0 or coord > 512:
                result['valid'] = False
                result['error'] = f"边界框第{i+1}个坐标值({coord})超出有效范围[0, 512]"
                return result
        
        # 检查x2 > x1和y2 > y1
        if bbox[2] <= bbox[0]:
            result['valid'] = False
            result['error'] = f"边界框x2({bbox[2]})必须大于x1({bbox[0]})"
            return result
            
        if bbox[3] <= bbox[1]:
            result['valid'] = False
            result['error'] = f"边界框y2({bbox[3]})必须大于y1({bbox[1]})"
            return result
    
    return result

def extract_all_bbox_coordinates(text):
    """从文本中提取所有以方括号[]括起来的坐标数组"""
    pattern = r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]'
    matches = re.findall(pattern, text)
    
    bboxes = []
    for match in matches:
        try:
            bbox = [float(coord) for coord in match]
            bboxes.append(bbox)
        except:
            continue
    
    return bboxes

def safe_json_load(file_path):
    """安全加载JSON文件，尝试多种编码"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read().strip()
            return content, json.loads(content), encoding
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    
    # 所有编码尝试失败后，返回原始内容
    try:
        with open(file_path, 'rb') as f:
            content = f.read().decode('utf-8', errors='replace')
        return content, None, None
    except:
        return "", None, None

def parse_bbox_dict(bbox_dict):
    """解析包含xmin/ymin/xmax/ymax或x1/y1/x2/y2的边界框字典"""
    if all(key in bbox_dict for key in ['xmin', 'ymin', 'xmax', 'ymax']):
        return [
            bbox_dict['xmin'],
            bbox_dict['ymin'],
            bbox_dict['xmax'],
            bbox_dict['ymax']
        ]
    elif all(key in bbox_dict for key in ['x1', 'y1', 'x2', 'y2']):
        return [
            bbox_dict['x1'],
            bbox_dict['y1'],
            bbox_dict['x2'],
            bbox_dict['y2']
        ]
    return None

def count_all_gt_bboxes(gt_folder, gt_prefix="result", strict_validation=False):
    """统计所有真实标注的边界框，支持多种格式"""
    total_gt = 0
    gt_files_info = {}
    skipped_files = []
    empty_files = []
    
    print(f"\n开始统计真实标注文件: {gt_folder}")
    
    for filename in os.listdir(gt_folder):
        if filename.endswith('.json'):
            file_id = get_number_id(filename, gt_prefix)
            if not file_id:
                print(f"无法识别的文件名格式: {filename}")
                continue
                
            file_path = os.path.join(gt_folder, filename)
            try:
                content, gt_data, encoding = safe_json_load(file_path)
                if encoding:
                    print(f"处理文件: {filename} (编码: {encoding})")
                else:
                    print(f"处理文件: {filename} (编码未知)")
                
                bboxes = []
                validations = []
                extracted_count = 0
                
                # 1. 尝试从结构化数据中提取
                if gt_data:
                    # 处理单边界框字典格式: {"xmin":..., "ymin":..., "xmax":..., "ymax":...}
                    if isinstance(gt_data, dict):
                        # 检查是否为单个边界框字典
                        parsed_single_bbox = parse_bbox_dict(gt_data)
                        if parsed_single_bbox:
                            extracted_count += 1
                            validation = validate_bbox_coords(
                                parsed_single_bbox, 
                                f"真实标注文件 {filename} - 单边界框字典", 
                                file_id,
                                strict_mode=strict_validation
                            )
                            validations.append(validation)
                            if validation['valid']:
                                bboxes.append(validation['bbox'])
                    
                    # 处理类别-边界框列表格式: "工业用地": [{"xmin":...,}, ...]
                    if isinstance(gt_data, dict):
                        for key, value in gt_data.items():
                            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                                for item in value:
                                    parsed_bbox = parse_bbox_dict(item)
                                    if parsed_bbox:
                                        extracted_count += 1
                                        validation = validate_bbox_coords(
                                            parsed_bbox, 
                                            f"真实标注文件 {filename} - 类别[{key}]", 
                                            file_id,
                                            strict_mode=strict_validation
                                        )
                                        validations.append(validation)
                                        if validation['valid']:
                                            bboxes.append(validation['bbox'])
                    
                    # 处理传统的边界框字段: bboxes, industrial_land_bboxes等
                    bbox_fields = [
                        'bboxes', 'annotations', 'objects', 'ground_truth',
                        'industrial_land_bboxes'
                    ]
                    for field in bbox_fields:
                        if field in gt_data:
                            for item in gt_data[field]:
                                if isinstance(item, dict):
                                    parsed_bbox = parse_bbox_dict(item)
                                    if parsed_bbox:
                                        extracted_count += 1
                                        validation = validate_bbox_coords(
                                            parsed_bbox, 
                                            f"真实标注文件 {filename} - {field}", 
                                            file_id,
                                            strict_mode=strict_validation
                                        )
                                        validations.append(validation)
                                        if validation['valid']:
                                            bboxes.append(validation['bbox'])
                                        continue
                                
                                # 从'bbox'或'coordinates'键获取
                                if isinstance(item, dict):
                                    if 'bbox' in item:
                                        extracted_count += 1
                                        validation = validate_bbox_coords(
                                            item['bbox'], 
                                            f"真实标注文件 {filename} - {field}", 
                                            file_id,
                                            strict_mode=strict_validation
                                        )
                                        validations.append(validation)
                                        if validation['valid']:
                                            bboxes.append(validation['bbox'])
                                    elif 'coordinates' in item:
                                        extracted_count += 1
                                        validation = validate_bbox_coords(
                                            item['coordinates'], 
                                            f"真实标注文件 {filename} - {field}", 
                                            file_id,
                                            strict_mode=strict_validation
                                        )
                                        validations.append(validation)
                                        if validation['valid']:
                                            bboxes.append(validation['bbox'])
                
                # 2. 如果没有结构化数据，从文本中提取
                if not bboxes and content:
                    extracted_bboxes = extract_all_bbox_coordinates(content)
                    extracted_count = len(extracted_bboxes)
                    for bbox in extracted_bboxes:
                        validation = validate_bbox_coords(
                            bbox, 
                            f"真实标注文件 {filename} - 文本提取", 
                            file_id,
                            strict_mode=strict_validation
                        )
                        validations.append(validation)
                        if validation['valid']:
                            bboxes.append(validation['bbox'])
                
                # 记录空文件
                if extracted_count == 0:
                    empty_files.append(filename)
                    print(f"警告: 文件 {filename} 未提取到任何边界框")
                
                count = len(bboxes)
                total_gt += count
                gt_files_info[file_id] = {
                    'path': file_path,
                    'count': count,
                    'extracted_count': extracted_count,
                    'valid_count': count,
                    'bboxes': bboxes,
                    'validations': validations
                }
                
                print(f"文件 {filename}: 提取到{extracted_count}个，有效{count}个")
                
            except Exception as e:
                skipped_files.append(filename)
                print(f"处理文件 {filename} 时出错: {str(e)}")
    
    # 输出统计summary
    print("\n真实标注统计结果:")
    print(f"总文件数: {len(os.listdir(gt_folder))}")
    print(f"成功处理文件数: {len(gt_files_info)}")
    print(f"跳过文件数: {len(skipped_files)}")
    print(f"空文件数: {len(empty_files)}")
    print(f"总提取数量: {sum(info['extracted_count'] for info in gt_files_info.values())}")
    print(f"总有效数量: {total_gt}")
    
    return total_gt, gt_files_info

def process_pred_file(pred_path, file_id, pred_prefix="predicted"):
    """处理单个预测文件，支持多种边界框格式"""
    pred_filename = os.path.basename(pred_path)
    pred_bboxes = []
    all_bboxes = []
    
    try:
        content, pred_data, _ = safe_json_load(pred_path)
        texts_to_search = []
        
        if pred_data and 'raw_response' in pred_data:
            texts_to_search.append(str(pred_data['raw_response']))
        texts_to_search.append(content)
        
        extracted_bboxes = []
        # 1. 从结构化数据中提取边界框
        if pred_data:
            # 处理单边界框字典格式: {"xmin":..., "ymin":..., "xmax":..., "ymax":...}
            if isinstance(pred_data, dict):
                parsed_single_bbox = parse_bbox_dict(pred_data)
                if parsed_single_bbox:
                    extracted_bboxes.append(parsed_single_bbox)
            
            # 处理类别-边界框列表格式: "工业用地": [{"xmin":...,}, ...]
            if isinstance(pred_data, dict):
                for key, value in pred_data.items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                        for item in value:
                            parsed_bbox = parse_bbox_dict(item)
                            if parsed_bbox:
                                extracted_bboxes.append(parsed_bbox)
            
            # 处理传统的边界框字段
            bbox_fields = [
                'bboxes', 'annotations', 'objects', 
                'industrial_land_bboxes'
            ]
            for field in bbox_fields:
                if field in pred_data and isinstance(pred_data[field], list):
                    for item in pred_data[field]:
                        if isinstance(item, dict):
                            parsed_bbox = parse_bbox_dict(item)
                            if parsed_bbox:
                                extracted_bboxes.append(parsed_bbox)
        
        # 2. 从文本中提取边界框
        for text in texts_to_search:
            extracted_bboxes.extend(extract_all_bbox_coordinates(text))
        
        # 去重处理
        unique_bboxes = []
        seen = set()
        for bbox in extracted_bboxes:
            bbox_tuple = tuple(bbox)
            if bbox_tuple not in seen:
                seen.add(bbox_tuple)
                unique_bboxes.append(bbox)
        
        # 验证每个提取的边界框
        for idx, bbox in enumerate(unique_bboxes):
            validation = validate_bbox_coords(
                bbox, 
                f"预测文件 {pred_filename} 提取的框[{idx}]",
                file_id
            )
            all_bboxes.append(validation)
            if validation['valid']:
                pred_bboxes.append(validation['bbox'])
    
    except Exception as e:
        raise ValueError(f"处理预测文件 {pred_filename} 时出错:{str(e)}")
    
    return pred_bboxes, len(pred_bboxes), all_bboxes

def calculate_pair_metrics(gt_bboxes, pred_bboxes, iou_threshold=0.3):
    """计算单个文件的匹配指标"""
    correct = 0
    matched_gt = set()
    
    for pred_bbox in pred_bboxes:
        max_iou = 0
        best_gt_idx = -1
        
        for i, gt_bbox in enumerate(gt_bboxes):
            if i not in matched_gt:
                iou = calculate_iou(gt_bbox, pred_bbox)
                if iou > max_iou:
                    max_iou = iou
                    best_gt_idx = i
        
        if max_iou >= iou_threshold and best_gt_idx != -1:
            correct += 1
            matched_gt.add(best_gt_idx)
    
    return correct, matched_gt

def save_bboxes_to_csv(bboxes, output_file, is_gt=True):
    """将边界框信息保存到CSV文件"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['file_id', 'box_id', 'x1', 'y1', 'x2', 'y2', 'source', 'is_matched', 'is_valid', 'error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for bbox in bboxes:
            writer.writerow(bbox)
    
    bbox_type = "真实标注" if is_gt else "预测框"
    print(f"\n已将{len(bboxes)}个{bbox_type}边界框保存到: {output_file}")

def main(gt_folder, pred_folder, iou_threshold=0.3, output_dir="bbox_results", strict_validation=False):
    """主函数：统计真实标注和预测结果的准确率和召回率"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 统计所有真实标注
    total_gt_all, gt_files_info = count_all_gt_bboxes(
        gt_folder, 
        strict_validation=strict_validation
    )
    
    # 2. 收集预测文件信息
    pred_files = defaultdict(str)
    for filename in os.listdir(pred_folder):
        if filename.endswith('.json'):
            file_id = get_number_id(filename, "predicted")
            if file_id:
                pred_files[file_id] = os.path.join(pred_folder, filename)
    
    # 3. 找到匹配的文件对
    matched_ids = set(gt_files_info.keys()) & set(pred_files.keys())
    unmatched_gt_ids = set(gt_files_info.keys()) - matched_ids
    total_matched_gt = total_gt_all - sum(gt_files_info[id]['count'] for id in unmatched_gt_ids)
    
    # 4. 处理匹配的文件对
    total_pred_all = 0
    total_correct_all = 0
    total_matched_gt_all = 0
    all_gt_bboxes = []
    all_pred_bboxes = []
    
    for file_id in sorted(matched_ids, key=int):
        gt_info = gt_files_info[file_id]
        gt_bboxes = gt_info['bboxes']
        gt_filename = os.path.basename(gt_info['path'])
        gt_count = gt_info['count']
        
        pred_path = pred_files[file_id]
        pred_filename = os.path.basename(pred_path)
        
        try:
            pred_bboxes, pred_count, all_pred_validations = process_pred_file(pred_path, file_id)
            total_pred_all += pred_count
            
            correct_count, matched_gt_indices = calculate_pair_metrics(gt_bboxes, pred_bboxes, iou_threshold)
            total_correct_all += correct_count
            total_matched_gt_all += len(matched_gt_indices)
            
            # 记录真实标注
            for idx, validation in enumerate(gt_info['validations']):
                all_gt_bboxes.append({
                    'file_id': file_id,
                    'box_id': idx,
                    'x1': validation['bbox'][0] if validation['valid'] else None,
                    'y1': validation['bbox'][1] if validation['valid'] else None,
                    'x2': validation['bbox'][2] if validation['valid'] else None,
                    'y2': validation['bbox'][3] if validation['valid'] else None,
                    'source': gt_filename,
                    'is_matched': idx in matched_gt_indices and validation['valid'],
                    'is_valid': validation['valid'],
                    'error': validation['error']
                })
            
            # 记录预测框
            for idx, validation in enumerate(all_pred_validations):
                is_matched = False
                if validation['valid']:
                    for gt_idx in matched_gt_indices:
                        if calculate_iou(gt_bboxes[gt_idx], validation['bbox']) >= iou_threshold:
                            is_matched = True
                            break
                
                all_pred_bboxes.append({
                    'file_id': file_id,
                    'box_id': idx,
                    'x1': validation['bbox'][0] if validation['valid'] else None,
                    'y1': validation['bbox'][1] if validation['valid'] else None,
                    'x2': validation['bbox'][2] if validation['valid'] else None,
                    'y2': validation['bbox'][3] if validation['valid'] else None,
                    'source': pred_filename,
                    'is_matched': is_matched,
                    'is_valid': validation['valid'],
                    'error': validation['error']
                })
            
            precision = correct_count / pred_count if pred_count > 0 else 0
            recall = len(matched_gt_indices) / gt_count if gt_count > 0 else 0
            print(f"文件对 {file_id}: 真实框={gt_count}, 预测框={pred_count}, 正确匹配={correct_count}, 精确率={precision:.4f}, 召回率={recall:.4f}")
            
        except Exception as e:
            print(f"处理文件对 {file_id} 时出错: {str(e)}")
            continue
    
    # 保存结果到CSV
    save_bboxes_to_csv(all_gt_bboxes, os.path.join(output_dir, "all_gt_bboxes.csv"), is_gt=True)
    save_bboxes_to_csv(all_pred_bboxes, os.path.join(output_dir, "all_pred_bboxes.csv"), is_gt=False)
    
    # 计算总指标
    overall_precision = total_correct_all / total_pred_all if total_pred_all > 0 else 0
    overall_recall = total_matched_gt_all / total_matched_gt if total_matched_gt > 0 else 0
    f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print("\n===== 总体统计 =====")
    print(f"总真实框数: {total_matched_gt}")
    print(f"总预测框数: {total_pred_all}")
    print(f"总正确匹配数: {total_correct_all}")
    print(f"精确率: {overall_precision:.4f}")
    print(f"召回率: {overall_recall:.4f}")
    print(f"F1分数: {f1_score:.4f}")

if __name__ == "__main__":
    # 配置路径（请根据实际情况修改）
    gt_folder_path = ""
    pred_folder_path = ""
    output_directory = "bbox_metrics_results"
    
    # 验证路径
    if not os.path.exists(gt_folder_path):
        print(f"真实标注文件夹不存在 - {gt_folder_path}")
    elif not os.path.exists(pred_folder_path):
        print(f"预测结果文件夹不存在 - {pred_folder_path}")
    else:
        main(
            gt_folder=gt_folder_path,
            pred_folder=pred_folder_path,
            iou_threshold=0.3,
            output_dir=output_directory,
            strict_validation=False
        )
    
