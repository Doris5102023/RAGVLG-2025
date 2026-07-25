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

def validate_bbox_coords(bbox, source):
    """验证边界框坐标是否有效（仅检查长度和数值范围，不验证是否为数字）"""
    if len(bbox) != 4:
        raise ValueError(f"{source}边界框格式错误，必须包含4个坐标值：{bbox}")
    
    for i, coord in enumerate(bbox):
        if coord > 512 or coord < 0:
            raise ValueError(f"{source}边界框第{i+1}个坐标值({coord})不符合要求")
    
    # 返回原始边界框
    return bbox

def safe_json_load(file_path):
    """安全加载JSON文件，处理常见的JSON格式错误，包括额外非JSON内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # 处理可能包含的代码块标记（如```json）
            if content.startswith('```'):
                print(f"警告：{file_path} 包含代码块标记，尝试移除...")
                content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.MULTILINE)
            
            # 处理可能的多JSON对象问题
            if content.count('{') > 1 or content.count('}') > 1:
                first_open = content.find('{')
                last_close = content.rfind('}')
                if first_open != -1 and last_close != -1:
                    content = content[first_open:last_close+1]
            
            # 处理可能的尾随逗号问题
            content = re.sub(r',\s*([}\]])', r'\1', content)
            
            return json.loads(content)
    except json.JSONDecodeError as e:
        try:
            import demjson3
            return demjson3.decode_file(file_path)
        except:
            raise ValueError(f"JSON解析失败：{str(e)}，文件内容可能存在格式错误")
    except Exception as e:
        raise ValueError(f"读取文件失败：{str(e)}")

import json
import re

import json
import re

def extract_json_from_raw_response(raw_response):
    """从raw_response中提取提取JSON内容，处理可能的代码块标记，并区分industrial_land的五种格式"""
    # 移除可能的代码块标记（支持带json标识和不带标识的情况）
    cleaned_response = re.sub(r'^```(json)?\s*|\s*```$', '', raw_response, flags=re.MULTILINE)
    
    try:
        # 尝试解析整个内容
        data = json.loads(cleaned_response)
        
        # 优先处理industrial_land字段的五种格式
        if 'industrial_land' in data and isinstance(data['industrial_land'], list):
            converted = []
            for item in data['industrial_land']:
                if isinstance(item, dict):
                    # 格式1: 包含top_left和bottom_right
                    if 'top_left' in item and 'bottom_right' in item:
                        tl = item['top_left']
                        br = item['bottom_right']
                        if isinstance(tl, dict) and isinstance(br, dict) and \
                           'x' in tl and 'y' in tl and 'x' in br and 'y' in br:
                            converted.append([tl['x'], tl['y'], br['x'], br['y']])
                    
                    # 格式2: 包含x_min/y_min/x_max/y_max
                    elif all(k in item for k in ['x_min', 'y_min', 'x_max', 'y_max']):
                        converted.append([
                            item['x_min'],
                            item['y_min'],
                            item['x_max'],
                            item['y_max']
                        ])
                    
                    # 格式3: 包含bbox数组（直接坐标数组）
                    elif 'bbox' in item and isinstance(item['bbox'], list) and len(item['bbox']) == 4:
                        converted.append(item['bbox'])
                    
                    # 格式4: 包含coordinates数组
                    elif 'coordinates' in item and isinstance(item['coordinates'], list) and len(item['coordinates']) == 4:
                        converted.append(item['coordinates'])
            
            if converted:
                return json.dumps(converted)
        
        # 新增处理industrial_land为对象且包含bbox数组（内部是xmin/ymin等字典）的格式
        if 'industrial_land' in data and isinstance(data['industrial_land'], dict):
            industrial_land = data['industrial_land']
            if 'bbox' in industrial_land and isinstance(industrial_land['bbox'], list):
                converted = []
                for bbox_item in industrial_land['bbox']:
                    if isinstance(bbox_item, dict) and all(k in bbox_item for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                        converted.append([
                            bbox_item['xmin'],
                            bbox_item['ymin'],
                            bbox_item['xmax'],
                            bbox_item['ymax']
                        ])
                if converted:
                    return json.dumps(converted)
        
        # 处理industrial_land_use字段的三种格式
        if 'industrial_land_use' in data:
            use_data = data['industrial_land_use']
            converted = []
            
            # 情况1: industrial_land_use是数组（包含多个对象）
            if isinstance(use_data, list):
                for item in use_data:
                    if isinstance(item, dict):
                        # 子情况1a: 包含top_left和bottom_right
                        if 'top_left' in item and 'bottom_right' in item:
                            tl = item['top_left']
                            br = item['bottom_right']
                            if isinstance(tl, dict) and isinstance(br, dict) and \
                               'x' in tl and 'y' in tl and 'x' in br and 'y' in br:
                                converted.append([tl['x'], tl['y'], br['x'], br['y']])
                        # 子情况1b: 包含bbox数组
                        elif 'bbox' in item and isinstance(item['bbox'], list) and len(item['bbox']) == 4:
                            converted.append(item['bbox'])
                        # 子情况1c: 包含x_min/y_min/x_max/y_max
                        elif all(k in item for k in ['x_min', 'y_min', 'x_max', 'y_max']):
                            converted.append([
                                item['x_min'],
                                item['y_min'],
                                item['x_max'],
                                item['y_max']
                            ])
            
            # 情况2: industrial_land_use是单个对象（包含x_min等字段）
            elif isinstance(use_data, dict):
                if all(k in use_data for k in ['x_min', 'y_min', 'x_max', 'y_max']):
                    converted.append([
                        use_data['x_min'],
                        use_data['y_min'],
                        use_data['x_max'],
                        use_data['y_max']
                    ])
                # 补充处理对象中包含bbox的情况
                elif 'bbox' in use_data and isinstance(use_data['bbox'], list):
                    bbox = use_data['bbox']
                    if isinstance(bbox[0], list):  # 嵌套数组如[[x1,y1,x2,y2]]
                        converted.extend([b for b in bbox if len(b) == 4])
                    elif len(bbox) == 4:  # 一维数组如[x1,y1,x2,y2]
                        converted.append(bbox)
            
            if converted:
                return json.dumps(converted)
        
        # 处理industrial_uses字段
        if 'industrial_uses' in data:
            uses_data = data['industrial_uses']
            converted = []
            
            # industrial_uses是数组（包含多个对象）
            if isinstance(uses_data, list):
                for item in uses_data:
                    if isinstance(item, dict):
                        # 处理包含coordinates对象（内部有x_min等字段）的情况
                        if 'coordinates' in item and isinstance(item['coordinates'], dict):
                            coord = item['coordinates']
                            if all(k in coord for k in ['x_min', 'y_min', 'x_max', 'y_max']):
                                converted.append([
                                    coord['x_min'],
                                    coord['y_min'],
                                    coord['x_max'],
                                    coord['y_max']
                                ])
                        # 处理包含bbox数组的情况
                        elif 'bbox' in item and isinstance(item['bbox'], list) and len(item['bbox']) == 4:
                            converted.append(item['bbox'])
            
            if converted:
                return json.dumps(converted)
        
        # 处理顶级字段包含x_min,y_min,x_max,y_max的对象（如{"industrial_land_bbox": {...}}）
        for key in data:
            if isinstance(data[key], dict) and all(k in data[key] for k in ['x_min', 'y_min', 'x_max', 'y_max']):
                return json.dumps([[
                    data[key]['x_min'],
                    data[key]['y_min'],
                    data[key]['x_max'],
                    data[key]['y_max']
                ]])
        
        # 处理顶级字段包含bbox的对象（如{"industrial_land_use": {"bbox": [...]}}）
        for key in data:
            if isinstance(data[key], dict) and 'bbox' in data[key]:
                bbox_data = data[key]['bbox']
                converted = []
                if isinstance(bbox_data, list) and len(bbox_data) > 0:
                    if isinstance(bbox_data[0], list):  # 二维数组
                        converted.extend([b for b in bbox_data if len(b) == 4])
                    elif len(bbox_data) == 4:  # 一维数组
                        converted.append(bbox_data)
                return json.dumps(converted)
        
        # 处理顶级bbox字段
        if 'bbox' in data:
            bbox_data = data['bbox']
            converted = []
            
            # 新增处理: bbox是包含xmin, ymin, xmax, ymax的字典（无下划线格式）
            if isinstance(bbox_data, dict) and all(k in bbox_data for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                converted.append([
                    bbox_data['xmin'],
                    bbox_data['ymin'],
                    bbox_data['xmax'],
                    bbox_data['ymax']
                ])
            
            # 风格1: bbox是对象列表，每个对象包含x, y, width, height
            elif isinstance(bbox_data, list) and len(bbox_data) > 0 and all(isinstance(item, dict) for item in bbox_data):
                for item in bbox_data:
                    if 'x' in item and 'y' in item and 'width' in item and 'height' in item:
                        x1 = item['x']
                        y1 = item['y']
                        x2 = x1 + item['width']
                        y2 = y1 + item['height']
                        converted.append([x1, y1, x2, y2])
            
            # 风格2: bbox是[x, y, width, height]或[x1, y1, x2, y2]数组
            elif isinstance(bbox_data, list) and len(bbox_data) == 4:
                if all(isinstance(v, (int, float)) for v in bbox_data):
                    if bbox_data[2] > bbox_data[0] and bbox_data[3] > bbox_data[1]:
                        converted.append(bbox_data)
                    else:  # 视为[x, y, width, height]
                        x1, y1, width, height = bbox_data
                        converted.append([x1, y1, x1 + width, y1 + height])
            
            # 风格3: bbox是包含x_min, y_min, x_max, y_max的对象（有下划线格式）
            elif isinstance(bbox_data, dict) and all(k in bbox_data for k in ['x_min', 'y_min', 'x_max', 'y_max']):
                converted.append([
                    bbox_data['x_min'], 
                    bbox_data['y_min'], 
                    bbox_data['x_max'], 
                    bbox_data['y_max']
                ])
            
            return json.dumps(converted)
        
        # 没有找到预期格式，返回空数组
        return '[]'
        
    except json.JSONDecodeError:
        # 尝试提取JSON片段
        start_idx = cleaned_response.find('{')
        end_idx = cleaned_response.rfind('}') + 1 if start_idx != -1 else 0
        
        if start_idx != -1 and end_idx > start_idx:
            try:
                data = json.loads(cleaned_response[start_idx:end_idx])
                return extract_json_from_raw_response(json.dumps(data))
            except:
                pass
                
        # 所有尝试失败，返回空数组
        return '[]'
        
def count_all_gt_bboxes(gt_folder, gt_prefix="result"):
    """独立统计所有真实框，仅依赖真实标注文件夹"""
    total_gt = 0
    gt_files_info = {}  # 存储每个文件的真实框信息
    
    for filename in os.listdir(gt_folder):
        if filename.endswith('.json'):
            file_id = get_number_id(filename, gt_prefix)
            if file_id:
                file_path = os.path.join(gt_folder, filename)
                try:
                    gt_data = safe_json_load(file_path)
                    bboxes = []
                    for bbox in gt_data.get('bboxes', []):
                        if 'bbox' in bbox:
                            valid_bbox = validate_bbox_coords(bbox['bbox'], f"真实标注文件 {filename}")
                            bboxes.append(valid_bbox)
                    
                    count = len(bboxes)
                    total_gt += count
                    gt_files_info[file_id] = {
                        'path': file_path,
                        'count': count,
                        'bboxes': bboxes
                    }
                except Exception as e:
                    print(f"处理真实标注文件 {filename} 时出错: {str(e)}")
    
    return total_gt, gt_files_info

def process_pred_file(pred_path, file_id, pred_prefix="predicted"):
    """处理单个预测文件，返回预测框及数量"""
    try:
        pred_data = safe_json_load(pred_path)
    except Exception as e:
        raise ValueError(f"预测结果文件解析错误：{str(e)}")
    
    pred_bboxes = []
    
    if 'bboxes' in pred_data:
        for bbox in pred_data['bboxes']:
            if 'bbox' in bbox:
                valid_bbox = validate_bbox_coords(bbox['bbox'], f"预测文件 {os.path.basename(pred_path)}")
                pred_bboxes.append(valid_bbox)
    else:
        raw_response = pred_data.get('raw_response', '[]')
        # 增强处理：提取并清理raw_response中的JSON内容
        pred_json = extract_json_from_raw_response(raw_response)
        
        try:
            pred_list = json.loads(pred_json)
        except json.JSONDecodeError:
            pred_json = re.sub(r',\s*([}\]])', r'\1', pred_json)
            pred_list = json.loads(pred_json)
            
        # 直接处理转换后的边界框列表
        for item in pred_list:
            if isinstance(item, list) and len(item) == 4:
                valid_bbox = validate_bbox_coords(item, f"预测文件 {os.path.basename(pred_path)} raw_response")
                pred_bboxes.append(valid_bbox)
    
    return pred_bboxes, len(pred_bboxes)

def calculate_pair_metrics(gt_bboxes, pred_bboxes, iou_threshold=0.3):
    """计算单对文件的匹配情况，返回正确匹配数和匹配的真实框索引"""
    correct = 0
    matched_gt = set()  # 记录已匹配的真实框索引
    
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
        fieldnames = ['file_id', 'box_id', 'x1', 'y1', 'x2', 'y2', 'source', 'is_matched']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for bbox in bboxes:
            writer.writerow(bbox)
    
    bbox_type = "真实标注" if is_gt else "预测结果"
    print(f"\n已将{len(bboxes)}个{bbox_type}边界框保存到: {output_file}")

def main(gt_folder, pred_folder, iou_threshold=0.3, output_dir="bbox_results"):
    """主函数：统计真实框、预测框并计算精度和召回率"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 独立统计所有真实框（仅依赖gt_folder，与预测文件夹无关）
    total_gt_all, gt_files_info = count_all_gt_bboxes(gt_folder)
    print(f"===== 真实框统计完成 =====")
    print(f"真实标注文件夹: {gt_folder}")
    print(f"总真实框数量: {total_gt_all}")
    print(f"有效真实标注文件数: {len(gt_files_info)}\n")
    
    # 2. 收集预测文件信息
    pred_files = defaultdict(str)
    invalid_pred_files = []
    pred_prefix = "predicted"
    
    for filename in os.listdir(pred_folder):
        if filename.endswith('.json'):
            file_id = get_number_id(filename, pred_prefix)
            if file_id:
                pred_files[file_id] = os.path.join(pred_folder, filename)
            else:
                invalid_pred_files.append(filename)
    
    print(f"===== 预测文件收集完成 =====")
    print(f"预测结果文件夹: {pred_folder}")
    print(f"有效预测文件数: {len(pred_files)}")
    print(f"无法解析的预测文件数: {len(invalid_pred_files)}\n")
    
    # 3. 找到匹配的文件对
    matched_ids = set(gt_files_info.keys()) & set(pred_files.keys())
    unmatched_gt_ids = set(gt_files_info.keys()) - matched_ids
    total_matched_gt = total_gt_all - sum(gt_files_info[id]['count'] for id in unmatched_gt_ids)
    
    print(f"===== 文件匹配情况 =====")
    print(f"匹配的文件对数: {len(matched_ids)}")
    print(f"有真实标注但无预测结果的文件数: {len(unmatched_gt_ids)}")
    print(f"可参与召回率计算的真实框数量: {total_matched_gt} (排除无预测结果的文件)\n")
    
    if not matched_ids:
        print("错误：未找到任何配对的文件，无法计算精度和召回率")
        return
    
    # 4. 处理匹配的文件对，统计预测框并计算指标
    total_pred_all = 0
    total_correct_all = 0  # 用于计算精度
    total_matched_gt_all = 0  # 用于计算召回率
    all_gt_bboxes = []  # 用于保存所有真实框详情
    all_pred_bboxes = []  # 用于保存所有预测框详情
    
    for file_id in sorted(matched_ids, key=int):
        # 获取真实框信息（已提前解析）
        gt_info = gt_files_info[file_id]
        gt_bboxes = gt_info['bboxes']
        gt_filename = os.path.basename(gt_info['path'])
        gt_count = gt_info['count']
        
        # 处理预测文件
        pred_path = pred_files[file_id]
        pred_filename = os.path.basename(pred_path)
        
        try:
            pred_bboxes, pred_count = process_pred_file(pred_path, file_id)
            total_pred_all += pred_count
            
            # 计算匹配情况
            correct_count, matched_gt_indices = calculate_pair_metrics(gt_bboxes, pred_bboxes, iou_threshold)
            total_correct_all += correct_count
            total_matched_gt_all += len(matched_gt_indices)
            
            # 保存边界框详情（包含是否匹配的标记）
            for idx, bbox in enumerate(gt_bboxes):
                all_gt_bboxes.append({
                    'file_id': file_id,
                    'box_id': idx,
                    'x1': bbox[0],
                    'y1': bbox[1],
                    'x2': bbox[2],
                    'y2': bbox[3],
                    'source': gt_filename,
                    'is_matched': idx in matched_gt_indices
                })
            
            for idx, bbox in enumerate(pred_bboxes):
                # 判断该预测框是否匹配到真实框
                is_matched = False
                for gt_idx in matched_gt_indices:
                    if calculate_iou(gt_bboxes[gt_idx], bbox) >= iou_threshold:
                        is_matched = True
                        break
                
                all_pred_bboxes.append({
                    'file_id': file_id,
                    'box_id': idx,
                    'x1': bbox[0],
                    'y1': bbox[1],
                    'x2': bbox[2],
                    'y2': bbox[3],
                    'source': pred_filename,
                    'is_matched': is_matched
                })
            
            # 计算单文件对的精度和召回率
            precision = correct_count / pred_count if pred_count > 0 else 0
            recall = len(matched_gt_indices) / gt_count if gt_count > 0 else 0
            print(f"文件对 {file_id}: 真实框={gt_count}, 预测框={pred_count}, 正确匹配={correct_count}, 精度={precision:.4f}, 召回率={recall:.4f}")
            
        except Exception as e:
            print(f"处理文件对 {file_id} 时出错: {str(e)}")
            # 出错时继续处理下一个文件对
            continue
    
    # 5. 保存边界框详情到CSV
    save_bboxes_to_csv(all_gt_bboxes, os.path.join(output_dir, "all_gt_bboxes.csv"), is_gt=True)
    save_bboxes_to_csv(all_pred_bboxes, os.path.join(output_dir, "all_pred_bboxes.csv"), is_gt=False)
    
    # 6. 计算并显示总体指标
    overall_precision = total_correct_all / total_pred_all if total_pred_all != 0 else 0.0
    overall_recall = total_matched_gt_all / total_matched_gt if total_matched_gt != 0 else 0.0
    # 计算F1分数（精度和召回率的调和平均）
    f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) != 0 else 0.0
    
    print("\n===== 总体统计结果 =====")
    print(f"总匹配文件对数: {len(matched_ids)}")
    print(f"总真实框数量: {total_gt_all}")
    print(f"可参与召回率计算的真实框数量: {total_matched_gt}")
    print(f"总预测框数量: {total_pred_all}")
    print(f"总正确匹配数量: {total_correct_all}")
    print(f"IoU阈值: {iou_threshold}")
    print(f"总体精度 (Precision): {overall_precision:.4f}")
    print(f"总体召回率 (Recall): {overall_recall:.4f}")
    print(f"F1分数 (F1-Score): {f1_score:.4f}")
    
    return total_gt_all, total_pred_all, overall_precision, overall_recall, f1_score

if __name__ == "__main__":
    # 配置路径
    gt_folder_path = "D:\\Remote Sensing\\gt_truth\\工业用地"
    pred_folder_path = r"D:\Remote Sensing\5880 results\home\yuling\test\test_result\250917GLM4.5v"  # 可更换为其他预测文件夹
    output_directory = "bbox_metrics_results"
    
    # 验证路径
    if not os.path.exists(gt_folder_path):
        print(f"错误：真实标注文件夹不存在 - {gt_folder_path}")
    elif not os.path.exists(pred_folder_path):
        print(f"错误：预测结果文件夹不存在 - {pred_folder_path}")
    else:
        main(
            gt_folder=gt_folder_path,
            pred_folder=pred_folder_path,
            iou_threshold=0.3,
            output_dir=output_directory
        )