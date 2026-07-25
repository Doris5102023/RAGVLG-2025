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
        if coord > 512:
            raise ValueError(f"{source}边界框第{i+1}个坐标值({coord})超过512，不符合要求")
    
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
                    # print(f"警告：{file_path} 包含多个JSON对象，已尝试提取第一个")
            
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

def extract_json_from_raw_response(raw_response):
    """从raw_response中提取JSON内容，处理可能的代码块标记"""
    # 移除可能的代码块标记
    cleaned_response = re.sub(r'^```json\s*|\s*```$', '', raw_response, flags=re.MULTILINE)
    
    # 尝试提取JSON数组
    start_idx = cleaned_response.find('[')
    end_idx = cleaned_response.rfind(']') + 1 if start_idx != -1 else 0
    
    if start_idx != -1 and end_idx > start_idx:
        return cleaned_response[start_idx:end_idx]
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
            
        for item in pred_list:
            if 'bbox_2d' in item:
                valid_bbox = validate_bbox_coords(item['bbox_2d'], f"预测文件 {os.path.basename(pred_path)} raw_response")
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
    gt_folder_path = r"E:\RemoteSensing\gt_truth"
    pred_folder_path = r"D:\RemoteSensing\工业用地72B GRAG"  # 可更换为其他预测文件夹
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
