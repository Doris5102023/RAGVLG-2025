import os
import re
import json
from typing import List, Tuple, Dict, Any

# ----------------------------
# 工具函数
# ----------------------------

def parse_bboxes_from_response(response: str) -> List[List[int]]:
    """从 response 字符串中提取所有 [x1,y1,x2,y2] 形式的 bbox"""
    # 匹配形如 [49, 52, 53, 56] 或 [44,78,58,92] 的模式
    pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    matches = re.findall(pattern, response)
    bboxes = []
    for m in matches:
        x1, y1, x2, y2 = map(int, m)
        # 确保坐标合法（x2 > x1, y2 > y1）
        if x2 > x1 and y2 > y1:
            bboxes.append([x1, y1, x2, y2])
        else:
            # 跳过非法框
            pass
    return bboxes

def extract_id_from_pred_filename(filename: str) -> str:
    """从 'crsim_2712_processed.png' 提取 '2712'"""
    match = re.search(r'crsim_(\d+)_processed', filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"无法从文件名提取ID: {filename}")

def load_ground_truth(gt_dir: str, img_id: str) -> List[List[int]]:
    """加载 gt/result {id}.json，正确解析嵌套 bbox"""
    gt_path = os.path.join(gt_dir, f"result_{img_id}.json")
    if not os.path.exists(gt_path):
        print(f"⚠️ 警告: 找不到 ground truth 文件: {gt_path}")
        return []

    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    gt_boxes = []

    if "bboxes" in data and isinstance(data["bboxes"], list):
        for item in data["bboxes"]:
            if isinstance(item, dict) and "bbox" in item:
                bbox = item["bbox"]
                if isinstance(bbox, list) and len(bbox) == 4:
                    gt_boxes.append([int(coord) for coord in bbox])
                else:
                    print(f"⚠️ 警告: bbox 格式无效: {bbox} (文件: {gt_path})")
            else:
                print(f"⚠️ 警告: bboxes 列表中的项缺少 'bbox' 字段: {item}")

    elif "bbox" in data:
        bbox = data["bbox"]
        if isinstance(bbox, list) and len(bbox) == 4:
            gt_boxes.append([int(coord) for coord in bbox])
        else:
            print(f"⚠️ 警告: bbox 格式无效: {bbox} (文件: {gt_path})")

    else:
        print(f"⚠️ 警告: {gt_path} 中未找到 'bbox' 或 'bboxes' 字段")

    return gt_boxes

def compute_iou(box1: List[int], box2: List[int]) -> float:
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

# ----------------------------
# 主评估函数
# ----------------------------

def evaluate_predictions(pred_json_path: str, gt_dir: str, iou_threshold: float = 0.3, max_pred_boxes: int = 3):
    """
    评估预测结果：最多选取 max_pred_boxes 个最优匹配的预测框参与计算
    :param pred_json_path: 预测结果JSON路径
    :param gt_dir: 真实框文件夹路径
    :param iou_threshold: IoU阈值
    :param max_pred_boxes: 参与计算的最大预测框数（默认3）
    :return: 评估指标字典
    """
    with open(pred_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_tp = 0
    total_fp = 0
    total_gt = 0  # 用于 recall 分母
    total_pred_all = 0  # 所有有效预测框总数
    total_pred_considered = 0  # 参与计算的预测框数（最多max_pred_boxes个）
    total_excess_pred = 0  # 多余未参与计算的预测框数

    predictions = data.get("image_results", [])

    for item in predictions:
        if not item.get("inference_success", False):
            continue

        filename = item["image_filename"]
        response = item["response"]

        try:
            img_id = extract_id_from_pred_filename(filename)
        except ValueError as e:
            print(e)
            continue

        # 1. 解析并缩放预测框
        pred_boxes = parse_bboxes_from_response(response)
        scaled_pred_boxes = []
        for box in pred_boxes:
            scaled_box = [int(coord * 5.12) for coord in box]
            if scaled_box[2] > scaled_box[0] and scaled_box[3] > scaled_box[1]:
                scaled_pred_boxes.append(scaled_box)
            else:
                print(f"⚠️ 缩放后 bbox 非法: 原 {box} → {scaled_box}")
        pred_boxes = scaled_pred_boxes
        current_pred_count = len(pred_boxes)
        total_pred_all += current_pred_count

        # 2. 加载真实框
        gt_boxes = load_ground_truth(gt_dir, img_id)
        total_gt += len(gt_boxes)

        # 3. 无有效预测框：所有GT为FN
        if not pred_boxes:
            print(f"📌 {filename}: 无有效预测框，{len(gt_boxes)} 个真实框均为FN")
            continue

        # 4. 筛选最优的max_pred_boxes个预测框（核心步骤）
        # 4.1 计算每个预测框与所有真实框的最大IoU（代表该预测框的匹配质量）
        pred_max_iou = []
        for pred_idx, pred_box in enumerate(pred_boxes):
            max_iou = max([compute_iou(pred_box, gt_box) for gt_box in gt_boxes], default=0.0)
            pred_max_iou.append((pred_idx, max_iou))  # (预测框索引, 最大IoU)

        # 4.2 按最大IoU降序排序，取前max_pred_boxes个（最优匹配）
        pred_max_iou_sorted = sorted(pred_max_iou, key=lambda x: x[1], reverse=True)
        top_pred_indices = [idx for idx, _ in pred_max_iou_sorted[:max_pred_boxes]]
        top_pred_boxes = [pred_boxes[idx] for idx in top_pred_indices]  # 最优的3个预测框
        excess_count = current_pred_count - len(top_pred_indices)  # 多余未参与计算的预测框数
        total_excess_pred += excess_count
        total_pred_considered += len(top_pred_boxes)

        # 5. 基于最优的3个预测框进行匹配计算（TP/FP）
        matched_gt = [False] * len(gt_boxes)
        tp = 0
        fp = 0

        for pred_box in top_pred_boxes:
            best_iou = -1
            best_gt_idx = -1
            # 找当前预测框的最佳未匹配真实框
            for gt_idx, gt_box in enumerate(gt_boxes):
                if not matched_gt[gt_idx]:
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            # 超过阈值则记为TP，否则为FP
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp += 1
                matched_gt[best_gt_idx] = True
            else:
                fp += 1

        # 6. 累计统计
        total_tp += tp
        total_fp += fp

        # 打印当前图片详细信息
        print(f"📌 {filename}: "
              f"真实框{len(gt_boxes)}个 | "
              f"预测框{current_pred_count}个 | "
              f"最优参与计算{len(top_pred_boxes)}个 | "
              f"多余未参与{excess_count}个 | "
              f"TP{tp}个 | FP{fp}个")

    # 7. 计算最终指标
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 输出汇总结果
    print("\n" + "=" * 60)
    print(f"📊 评估结果 (IoU ≥ {iou_threshold} | 最多{max_pred_boxes}个最优预测框参与计算)")
    print("=" * 60)
    print(f"总真实框数: {total_gt}")
    print(f"总预测框数: {total_pred_all}")
    print(f"参与计算的最优预测框数: {total_pred_considered}")
    print(f"多余未参与计算的预测框数: {total_excess_pred}")
    print("-" * 30)
    print(f"True Positives (TP): {total_tp}")
    print(f"False Positives (FP): {total_fp}")
    print(f"False Negatives (FN): {total_gt - total_tp}")
    print("-" * 30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("=" * 60)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_gt - total_tp,
        "total_pred_all": total_pred_all,
        "total_pred_considered": total_pred_considered,
        "total_excess_pred": total_excess_pred,
        "total_gt": total_gt,
        "iou_threshold": iou_threshold,
        "max_pred_boxes": max_pred_boxes
    }

# ----------------------------
# 使用示例
# ----------------------------

if __name__ == "__main__":
    PRED_FILE = r"D:\Remote Sensing\batch_inference_result.json"  # 替换为你的预测 JSON 文件路径
    GT_DIR = "gt_truth\工业用地"  # ground truth 文件夹路径
    IOU_THRESHOLD = 0.3  # IoU阈值
    MAX_PRED_BOXES = 4  # 最多选取3个最优预测框参与计算

    results = evaluate_predictions(
        pred_json_path=PRED_FILE,
        gt_dir=GT_DIR,
        iou_threshold=IOU_THRESHOLD,
        max_pred_boxes=MAX_PRED_BOXES
    )