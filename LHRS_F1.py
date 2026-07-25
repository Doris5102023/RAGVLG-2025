import csv
import json
import os
import numpy as np

def compute_iou(box1, box2):
    """计算两个 bbox 的 IoU"""
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    inter_x1 = max(x1, x1_gt)
    inter_y1 = max(y1, y1_gt)
    inter_x2 = min(x2, x2_gt)
    inter_y2 = min(y2, y2_gt)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def load_predictions(csv_file):
    """加载预测结果"""
    predictions = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row['image_name']
            bbox_str = row['bbox']
            try:
                # 使用 ast.literal_eval 更安全（可选）
                bbox = eval(bbox_str)  # 或者用 ast.literal_eval
                if bbox == [0, 0, 0, 0]:
                    bbox = None
            except Exception as e:
                print(f"Error parsing bbox {bbox_str}: {e}")
                bbox = None
            predictions[img_name] = bbox
    return predictions


def find_ground_truth_file(img_name):
    """根据 image_name 推断对应的 JSON 文件名"""
    # 示例：unchange_63.png -> result_63.json
    if not img_name.startswith('crsim_') and not img_name.startswith('unchange_'):
        raise ValueError(f"Unexpected image name format: {img_name}")

    # 提取数字部分，如 63
    import re
    match = re.search(r'(\d+)', img_name)
    if not match:
        raise ValueError(f"No number found in {img_name}")
    id_num = match.group(1)
    json_filename = f"result_{id_num}.json"
    return json_filename


def load_ground_truths(predictions, json_dir):
    """加载所有对应的 ground truth"""
    gts = {}
    for img_name in predictions.keys():
        json_file = find_ground_truth_file(img_name)
        json_path = os.path.join(json_dir, json_file)

        if not os.path.exists(json_path):
            print(f"Warning: Ground truth file not found: {json_path}")
            gts[img_name] = []
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        # 确保 imageid 匹配
        if "image_name" in data:
            gt_img_name = data["image_name"]
        elif "imageid" in data:
            gt_img_name = data["imageid"]
        else:
            gt_img_name = None

        if gt_img_name and gt_img_name != img_name:
            print(f"Warning: imageid mismatch in {json_file}: expected {img_name}, got {gt_img_name}")

        # 提取 bboxes
        bboxes_list = []
        if "bboxes" in data:
            for bbox_dict in data["bboxes"]:
                if "bbox" in bbox_dict and isinstance(bbox_dict["bbox"], list):
                    bboxes_list.append(bbox_dict["bbox"])
        elif "bbox" in data and isinstance(data["bbox"], list):
            bboxes_list.append(data["bbox"])

        gts[img_name] = bboxes_list
    return gts


def evaluate(predictions, gts, iou_thresh=0.3):
    """计算 TP, FP, FN"""
    tp = 0
    fp = 0
    fn = 0

    for img_name, pred_bbox in predictions.items():
        gt_bboxes = gts[img_name]

        if pred_bbox is None:
            # 没有预测框 → 所有 GT 都漏检
            fn += len(gt_bboxes)
            continue

        # 查找最佳匹配的 GT 框（IoU 最大）
        best_iou = 0
        matched = False
        for gt_bbox in gt_bboxes:
            iou = compute_iou(pred_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou

        if best_iou >= iou_thresh:
            tp += 1
            matched = True
        else:
            fp += 1

        # 每个未被命中的 GT 都算 FN（但只有一条预测，最多命中一个）
        fn += len(gt_bboxes) - (1 if matched else 0)

    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }


# ========================
# 主程序
# ========================

if __name__ == "__main__":
    # 输入路径
    csv_file = ''
    json_dir = ''  # 放置 result_*.json 的目录

    # 加载预测
    predictions = load_predictions(csv_file)
    print(f"Loaded {len(predictions)} predictions.")

    # 加载 ground truth
    gts = load_ground_truths(predictions, json_dir)
    print(f"Loaded ground truths for {len(gts)} images.")

    # 评估
    results = evaluate(predictions, gts, iou_thresh=0.3)

    # 输出结果
    print("\nEvaluation Results:")
    print(f"TP: {results['TP']}")
    print(f"FP: {results['FP']}")
    print(f"FN: {results['FN']}")
    print(f"Precision: {results['Precision']:.4f}")
    print(f"Recall: {results['Recall']:.4f}")
    print(f"F1-score: {results['F1-score']:.4f}")
