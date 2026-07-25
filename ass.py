import json
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from shapely.geometry import Polygon
from rasterstats import zonal_stats
from scipy.ndimage import uniform_filter

# ========= 配置参数 =========
# 数据路径（请根据实际情况修改）
JSON_PATH = r"D:\Remote Sensing\Shenzhen\mean_results\mean_results_withbbox\lst_ndvi_mean_with_bbox_results.json"
LST_PATH = r"D:\Remote Sensing\LST_23.tif"
NDVI_PATH = r"D:\Remote Sensing\NDVI_23.tif"

# 分析参数（可根据需求调整）
BUFFER_DISTANCE = 500  # 邻域环带缓冲距离（米）
GREEN_THRESHOLD = 0.3  # 绿地NDVI阈值
INDUS_THRESHOLD = 0.2  # 工业候选区NDVI阈值（可调整）
WATER_THRESHOLD = 0.05  # 水体排除NDVI阈值
MIN_VALID_PIXELS = 5  # 最小有效像元数
INDUS_FRAC_THRESHOLD = 0.2  # 工业占比最低阈值
LST_FILL_METHOD = "median"  # 缺失值填充方法："median"或"mean"
MIN_BBOX_AREA = 50  # 最小BBox面积要求
DIAGNOSTIC_SAMPLES = 10  # 诊断样本数量

# 输出路径
OUTPUT_DIR = r"D:\Remote Sensing\Shenzhen\assessment_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_GEOJSON = os.path.join(OUTPUT_DIR, "industrial_heat_green_assessment.geojson")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "industrial_heat_green_assessment.csv")
DIAGNOSTIC_DIR = os.path.join(OUTPUT_DIR, "diagnostics")
os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
INDUS_DIAG_DIR = os.path.join(DIAGNOSTIC_DIR, "industrial_frac")
os.makedirs(INDUS_DIAG_DIR, exist_ok=True)

# 全局变量：存储转换后的LST数据（摄氏度）
converted_lst = None
raw_ndvi = None

# ========= 1. 数据加载与预处理 =========
def load_and_preprocess_data():
    """加载并预处理JSON、LST和NDVI数据，确保LST从开尔文转换为摄氏度"""
    print("="*60)
    print("1. 开始数据加载与预处理...")
    
    # 1.1 加载JSON数据
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        bbox_data = json.load(f)
    print(f"成功加载JSON数据，包含 {len(bbox_data)} 个工业用地patch")
    
    # 1.2 转换JSON数据为GeoDataFrame
    gdf = create_geodataframe(bbox_data)
    print(f"转换为GeoDataFrame，保留 {len(gdf)} 个有效边界框")
    
    # 1.3 加载LST和NDVI数据
    lst = rxr.open_rasterio(LST_PATH).squeeze()
    ndvi = rxr.open_rasterio(NDVI_PATH).squeeze()
    print(f"加载LST数据：形状{lst.shape}，投影{lst.rio.crs}")
    print(f"加载NDVI数据：形状{ndvi.shape}，投影{ndvi.rio.crs}")
    
    # 1.4 对齐NDVI到LST的投影和分辨率
    ndvi = ndvi.rio.reproject_match(
        lst,
        resampling=Resampling.bilinear,
        nodata=np.nan
    )
    print("完成NDVI与LST的投影和分辨率对齐")
    
    # 1.5 关键步骤：确保LST从开尔文转换为摄氏度
    global converted_lst
    lst_mean = lst.mean().item()
    print(f"LST数据均值（转换前）: {lst_mean:.2f}")
    
    if lst_mean > 273:  # 开尔文转摄氏度（绝对零度约为273.15K）
        print("检测到LST为开尔文单位，执行转换为摄氏度")
        converted_lst = lst - 273.15
        print(f"LST数据均值（转换后）: {converted_lst.mean().item():.2f}℃")
    else:
        print("LST数据已为摄氏度单位，无需转换")
        converted_lst = lst.copy()
        
    # 1.6 数据清洗（处理异常值并填充NaN）
    # 过滤不合理的温度值（正常地表温度一般在20-45℃之间）
    converted_lst = clean_raster(converted_lst, min_val=10, max_val=50)  # 放宽范围便于检测异常
    ndvi = clean_raster(ndvi, min_val=-1, max_val=1, fill_nan=True)  # NDVI范围，填充缺失值
    
    # 保存原始NDVI用于诊断
    global raw_ndvi
    raw_ndvi = ndvi.copy()
    
    print("数据加载与预处理完成")
    print("="*60 + "\n")
    return gdf, converted_lst, ndvi

def create_geodataframe(bbox_data):
    """将JSON数据转换为带几何信息的GeoDataFrame，过滤过小的BBox"""
    features = []
    small_bbox_count = 0
    
    for item in bbox_data:
        # 提取坐标信息
        coords = item.get("bbox_coordinates", {})
        top_left = coords.get("top_left", {})
        bottom_right = coords.get("bottom_right", {})
        
        # 验证坐标完整性
        if not all([top_left.get("x"), top_left.get("y"), 
                   bottom_right.get("x"), bottom_right.get("y")]):
            print(f"警告: 跳过坐标不完整的patch {item.get('original_patch')}")
            continue
        
        # 创建多边形几何
        polygon = Polygon([
            (top_left["x"], top_left["y"]),
            (bottom_right["x"], top_left["y"]),
            (bottom_right["x"], bottom_right["y"]),
            (top_left["x"], bottom_right["y"]),
            (top_left["x"], top_left["y"])  # 闭合多边形
        ])
        
        # 过滤过小的BBox
        if polygon.area < MIN_BBOX_AREA:
            small_bbox_count += 1
            continue
        
        # 提取属性信息
        feature = {
            "original_patch": item.get("original_patch"),
            "bbox_index": item.get("bbox_index"),
            "lst_mean_box": item.get("lst_mean"),
            "ndvi_mean_box": item.get("ndvi_mean"),
            "geometry": polygon,
            "bbox_area": polygon.area  # 记录BBox面积
        }
        features.append(feature)
    
    if not features:
        raise ValueError("没有有效的边界框数据可处理")
    
    print(f"过滤了 {small_bbox_count} 个面积小于 {MIN_BBOX_AREA} 平方米的过小BBox")
    # 创建GeoDataFrame并设置坐标参考系
    gdf = gpd.GeoDataFrame(features)
    return gdf

def clean_raster(raster, min_val, max_val, fill_nan=False):
    """清洗栅格数据，移除异常值并可选填充NaN"""
    data = raster.data.copy()
    
    # 标记异常值为NaN
    out_of_range_mask = (data < min_val) | (data > max_val)
    out_of_range_count = np.sum(out_of_range_mask)
    if out_of_range_count > 0:
        print(f"移除 {out_of_range_count} 个超出范围的异常值（有效范围: {min_val}-{max_val}）")
        data[out_of_range_mask] = np.nan
    
    # 填充NaN值（如果需要）
    if fill_nan and np.isnan(data).any():
        mask = np.isnan(data)
        # 使用3x3窗口的均值填充缺失值
        data[mask] = uniform_filter(data, size=3, mode='nearest')[mask]
        print(f"填充了 {np.sum(mask)} 个缺失值")
    
    raster.data = data
    return raster

# ========= 2. 计算BBox内基础指标 =========
def calculate_bbox_metrics(gdf, lst, ndvi):
    """计算边界框内的基础指标，精确计算绿地覆盖率"""
    print("="*60)
    print("2. 开始计算BBox内基础指标...")
    
    # 确保GeoDataFrame与栅格坐标一致
    gdf = gdf.set_crs(lst.rio.crs)
    
    # 2.1 保存转换后的LST和原始NDVI栅格用于精确计算
    lst_path = os.path.join(OUTPUT_DIR, "tmp_lst_celsius.tif")
    lst.rio.to_raster(lst_path, compress="LZW")
    
    ndvi_path = os.path.join(OUTPUT_DIR, "tmp_ndvi_raw.tif")
    ndvi.rio.to_raster(ndvi_path, compress="LZW")
    
    # 2.2 为每个BBox计算精确的LST均值（确保使用摄氏度数据）
    lst_means = []
    for idx, geom in enumerate(gdf.geometry):
        stats = zonal_stats([geom], lst_path, stats=["mean"])[0]
        lst_means.append(stats["mean"])
        
        # 打印进度（每10个BBox）
        if (idx + 1) % 10 == 0:
            print(f"已处理 {idx + 1}/{len(gdf)} 个BBox，示例LST均值: {lst_means[-1]:.2f}℃")
    
    # 更新gdf中的LST均值为转换后的摄氏度值
    gdf["lst_mean_box"] = lst_means
    
    # 2.3 计算绿地覆盖率
    green_covers = []
    total_pixels_list = []
    green_pixels_list = []
    
    for idx, geom in enumerate(gdf.geometry):
        stats = zonal_stats(
            [geom], 
            ndvi_path, 
            stats=["count"],
            add_stats={"green_pct": lambda x: np.mean(x > GREEN_THRESHOLD) * 100,
                      "green_count": lambda x: np.sum(x > GREEN_THRESHOLD),
                      "total_valid": lambda x: len(x)}
        )[0]
        
        green_pct = stats["green_pct"] if stats["total_valid"] > 0 else np.nan
        green_covers.append(green_pct)
        total_pixels_list.append(stats["total_valid"])
        green_pixels_list.append(stats["green_count"])
    
    # 保存计算结果
    gdf["Green_cover_pct"] = green_covers
    gdf["total_valid_pixels"] = total_pixels_list
    gdf["green_pixels"] = green_pixels_list
    
    print("BBox内基础指标计算完成")
    print(f"LST均值范围: {gdf['lst_mean_box'].min():.2f}℃ - {gdf['lst_mean_box'].max():.2f}℃")
    print(f"绿地覆盖率范围: {gdf['Green_cover_pct'].min():.1f}% - {gdf['Green_cover_pct'].max():.1f}%")
    print("="*60 + "\n")
    return gdf, ndvi_path, lst_path

# ========= 3. 计算工业表面指标 =========
def calculate_industrial_metrics(gdf, lst, ndvi, ndvi_path, lst_path):
    """计算工业表面相关指标，精确计算工业占比"""
    print("="*60)
    print("3. 开始计算工业表面指标...")
    
    # 3.1 保存原始NDVI用于工业占比精确计算
    print(f"使用工业阈值范围: NDVI > {WATER_THRESHOLD} 且 NDVI < {INDUS_THRESHOLD}")
    
    # 3.2 为每个BBox单独计算工业占比
    indus_fracs = []
    indus_pixels_list = []
    
    for idx, geom in enumerate(gdf.geometry):
        # 对每个BBox单独计算工业像元占比
        stats = zonal_stats(
            [geom],
            ndvi_path,
            add_stats={
                "indus_frac": lambda x: np.mean((x > WATER_THRESHOLD) & (x < INDUS_THRESHOLD)),
                "indus_count": lambda x: np.sum((x > WATER_THRESHOLD) & (x < INDUS_THRESHOLD)),
                "total_valid": lambda x: len(x)
            }
        )[0]
        
        # 处理有效像元数为0的情况
        if stats["total_valid"] == 0:
            indus_frac = np.nan
            indus_count = 0
        else:
            indus_frac = stats["indus_frac"]
            indus_count = stats["indus_count"]
            
        indus_fracs.append(indus_frac)
        indus_pixels_list.append(indus_count)
        
        # 打印进度（每10个BBox）
        if (idx + 1) % 10 == 0:
            print(f"已处理 {idx + 1}/{len(gdf)} 个BBox，示例工业占比: {indus_frac:.3f}")
    
    # 保存计算结果
    gdf["Indus_frac"] = indus_fracs
    gdf["indus_pixels"] = indus_pixels_list
    
    # 填充工业占比的缺失值为0
    gdf["Indus_frac"] = gdf["Indus_frac"].fillna(0)
    
    # 3.3 诊断工业占比计算结果
    diagnose_industrial_frac(gdf, raw_ndvi)
    
    # 3.4 计算工业区域LST均值（使用已转换的摄氏度数据）
    # 创建工业区域LST掩膜
    indus_mask = ((ndvi.data > WATER_THRESHOLD) & 
                 (ndvi.data < INDUS_THRESHOLD)).astype(np.uint8)
    indus_mask = np.nan_to_num(indus_mask, nan=0)
    
    lst_data = lst.data.copy()  # 这里的lst已经是转换后的摄氏度数据
    lst_data[indus_mask == 0] = np.nan  # 非工业区域设为NaN
    indus_lst = lst.copy(data=lst_data)
    indus_lst.rio.write_nodata(np.nan, inplace=True)
    indus_lst_path = os.path.join(OUTPUT_DIR, "tmp_industrial_lst_celsius.tif")
    indus_lst.rio.to_raster(indus_lst_path, compress="LZW")
    
    # 计算工业区域LST均值
    gdf["LST_indus"] = [
        s["mean"] for s in zonal_stats(gdf.geometry, indus_lst_path, stats=["mean"])
    ]
    
    # 处理工业占比过低的情况，回退为BBox内平均LST
    low_indus_mask = (gdf["Indus_frac"] < INDUS_FRAC_THRESHOLD)
    gdf.loc[low_indus_mask, "LST_indus"] = gdf.loc[low_indus_mask, "lst_mean_box"]
    
    # 填充LST_indus的缺失值
    lst_fill_val = gdf["LST_indus"].median() if LST_FILL_METHOD == "median" else gdf["LST_indus"].mean()
    gdf["LST_indus"] = gdf["LST_indus"].fillna(lst_fill_val)
    
    print("工业表面指标计算完成")
    print(f"工业区域LST均值范围: {gdf['LST_indus'].min():.2f}℃ - {gdf['LST_indus'].max():.2f}℃")
    print(f"工业占比范围: {gdf['Indus_frac'].min():.3f} - {gdf['Indus_frac'].max():.3f}")
    print(f"填充LST_indus缺失值使用: {LST_FILL_METHOD} = {lst_fill_val:.2f}℃")
    print("="*60 + "\n")
    return gdf, indus_lst_path

def diagnose_industrial_frac(gdf, ndvi):
    """诊断工业占比计算结果"""
    print(f"正在分析 {DIAGNOSTIC_SAMPLES} 个随机BBox的工业占比分布...")
    
    # 确保有足够的样本
    sample_size = min(DIAGNOSTIC_SAMPLES, len(gdf))
    if sample_size == 0:
        return
        
    # 创建诊断数据记录
    diagnostic_data = []
        
    # 随机选择样本，包括极端值样本
    random_indices = np.random.choice(len(gdf), max(1, sample_size//2), replace=False)
    extreme_indices = []
    
    # 添加Indus_frac接近1的样本
    high_indus = gdf[gdf["Indus_frac"] > 0.8].index
    if len(high_indus) > 0:
        extreme_indices.append(np.random.choice(high_indus))
    
    # 添加Indus_frac接近0的样本
    low_indus = gdf[gdf["Indus_frac"] < 0.2].index
    if len(low_indus) > 0:
        extreme_indices.append(np.random.choice(low_indus))
    
    # 合并样本索引
    sample_indices = np.unique(np.concatenate([random_indices, extreme_indices]))[:sample_size]
    
    for idx in sample_indices:
        # 获取BBox信息
        geom = gdf.geometry.iloc[idx]
        patch_id = gdf["original_patch"].iloc[idx]
        bbox_idx = gdf["bbox_index"].iloc[idx]
        indus_frac = gdf["Indus_frac"].iloc[idx]
        total_pixels = gdf["total_valid_pixels"].iloc[idx]
        indus_pixels = gdf["indus_pixels"].iloc[idx]
        
        try:
            # 裁剪该BBox范围内的NDVI数据
            ndvi_crop = ndvi.rio.clip([geom])
            ndvi_values = ndvi_crop.data[~np.isnan(ndvi_crop.data)]
            
            # 记录诊断数据
            diagnostic_data.append({
                "patch_id": patch_id,
                "bbox_index": bbox_idx,
                "indus_frac": indus_frac,
                "total_valid_pixels": total_pixels,
                "indus_pixels": indus_pixels,
                "ndvi_mean": np.mean(ndvi_values) if len(ndvi_values) > 0 else np.nan,
                "ndvi_min": np.min(ndvi_values) if len(ndvi_values) > 0 else np.nan,
                "ndvi_max": np.max(ndvi_values) if len(ndvi_values) > 0 else np.nan
            })
            
            # 绘制直方图并标记工业阈值范围
            plt.figure(figsize=(10, 4))
            plt.hist(ndvi_values, bins=20, color='gray', alpha=0.7)
            plt.axvline(x=WATER_THRESHOLD, color='blue', linestyle='--', 
                       label=f'水体阈值 ({WATER_THRESHOLD})')
            plt.axvline(x=INDUS_THRESHOLD, color='red', linestyle='--', 
                       label=f'工业上限阈值 ({INDUS_THRESHOLD})')
            plt.axvspan(WATER_THRESHOLD, INDUS_THRESHOLD, color='yellow', alpha=0.3,
                       label='工业区域范围')
            plt.title(f'BBox {patch_id} (index {bbox_idx})\n工业占比: {indus_frac:.3f}')
            plt.xlabel('NDVI值')
            plt.ylabel('像元数')
            plt.xlim(-1, 1)  # NDVI理论范围
            plt.legend()
            
            # 保存图表
            plot_path = os.path.join(INDUS_DIAG_DIR, f'indus_frac_{patch_id}_{bbox_idx}.png')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"分析BBox {patch_id} 时出错: {str(e)}")
    
    # 保存诊断数据到CSV
    diag_df = pd.DataFrame(diagnostic_data)
    diag_df.to_csv(os.path.join(INDUS_DIAG_DIR, "industrial_frac_diagnostics.csv"), index=False)
    print(f"工业占比诊断数据已保存到 {os.path.join(INDUS_DIAG_DIR, 'industrial_frac_diagnostics.csv')}")

# ========= 4. 计算邻域基线指标 =========
def calculate_neighborhood_metrics(gdf, lst, ndvi):
    """计算邻域环带基线指标（使用摄氏度LST数据）"""
    print("="*60)
    print("4. 开始计算邻域基线指标...")
    
    # 保存转换后的LST和NDVI用于环带计算
    lst_path = os.path.join(OUTPUT_DIR, "tmp_lst_celsius.tif")
    ndvi_path = os.path.join(OUTPUT_DIR, "tmp_ndvi.tif")
    lst.rio.to_raster(lst_path, compress="LZW")
    ndvi.rio.to_raster(ndvi_path, compress="LZW")
    
    # 计算每个BBox的环带均值
    gdf["LST_ring_mean"] = gdf.geometry.apply(
        lambda geom: calculate_ring_mean(geom, lst_path, BUFFER_DISTANCE)
    )
    gdf["NDVI_ring_mean"] = gdf.geometry.apply(
        lambda geom: calculate_ring_mean(geom, ndvi_path, BUFFER_DISTANCE)
    )
    
    # 填充环带均值的缺失值
    lst_ring_fill = gdf["LST_ring_mean"].median() if LST_FILL_METHOD == "median" else gdf["LST_ring_mean"].mean()
    gdf["LST_ring_mean"] = gdf["LST_ring_mean"].fillna(lst_ring_fill)
    
    ndvi_ring_fill = gdf["NDVI_ring_mean"].median()
    gdf["NDVI_ring_mean"] = gdf["NDVI_ring_mean"].fillna(ndvi_ring_fill)
    
    print(f"邻域环带LST均值范围: {gdf['LST_ring_mean'].min():.2f}℃ - {gdf['LST_ring_mean'].max():.2f}℃")
    print(f"填充LST_ring_mean缺失值使用: {LST_FILL_METHOD} = {lst_ring_fill:.2f}℃")
    print("邻域基线指标计算完成")
    print("="*60 + "\n")
    return gdf

def calculate_ring_mean(geom, raster_path, buffer_distance):
    """计算边界框外缓冲环带的均值"""
    try:
        # 创建外缓冲和环带
        outer_buffer = geom.buffer(buffer_distance)
        ring = outer_buffer.difference(geom)
        
        # 如果环带为空，返回NaN
        if ring.is_empty:
            return np.nan
        
        # 计算环带内均值
        stats = zonal_stats([ring], raster_path, stats=["mean"])
        return stats[0]["mean"] if stats[0]["mean"] is not None else np.nan
    except Exception as e:
        print(f"计算环带均值时出错: {str(e)}")
        return np.nan

# ========= 5. 计算相对异常与排序 =========
def calculate_anomalies_and_ranking(gdf):
    """计算相对异常值和优先级排序"""
    print("="*60)
    print("5. 开始计算相对异常与排序...")
    
    # 5.1 计算相对异常和绿量缺口
    gdf["LST_excess_local"] = gdf["LST_indus"] - gdf["LST_ring_mean"]
    gdf["NDVI_gap_local"] = (gdf["NDVI_ring_mean"] - gdf["ndvi_mean_box"]).clip(lower=0)
    
    # 检查LST异常值范围（应在合理范围内）
    print(f"LST相对异常值范围: {gdf['LST_excess_local'].min():.2f}℃ - {gdf['LST_excess_local'].max():.2f}℃")
    
    # 5.2 计算置信度权重
    gdf["w_conf"] = gdf["Indus_frac"].apply(
        lambda x: min(1.0, x / 0.6) if pd.notna(x) else 0.0
    )
    
    # 5.3 计算排序分数（处理可能的NaN值）
    lst_excess_clean = gdf["LST_excess_local"].fillna(gdf["LST_excess_local"].median())
    gdf["rank_LST"] = lst_excess_clean.rank(method="average", pct=True, ascending=True)
    
    gdf["rank_NDVI"] = gdf["NDVI_gap_local"].rank(method="average", pct=True, ascending=True)
    
    # 计算最终优先级
    gdf["Priority"] = gdf["w_conf"] * (0.7 * gdf["rank_LST"] + 0.3 * gdf["rank_NDVI"])
    
    # 5.4 双变量分级（3×3）
    lst_excess_for_class = gdf["LST_excess_local"].fillna(gdf["LST_excess_local"].median())
    lst_quantiles = lst_excess_for_class.quantile([0.33, 0.67]).values
    ndvi_quantiles = gdf["NDVI_gap_local"].quantile([0.33, 0.67]).values
    
    gdf["LST_class"] = lst_excess_for_class.apply(
        lambda x: classify_value(x, lst_quantiles)
    )
    gdf["NDVI_class"] = gdf["NDVI_gap_local"].apply(
        lambda x: classify_value(x, ndvi_quantiles)
    )
    gdf["BiVar_class"] = gdf["LST_class"] + "-" + gdf["NDVI_class"]
    
    # 5.5 质量控制标记
    gdf["flag_low_quality"] = (gdf["total_valid_pixels"] < MIN_VALID_PIXELS) | \
                             (gdf["Indus_frac"] < INDUS_FRAC_THRESHOLD)
    
    # 统计w_conf分布
    print(f"w_conf值范围: {gdf['w_conf'].min():.3f} - {gdf['w_conf'].max():.3f}")
    print(f"w_conf中间值: {gdf['w_conf'].median():.3f}")
    print("相对异常与排序计算完成")
    print("="*60 + "\n")
    return gdf

def classify_value(value, quantiles):
    """将值按分位数分为低、中、高三类"""
    if pd.isna(value):
        return "NA"
    if value <= quantiles[0]:
        return "Low"
    elif value <= quantiles[1]:
        return "Mid"
    else:
        return "High"

# ========= 6. 结果导出与总结 =========
def export_results(gdf):
    """导出分析结果并生成总结"""
    print("="*60)
    print("6. 导出分析结果...")
    
    # 保存结果
    gdf.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    gdf.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)
    
    # 生成LST指标统计摘要（验证单位是否正确）
    lst_columns = ["lst_mean_box", "LST_indus", "LST_ring_mean", "LST_excess_local"]
    lst_stats = gdf[lst_columns].describe()
    print("\nLST指标统计摘要 (单位: ℃):")
    print(lst_stats.round(2).to_string())
    
    # 生成工业占比统计摘要
    indus_stats = gdf["Indus_frac"].describe()
    print("\n工业占比(Indus_frac)统计摘要:")
    print(indus_stats.round(3).to_string())
    
    # 生成工业占比区间分布
    indus_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    indus_labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    gdf["indus_range"] = pd.cut(
        gdf["Indus_frac"], 
        bins=indus_bins, 
        labels=indus_labels,
        include_lowest=True
    )
    print("\n工业占比区间分布:")
    print(gdf["indus_range"].value_counts().sort_index().to_string())
    
    print(f"\n结果已导出至:")
    print(f"GeoJSON: {OUTPUT_GEOJSON}")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"工业占比诊断图表: {INDUS_DIAG_DIR}")
    print("="*60)

# ========= 主函数 =========
def main():
    """主函数：执行完整的热绿评估流程，确保所有LST指标为摄氏度"""
    # 1. 数据加载与预处理（包含LST单位转换）
    gdf, lst, ndvi = load_and_preprocess_data()
    
    # 2. 计算BBox内基础指标（使用转换后的LST数据）
    gdf, ndvi_path, lst_path = calculate_bbox_metrics(gdf, lst, ndvi)
    
    # 3. 计算工业表面指标（确保LST_indus为摄氏度）
    gdf, indus_lst_path = calculate_industrial_metrics(gdf, lst, ndvi, ndvi_path, lst_path)
    
    # 4. 计算邻域基线指标（LST_ring_mean为摄氏度）
    gdf = calculate_neighborhood_metrics(gdf, lst, ndvi)
    
    # 5. 计算相对异常与排序（LST_excess_local为摄氏度差值）
    gdf = calculate_anomalies_and_ranking(gdf)
    
    # 6. 导出结果与总结（验证所有LST指标单位）
    export_results(gdf)
    
    # 清理临时文件
    temp_files = [ndvi_path, lst_path, indus_lst_path,
                 os.path.join(OUTPUT_DIR, "tmp_lst_celsius.tif"),
                 os.path.join(OUTPUT_DIR, "tmp_ndvi.tif")]
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"已删除临时文件: {f}")

if __name__ == "__main__":
    main()
    