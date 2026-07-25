import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import os

# ========= 配置参数 (Configuration Parameters) =========
INPUT_CSV = r"D:\Remote Sensing\Shenzhen\assessment_results\industrial_heat_green_assessment.csv"
INPUT_GEOJSON = r"D:\Remote Sensing\Shenzhen\assessment_results\industrial_heat_green_assessment.geojson"

OUTPUT_DIR = r"D:\Remote Sensing\Shenzhen\clustering_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "priority_based_risk_clusters.csv")
OUTPUT_GEOJSON = os.path.join(OUTPUT_DIR, "priority_based_risk_clusters.geojson")
PLOT_DIR = os.path.join(OUTPUT_DIR, "enhanced_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# 聚类设置
CLUSTER_NUM = 5
RANDOM_STATE = 42
CLUSTER_FEATURES = ["Priority"]
RISK_ORDER = [1, 2, 3, 4, 5]
# 风险等级名称映射
RISK_LEVEL_NAMES = {
    1: "extremely high risk",
    2: "high risk",
    3: "medium risk",
    4: "low risk",
    5: "extremely low risk"
}
# 为每个风险等级指定RGB颜色 (0-1范围)
RISK_LEVEL_COLORS = {
    1: (1.0, 0.5098, 0.0),    # 红色 - 极高风险
    2: (1.0, 0.6510, 0.0),    # 橙色 - 高风险
    3: (1.0, 0.7922, 0.1569),    # 黄色 - 中等风险
    4: (1.0, 0.9020, 0.3137),    # 绿色 - 低风险
    5: (1.0, 0.9216, 0.7451)     # 深绿色 - 极低风险
}
# 生成与风险等级顺序匹配的颜色列表（用于调色板）
COLOR_PALETTE = [RISK_LEVEL_COLORS[lvl] for lvl in RISK_ORDER]

# ========= 数据加载与预处理 =========
def load_and_preprocess_data():
    print("="*60)
    print("1. Loading and Preprocessing Data...")
    
    df = pd.read_csv(INPUT_CSV)
    gdf = gpd.read_file(INPUT_GEOJSON)
    
    print(f"Raw data contains {len(df)} samples, using feature: {CLUSTER_FEATURES[0]}")
    
    # 处理缺失值
    missing_values = df[CLUSTER_FEATURES].isnull().sum()
    print("\nMissing Values Summary:")
    print(missing_values[missing_values > 0].to_string())
    
    for feature in CLUSTER_FEATURES:
        if df[feature].isnull().any():
            median_val = df[feature].median()
            df[feature].fillna(median_val, inplace=True)
            print(f"Filled {feature} missing values with median: {median_val:.3f}")
    
    # 数据标准化
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[CLUSTER_FEATURES])
    scaled_df = pd.DataFrame(
        scaled_features, 
        columns=[f"{f}_scaled" for f in CLUSTER_FEATURES]
    )
    
    print("\nData Preprocessing Completed")
    print("="*60 + "\n")
    return df, gdf, scaled_df, scaler

# ========= 执行KMeans聚类 =========
def perform_kmeans_clustering(scaled_df):
    print("="*60)
    print(f"2. Performing KMeans Clustering (n_clusters={CLUSTER_NUM})...")
    
    kmeans = KMeans(
        n_clusters=CLUSTER_NUM,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=300
    )
    clusters = kmeans.fit_predict(scaled_df)
    
    inertia = kmeans.inertia_
    cluster_dist = pd.Series(clusters).value_counts().sort_index()
    
    print(f"Clustering Completed. Inertia: {inertia:.4f}")
    print("\nCluster Distribution (sorted by cluster ID):")
    for cluster_id in sorted(cluster_dist.index):
        print(f"Cluster {cluster_id}: {cluster_dist[cluster_id]} samples")
    
    print("\nKMeans Clustering Completed")
    print("="*60 + "\n")
    return clusters, kmeans

# ========= 聚类结果分析 =========
def analyze_clustering_results(df, scaled_df, clusters, kmeans):
    print("="*60)
    print("3. Analyzing Clustering Results...")
    
    result_df = df.copy()
    result_df["cluster"] = clusters
    result_df["cluster"] = result_df["cluster"].astype("category")
    
    # 计算聚类统计量
    analysis_features = ["Priority", "LST_excess_local", "NDVI_gap_local", "Green_cover_pct", "Indus_frac"]
    cluster_stats = result_df.groupby("cluster")[analysis_features].agg(["mean", "std"]).round(3)
    
    # 扁平化列名
    cluster_stats.columns = [f"{feat}_{stat}" for feat, stat in cluster_stats.columns]
    print("\nCluster Statistics (Mean & Std):")
    print(cluster_stats.to_string())
    
    # 映射聚类到风险等级
    priority_means = result_df.groupby("cluster")["Priority"].mean().sort_values(ascending=False)
    risk_mapping = {cluster: i+1 for i, cluster in enumerate(priority_means.index)}
    
    # 使用pd.Categorical创建有序分类
    result_df["risk_level"] = pd.Categorical(
        result_df["cluster"].map(risk_mapping),
        categories=RISK_ORDER,
        ordered=True
    )
    
    # 增加风险等级名称列，便于绘图
    result_df["risk_name"] = result_df["risk_level"].map(RISK_LEVEL_NAMES)
    
    # 打印风险等级映射
    print("\nRisk Level Mapping (Ordered by Priority):")
    print("Risk Level | Cluster ID | Priority Mean | Sample Count")
    print("-"*60)
    for risk_lvl in RISK_ORDER:
        cluster_id = [cid for cid, rl in risk_mapping.items() if rl == risk_lvl][0]
        mean_prio = priority_means[cluster_id]
        sample_count = len(result_df[result_df["risk_level"] == risk_lvl])
        print(f"{risk_lvl:10d} | {cluster_id:9d} | {mean_prio:13.3f} | {sample_count:12d}")
    
    cluster_stats.to_csv(os.path.join(OUTPUT_DIR, "priority_cluster_statistics.csv"))
    
    print("\nClustering Result Analysis Completed")
    print("="*60 + "\n")
    return result_df, cluster_stats, risk_mapping, priority_means

# ========= 增强可视化功能 =========
def plot_priority_vs_lst(result_df):
    """只绘制一张Priority与LST_excess_local的关系散点图"""
    plt.figure(figsize=(10, 7))
    
    # 使用hue和palette参数处理多类别颜色
    sns.scatterplot(
        data=result_df,
        x="LST_excess_local",
        y="Priority",
        hue="risk_name",  # 使用风险等级名称作为分类依据
        hue_order=[RISK_LEVEL_NAMES[lvl] for lvl in RISK_ORDER],  # 保持顺序
        palette=COLOR_PALETTE,  # 使用自定义颜色
        alpha=0.7,
        s=60
    )
    
    plt.xlabel("LST_excess_local (°C)", fontsize=12)
    plt.ylabel("Priority", fontsize=12)
    plt.legend(title="Risk Level", title_fontsize=10, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "priority_vs_lst_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Scatter Plot Saved: Priority vs LST_excess_local")

def plot_priority_histogram_by_risk(result_df):
    plt.figure(figsize=(12, 8))
    
    for idx, risk_lvl in enumerate(RISK_ORDER, 1):
        plt.subplot(2, 3, idx)
        subset = result_df[result_df["risk_level"] == risk_lvl]
        risk_name = RISK_LEVEL_NAMES[risk_lvl]
        
        sns.histplot(
            data=subset,
            x="Priority",
            bins=15,
            kde=True,
            color=RISK_LEVEL_COLORS[risk_lvl],  # 单一颜色，使用color参数
            alpha=0.6,
            edgecolor="black",
            linewidth=0.5
        )
        
        mean_prio = subset["Priority"].mean()
        plt.axvline(x=mean_prio, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_prio:.3f}")
        plt.title(f"Priority Distribution - {risk_name}", fontsize=12)
        plt.xlabel("Priority", fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
        plt.legend(fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "priority_histogram_by_risk.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Histogram Plot Saved: Priority Distribution by Risk Level")

def plot_risk_level_boxplots(result_df):
    features_to_plot = {
        "Priority": "Priority",
        "LST_excess_local": "Local LST Anomaly (°C)",
        "NDVI_gap_local": "Local NDVI Gap",
        "Green_cover_pct": "Green Cover Percentage (%)",
        "Indus_frac": "Industrial Fraction"
    }
    
    plt.figure(figsize=(18, 12))
    
    for idx, (col, label) in enumerate(features_to_plot.items(), 1):
        plt.subplot(3, 2, idx)
        sns.boxplot(
            data=result_df,
            x="risk_level",
            y=col,
            palette=COLOR_PALETTE,  # 使用自定义颜色列表
            order=RISK_ORDER,
            linewidth=1.2,
            flierprops=dict(marker="o", markerfacecolor="red", markersize=4)
        )
        
        plt.title(f"{label} by Risk Level", fontsize=12)
        plt.xticks(
            ticks=range(len(RISK_ORDER)),
            labels=[RISK_LEVEL_NAMES[lvl] for lvl in RISK_ORDER],
            fontsize=10,
            rotation=30,
            ha='right'
        )
        plt.xlabel("Risk Level", fontsize=10)
        plt.ylabel(label, fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.3, axis="y")
        
        sample_counts = result_df.groupby("risk_level").size()
        for i, risk_lvl in enumerate(RISK_ORDER):
            count = sample_counts[risk_lvl]
            plt.text(i, result_df[col].min() * 0.95, f"n={count}", 
                     ha="center", va="top", fontsize=9, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "risk_level_boxplots.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Boxplot Plot Saved: Key Indicators by Risk Level")

def plot_priority_vs_other_indicators(result_df):
    """创建两个散点图，展示Priority与NDVI缺口和绿色覆盖率的关系"""
    indicators = [
        {"column": "NDVI_gap_local", "label": "NDVI_gap_local", "filename": "priority_vs_ndvi_gap"},
        {"column": "Green_cover_pct", "label": "Green_cover_pct (%)", "filename": "priority_vs_green_cover"}
    ]
    
    for indicator in indicators:
        plt.figure(figsize=(10, 7))
        
        # 使用hue和palette参数处理多类别颜色
        sns.scatterplot(
            data=result_df,
            x=indicator["column"],
            y="Priority",
            hue="risk_name",  # 使用风险等级名称作为分类依据
            hue_order=[RISK_LEVEL_NAMES[lvl] for lvl in RISK_ORDER],  # 保持顺序
            palette=COLOR_PALETTE,  # 使用自定义颜色
            alpha=0.7,
            s=60
        )
        
        plt.xlabel(indicator["label"], fontsize=12)
        plt.ylabel("Priority", fontsize=12)
        plt.legend(title="Risk Level", title_fontsize=10, fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(PLOT_DIR, f"{indicator['filename']}_scatter.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Scatter Plot Saved: Priority vs {indicator['label']}")

def plot_risk_level_distribution(result_df):
    """饼图展示风险等级分布，使用指定的RGB颜色"""
    plt.figure(figsize=(12, 10))
    
    # 计算各风险等级的数量
    risk_dist = result_df["risk_level"].value_counts().reindex(RISK_ORDER)
    # 获取对应的风险等级名称和颜色
    risk_labels = [RISK_LEVEL_NAMES[lvl] for lvl in RISK_ORDER]
    
    # 绘制饼图
    wedges, texts, autotexts = plt.pie(
        risk_dist.values,
        labels=risk_labels,
        autopct='%1.1f%%',
        textprops={'fontsize': 20},
        colors=COLOR_PALETTE,  # 使用自定义颜色列表
        startangle=140,
        pctdistance=0.85
    )
    
    # 设置百分比文本的样式
    for text in autotexts:
        text.set_fontsize(20)
        text.set_fontweight('bold')
    
    # 为每个扇形添加样本数量标签
    for i, autotext in enumerate(autotexts):
        count = risk_dist.values[i]
        autotext.set_text(f"{autotext.get_text()}\nn={count}")
    
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "risk_level_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Pie Chart Saved: Risk Level Distribution")

def plot_risk_level_bar_chart(result_df):
    """条形图展示风险等级分布，使用指定的RGB颜色"""
    plt.figure(figsize=(12, 8))
    
    # 计算各风险等级的数量
    risk_dist = result_df["risk_level"].value_counts().reindex(RISK_ORDER)
    # 获取对应的风险等级名称
    risk_labels = [RISK_LEVEL_NAMES[lvl] for lvl in RISK_ORDER]
    
    # 绘制条形图
    bars = plt.bar(
        risk_labels,
        risk_dist.values,
        color=COLOR_PALETTE,  # 使用自定义颜色列表
        edgecolor='black'
    )
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height}', ha='center', va='bottom', fontsize=12)
    
    plt.xlabel('Risk Level', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.title('Distribution of Risk Levels', fontsize=16)
    plt.xticks(rotation=30, ha='right', fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3, axis="y")
    plt.tight_layout()
    
    plt.savefig(os.path.join(PLOT_DIR, "risk_level_bar_chart.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Bar Chart Saved: Risk Level Distribution")

# ========= 执行可视化 =========
def visualize_clustering_results(result_df, cluster_stats, risk_mapping, priority_means):
    print("="*60)
    print("4. Generating Enhanced Visualizations...")
    
    plot_priority_vs_lst(result_df)
    plot_priority_histogram_by_risk(result_df)
    plot_risk_level_boxplots(result_df)
    plot_priority_vs_other_indicators(result_df)
    plot_risk_level_distribution(result_df)
    plot_risk_level_bar_chart(result_df)  # 添加条形图
    
    print(f"\nAll Enhanced Visualizations Saved to: {PLOT_DIR}")
    print("="*60 + "\n")

# ========= 保存聚类结果 =========
def save_clustering_results(result_df, gdf, risk_mapping, priority_means):
    print("="*60)
    print("5. Saving Clustering Results...")
    
    result_df_sorted = result_df.sort_values(["risk_level", "Priority"], ascending=[True, False])
    result_df_sorted.to_csv(OUTPUT_CSV, index=False)
    
    merged_gdf = gdf.merge(
        result_df[["original_patch", "cluster", "risk_level"]],
        on="original_patch",
        how="left"
    )
    
    merged_gdf["risk_level"] = pd.Categorical(
        merged_gdf["risk_level"],
        categories=RISK_ORDER,
        ordered=True
    )
    merged_gdf.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    
    risk_mapping_df = pd.DataFrame({
        "Risk_Level": RISK_ORDER,
        "Risk_Name": [RISK_LEVEL_NAMES[lvl] for lvl in RISK_ORDER],
        "Cluster_ID": [cid for cid, rl in sorted(risk_mapping.items(), key=lambda x: x[1])],
        "Priority_Mean": [priority_means[cid] for cid, rl in sorted(risk_mapping.items(), key=lambda x: x[1])],
        "RGB_Color": [f"RGB{RISK_LEVEL_COLORS[lvl]}" for lvl in RISK_ORDER]  # 添加颜色信息
    })
    risk_mapping_df.to_csv(os.path.join(OUTPUT_DIR, "risk_level_mapping.csv"), index=False)
    
    print(f"Results Saved To:")
    print(f"- CSV: {OUTPUT_CSV}")
    print(f"- GeoJSON: {OUTPUT_GEOJSON}")
    print(f"- Risk Mapping: {os.path.join(OUTPUT_DIR, 'risk_level_mapping.csv')}")
    print("="*60 + "\n")

# ========= 主函数 =========
def main():
    df, gdf, scaled_df, scaler = load_and_preprocess_data()
    clusters, kmeans = perform_kmeans_clustering(scaled_df)
    result_df, cluster_stats, risk_mapping, priority_means = analyze_clustering_results(df, scaled_df, clusters, kmeans)
    visualize_clustering_results(result_df, cluster_stats, risk_mapping, priority_means)
    save_clustering_results(result_df, gdf, risk_mapping, priority_means)
    
    print("Enhanced Priority-Based Clustering Workflow Completed!")
    print("Key Notes:")
    print("- Risk levels are ordered by Priority (1=extremely high, 5=extremely low)")
    print("- All visualizations follow risk level order")
    print("- Results are sorted by risk level and Priority")

if __name__ == "__main__":
    main()