# RAGVLG
Overview
This work proposes RAGVLG, a framework that integrates Retrieval-Augmented Generation (RAG) and Vision-Language Models (VLMs). It targets industrial land recognition, ecological risk evaluation, risk grade clustering and spatial correlation analysis based on remote sensing imagery. Experiments are carried out on remote sensing datasets covering Shenzhen and Hong Kong.
All original remote sensing data, annotated labels and intermediate files are hosted on Google Drive.
Repository Structure
plaintext
├── Clustering_results/                # Final risk clustering results & spatial analysis outputs
│   ├── priority_based_risk_clusters.csv       # Risk clustering index and priority table
│   ├── priority_based_risk_clusters.geojson   # Spatial file for QGIS risk visualization
│   ├── priority_cluster_statistics.csv        # Statistical information of each risk cluster
│   ├── risk_level_mapping.csv                 # Mapping table of risk levels
│   └── LISA_analysis_results/                 # Local Indicators of Spatial Association (LISA) outputs
├── Cropped_patches/                    # Industrial land detection outputs from Qwen2.5-72B
│   ├── Image files                          # Cropped industrial land image patches
│   └── JSON files                           # Bounding box coordinates of detected targets
├── gt_truth/                           # Ground truth dataset
│   ├── Remote sensing images                # Original images with manual annotations
│   └── JSON files                           # True bounding box coordinates of industrial land
├── Industrial-Land/                    # Dataset & results for visual grounding task
│   ├── Remote sensing images                # Image data for MLLM visual grounding
│   └── JSON files                           # Visual grounding inference results of Qwen model
├── RSI(final version)                 # Original remote sensing image patches
│       # Filtered valid patches of Shenzhen and Hong Kong (exclude sea, mountain and edge areas)
├── RAGVLG/                            # Core experimental code folder
│   ├── change_removal.py        # Remove changed areas via change mask to build RSIL dataset
│   ├── detection.py             # MLLM visual grounding & industrial land evaluation
│   ├── patch.py                 # Slice full remote sensing images into standard patches
│   ├── geobbox.py               # Extract bbox coordinates of detected industrial land patches
│   ├── geoprocessor.py          # Batch crop LST & NDVI images using gdalwarp
│   ├── lstndvi.py               # Calculate average LST and NDVI for each bbox region
│   ├── ass.py                   # Ecological risk assessment for industrial land
│   ├── kmeans.py                # K-Means clustering to divide 5 risk levels
│   ├── lisa_analysis.py         # LISA spatial correlation analysis for risk distribution
│   └── XXX_F1.py                # Calculate F1-score for visual grounding evaluation
└── llm-graph-builder-main/            # Modified RAG framework
        # Derived from Neo4j llm-graph-builder, for knowledge graph construction and retrieval
Dataset
All datasets used in this project are stored on Google Drive:
[Google Drive Link](Replace with your actual link)
The package includes raw remote sensing patches, ground truth annotations, model detection results and intermediate statistical files for Shenzhen and Hong Kong.
Environment Requirements
Runtime: Python 3.8+, Node.js, Yarn
Database: Neo4j 5.26.0 (Java 17 / Java 21 is mandatory)
Core libraries: GDAL, PyTorch, ModelScope, Sentence-Transformers, pysal, geopandas
Visualization tool: QGIS (for spatial display of GeoJSON files and LISA maps)
RAG Framework Deployment
This RAG module adopts a front-end and back-end separated architecture powered by Neo4j knowledge graph:
Start the local Neo4j database.
Complete environment configuration, then launch frontend and backend services sequentially.
Connect Neo4j via the web interface, upload document files. After parsing, you can perform knowledge retrieval and question answering.
Main Experimental Workflow
1. Dataset Construction
We build the RSIL dataset referring to the EVLab-CMCD framework. By combining remote sensing images and change masks, invalid changed areas are removed to obtain standard industrial land datasets for subsequent experiments.
2. Visual Grounding Experiment
We adopt Qwen2.5-72B as the main large vision-language model. Two types of prompts (standard prompt and RAG-augmented prompt) are designed to implement industrial land detection on remote sensing images. With IoU set to 0.3, we calculate F1-score to quantitatively evaluate model performance on datasets of Shenzhen and Hong Kong.
3. Multi-indicator Calculation
Split full remote sensing images into standard image patches and filter out low-quality samples.
Extract bounding box coordinates of detected industrial land areas.
Use GDAL tools to batch crop corresponding LST (Land Surface Temperature) and NDVI (Normalized Difference Vegetation Index) regions, then compute the average value of LST and NDVI for each industrial land parcel.
4. Ecological Risk Assessment
Combining LST and NDVI indicators, we calculate vegetation coverage ratio, industrial area proportion, temperature excess and vegetation gap between industrial parcels and their surrounding buffer zones. A weighted scoring strategy is applied to evaluate the ecological risk of each industrial land.
5. Risk Clustering & Spatial Analysis
K-Means algorithm is used to divide all industrial lands into five risk levels.
Conduct LISA (Local Indicators of Spatial Association) analysis to explore spatial agglomeration and correlation characteristics of ecological risks across Shenzhen and Hong Kong.
Open the exported GeoJSON and LISA result files with QGIS, sort data by priority, and view regional risk distribution and spatial clustering patterns with differentiated color rendering.
Output Files Description
Detection results: Cropped industrial land images and JSON files recording bounding box coordinates.
Statistical results: Calculated mean LST, mean NDVI and comprehensive risk scores.
Clustering results: Risk classification tables and GeoJSON files for spatial mapping.
LISA analysis results: Spatial correlation reports, clustering distribution files and visualization data of ecological risks.
Notes
Raw remote sensing data is large-sized and not stored in this repository. Please download datasets from the provided Google Drive link.
Neo4j can only run normally with Java 17 or Java 21 installed.
LISA analysis relies on pysal and geopandas, please install relevant dependencies in advance.
This project is developed based on open-source projects: EVLab-CMCD and llm-graph-builder. Please follow relevant open-source licenses when using the code.
