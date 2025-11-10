````markdown
# 🛰️ Human Trafficking Detection  
### An Integrated Pipeline Utilizing **Natural Language Processing (NLP)**, **Graph Neural Networks (GNNs)**, and **Geospatial Information Systems (GIS)**  

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red.svg)](https://streamlit.io/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-GNN-orange.svg)](https://pytorch.org/)  
[![spaCy](https://img.shields.io/badge/spaCy-NLP-green.svg)](https://spacy.io/)  
[![Folium](https://img.shields.io/badge/Folium-GIS-yellow.svg)](https://python-visualization.github.io/folium/)  
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)  

---

## 📘 Abstract  

This repository implements a **multi-modal, end-to-end system** for detecting, analyzing, and visualizing human trafficking patterns through **textual narratives**, **network relationships**, and **spatial trajectories**.  

It integrates three powerful analytical paradigms:  

1. **Natural Language Processing (NLP)** for entity and relation extraction from testimonies or case data.  
2. **Graph Neural Networks (GNN)** for relational reasoning and community detection across trafficking networks.  
3. **Geographic Information Systems (GIS)** for interactive visualization of spatio-temporal victim movement trajectories.  

The goal is to provide a reproducible and extensible pipeline for **policy research, investigative analytics, and humanitarian data intelligence**.

---

## 🧠 Repository Overview

```bash
Human-Trafficking-Detection-An-Integrated-Pipeline-Utilizing-NLP-GNN-GIS/
│
├── backend/
│   ├── api/
│   │   ├── gis_data.py                 # Geospatial data processing & trajectory building
│   │   ├── graph_queries.py            # Graph API: query, merge, traversal, stats
│   │   ├── nlp_pipeline.py             # Core NLP preprocessing, NER, and entity linking
│   │
│   ├── core/
│   │   ├── dataset_registry.py         # Unified dataset management and metadata storage
│   │
│   ├── geo/
│   │   ├── geo_utils.py                # Fuzzy geocoding, coordinate resolution, cache management
│   │   ├── gazetteer.py                # Gazetteer ingestion, listing, and lookup services
│   │   ├── build_custom_gazetteer.py   # Custom token-to-location gazetteer generator
│   │
│   ├── gis/
│   │   ├── gis_mapper.py               # Robust CSV/ZIP ingesters for GeoNames datasets
│   │
│   └── gnn/
│       ├── model.py                    # Graph Neural Network model (GCN/SAGE variant)
│       ├── trainer.py                  # Training, evaluation, and inference logic
│       ├── utils.py                    # Graph construction, edge indexing, normalization
│
├── frontend/
│   ├── streamlit_app.py                # Streamlit entry point for the UI
│   ├── pages/
│   │   ├── 5_NLP_Processing.py         # NLP interface: run entity extraction & classification
│   │   ├── 6_Graph_Network_Analyzer.py # Interactive GNN-based network exploration
│   │   ├── 8_Map_GIS_Visualizer.py     # Folium-based GIS visualizer with animation & clustering
│   │   ├── 11_Admin_File_Manager.py    # Admin tools: dataset upload, merge, and delete
│
├── data/
│   ├── Africa Dataset.csv              # Core dataset (narratives + locations + metadata)
│   ├── Gazetteer.txt                   # Custom token-based gazetteer
│   ├── geonames-all-cities-with-a-population-1000.csv
│
├── models/
│   ├── trained_gnn.pt                  # Trained PyTorch model weights
│
├── requirements.txt
└── README.md
````

---

## 🧩 System Architecture

```mermaid
flowchart TD
    A[Raw Interview Data / Case Narratives] --> B[NLP Processing Layer]
    B --> C[Entity Extraction (Victim, Perpetrator, Location, Role)]
    C --> D[Relation Extraction (Victim–Trafficker–Location)]
    D --> E[Graph Construction via NetworkX / PyTorch Geometric]
    E --> F[GNN Model (Node Embeddings & Classification)]
    E --> G[GIS Mapper (Trajectory Building)]
    G --> H[Spatio-Temporal Visualization (Streamlit + Folium)]
    F --> I[Insights Dashboard / Graph Metrics]
    I --> J[Exportable Reports (CSV/JSON/HTML)]
```

---

## ⚙️ Core Modules

### 🧾 1. Natural Language Processing (NLP)

#### Objective:

Extract **key entities and relationships** from unstructured narratives describing human trafficking incidents.

#### Pipeline Steps:

| Step | Component                          | Description                                                                   |
| ---- | ---------------------------------- | ----------------------------------------------------------------------------- |
| 1    | **Preprocessing**                  | Text cleaning, sentence segmentation, tokenization                            |
| 2    | **Named Entity Recognition (NER)** | Detect entities: *Victim, Trafficker, Chief, Location, Nationality*           |
| 3    | **Entity Normalization**           | Standardize spellings & resolve aliases                                       |
| 4    | **Relation Extraction**            | Identify “victim-of”, “located-in”, “moved-to”, “controlled-by” relationships |
| 5    | **Coreference Resolution**         | Merge pronouns or indirect mentions into single entities                      |
| 6    | **Output Structuring**             | Generate structured JSON per case                                             |

#### Example NLP Output:

```json
{
  "Victim": "---XYZ---",
  "Traffickers": ["---HA---", "---FK---"],
  "Chiefs": ["---IM---"],
  "Locations": ["Tripoli", "Agadez", "Sabha"],
  "Time Spent (days)": [5, 10, 3]
}
```

---

### 🕸️ 2. Graph Neural Network (GNN)

#### Objective:

Model the **relational topology** of trafficking networks, identifying communities, key perpetrators, and central routes.

#### Components:

| Submodule          | Description                                                                |
| ------------------ | -------------------------------------------------------------------------- |
| `graph_queries.py` | Extracts edges and computes node metrics (degree, betweenness, modularity) |
| `model.py`         | Implements a **Graph Convolutional Network (GCN)** for node classification |
| `trainer.py`       | Trains the model on graph embeddings (PyTorch)                             |
| `utils.py`         | Handles edge normalization, adjacency matrices, and graph construction     |

#### Graph Structure:

* **Nodes:** Victims, Traffickers, Chiefs, Locations
* **Edges:** “interaction”, “movement”, or “hierarchical link”

#### Mathematical Formulation:

The GCN layer is defined as:

$$
H^{(l+1)} = \sigma \left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)
$$

Where:

* ( \tilde{A} = A + I_N ) is the adjacency matrix with self-loops
* ( \tilde{D} ) is the degree matrix
* ( H^{(l)} ) are node embeddings at layer ( l )
* ( W^{(l)} ) are trainable weights
* ( \sigma ) is a non-linear activation (ReLU)

#### Node Classification Example:

* Label 0 → Victim
* Label 1 → Trafficker
* Label 2 → Chief
* Label 3 → Location

---

### 🌍 3. GIS & Spatio-Temporal Visualization

#### Objective:

Map and animate victim trajectories using **Folium** and **Leaflet**, integrating GeoNames gazetteers for geocoding.

#### Key Features:

| Feature                   | Description                                                         |
| ------------------------- | ------------------------------------------------------------------- |
| **Gazetteer Integration** | Supports custom GeoNames `.zip` or `.csv` uploads                   |
| **Fuzzy Matching**        | Approximate string matching for misspelled or regional variations   |
| **Explicit Lookups**      | User-uploaded (location → lat/lon) mappings override auto geocoding |
| **Heatmaps**              | Density visualization of high-traffic areas                         |
| **Trajectory Animation**  | Temporal movement animation using `TimestampedGeoJson`              |

#### Example:

A victim’s trajectory:
`Eritrea → Ethiopia → Sudan → Libya → Italy`
Animated across the timeline using **Time Spent (days)** as duration per hop.

---

## 📊 Dataset Schema

| Column                     | Description                                     |
| -------------------------- | ----------------------------------------------- |
| `Serialized ID`            | Unique identifier for each victim               |
| `Unique ID`                | Case-level identifier                           |
| `Location`                 | Primary location reference                      |
| `City / Locations Crossed` | Path of movement (may contain list-like string) |
| `Time Spent (days)`        | Duration at each stop                           |
| `Perpetrators (NLP)`       | Extracted perpetrator names                     |
| `Chiefs (NLP)`             | Extracted chief names                           |
| `Gender of Victim`         | Gender metadata                                 |
| `Nationality of Victim`    | Country of origin                               |
| `Date of Interview`        | Date reference                                  |
| `Borders Crossed`          | Number of country borders crossed               |

---

## 🔢 Algorithms Implemented

### 1. NLP Relation Extraction (Custom)

```text
For each text:
  → Tokenize sentences
  → Apply pretrained NER (spaCy transformer)
  → Extract entities {Victim, Trafficker, Location, Chief}
  → Detect co-occurrence pairs (Victim–Trafficker)
  → Output edges for graph building
```

### 2. GNN Training Algorithm

```python
for epoch in range(num_epochs):
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    loss = F.cross_entropy(out[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
```

### 3. GIS Trajectory Builder

```python
def build_timestamped_geojson(df, place_col, time_col, default_days=7):
    # 1. Group by Victim ID
    # 2. Sort by Route_Order
    # 3. Resolve lat/lon for each hop
    # 4. Build temporal edges
    # 5. Return Folium-ready GeoJSON
```

---

## 🖥️ Frontend Interface (Streamlit)

| Page                       | Description                                                  |
| -------------------------- | ------------------------------------------------------------ |
| **NLP Processing**         | Upload text dataset, extract entities, and visualize results |
| **Graph Network Analyzer** | Build, train, and visualize trafficking network graphs       |
| **Map & GIS Visualizer**   | Display interactive trajectories, heatmaps, and predictions  |
| **Admin File Manager**     | Manage, rename, and delete datasets or gazetteers            |

---

## 🧩 Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Mehulupase01/Human-Trafficking-Detection-An-Integrated-Pipeline-Utilizing-NLP-GNN-GIS.git
cd Human-Trafficking-Detection-An-Integrated-Pipeline-Utilizing-NLP-GNN-GIS
```

### 2. Create Environment

```bash
conda create -n trafficking python=3.10
conda activate trafficking
pip install -r requirements.txt
```

### 3. Launch Streamlit App

```bash
streamlit run frontend/streamlit_app.py
```

---

## 🧭 Typical Workflow

| Stage      | Action                                               |
| ---------- | ---------------------------------------------------- |
| **Step 1** | Upload processed dataset under “Admin File Manager”  |
| **Step 2** | Run “NLP Processing” to extract entities & locations |
| **Step 3** | Build and train GNN on extracted entities            |
| **Step 4** | Use “Map GIS Visualizer” to map trajectories         |
| **Step 5** | Animate victim movement and generate heatmaps        |
| **Step 6** | Export final results as HTML or CSV                  |

---

## 📈 Results & Insights

| Metric                   | Description                                        | Example                         |
| ------------------------ | -------------------------------------------------- | ------------------------------- |
| **Nodes**                | Entities (victims, traffickers, chiefs, locations) | 3,241                           |
| **Edges**                | Relations or movements                             | 7,835                           |
| **Communities Detected** | Distinct trafficking groups                        | 12                              |
| **Top Central Nodes**    | High betweenness centrality                        | `Tripoli`, `Agadez`, `Khartoum` |
| **Trajectory Duration**  | Mean time from origin to destination               | 48.7 days                       |

---

## 📚 Future Extensions

* Integration with **OSINT** sources (news feeds, UN reports)
* Real-time **network anomaly detection**
* Advanced **spatio-temporal GNNs** (e.g., TGAT, EvolveGCN)
* Cloud-hosted dashboard (Streamlit Cloud or Hugging Face Spaces)

---

## 🧾 Citation

If you use this work, please cite:

> **Upase, Mehul (2025).**
> *Human Trafficking Detection: An Integrated Pipeline Utilizing NLP, Graph Neural Networks, and GIS Framework.*
> Leiden University, Master’s Thesis Repository.

```
@thesis{upase2025humantrafficking,
  author    = {Mehul Upase},
  title     = {Human Trafficking Detection: An Integrated Pipeline Utilizing NLP, GNN, and GIS Framework},
  year      = {2025},
  school    = {Leiden University},
  url       = {https://github.com/Mehulupase01/Human-Trafficking-Detection-An-Integrated-Pipeline-Utilizing-NLP-GNN-GIS}
}
```

---

## 🙌 Acknowledgments

This research is part of the **Leiden University Master’s Thesis** program.
Special thanks to the faculty advisors and the **Human Trafficking Data Lab** for dataset access and feedback.

> Built with ❤️ using **Python, PyTorch, Streamlit, spaCy, Folium, and science.**

```