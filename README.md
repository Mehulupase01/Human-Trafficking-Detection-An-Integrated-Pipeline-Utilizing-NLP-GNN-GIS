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

This repository implements a **multi-modal, end-to-end system** for detecting, analyzing, and visualizing human trafficking patterns from narrative data.  

It integrates three major analytical paradigms:  

1. **Natural Language Processing (NLP)** – entity and relation extraction from testimonies or case documents.  
2. **Graph Neural Networks (GNN)** – modeling relational structures and discovering trafficking hierarchies.  
3. **Geospatial Information Systems (GIS)** – visualizing spatio-temporal trajectories and movement networks.  

The objective is to create a reproducible, extensible analytical framework for **policy analysis**, **law enforcement**, and **research on human trafficking networks**.

---

## 🧠 Repository Overview

```bash
Human-Trafficking-Detection-An-Integrated-Pipeline-Utilizing-NLP-GNN-GIS/
│
├── backend/
│   ├── api/
│   │   ├── gis_data.py                 # Geospatial data processing & trajectory building
│   │   ├── graph_queries.py            # Graph API: query, merge, traversal, and network metrics
│   │   ├── nlp_pipeline.py             # Entity extraction and relation detection
│   │
│   ├── core/
│   │   ├── dataset_registry.py         # Dataset registration and management
│   │
│   ├── geo/
│   │   ├── geo_utils.py                # Fuzzy geocoding and coordinate resolution
│   │   ├── gazetteer.py                # Gazetteer ingestion and active lookup
│   │
│   ├── gis/
│   │   ├── gis_mapper.py               # Custom CSV/GeoNames ingestion and mapping
│   │
│   └── gnn/
│       ├── model.py                    # Graph Neural Network model
│       ├── trainer.py                  # Model training and evaluation
│       ├── utils.py                    # Graph preprocessing utilities
│
├── frontend/
│   ├── streamlit_app.py                # Streamlit main entry point
│   ├── pages/
│   │   ├── 5_NLP_Processing.py         # NLP interface for text extraction
│   │   ├── 6_Graph_Network_Analyzer.py # GNN visualization and analysis
│   │   ├── 8_Map_GIS_Visualizer.py     # GIS map rendering and trajectory animation
│
├── data/
│   ├── Africa Dataset.csv              # Sample dataset
│   ├── Gazetteer.txt                   # Custom gazetteer
│
├── models/
│   ├── trained_gnn.pt                  # Saved PyTorch GNN weights
│
├── requirements.txt
└── README.md
````

---

## 🧩 System Architecture

> ⚙️ This system processes narrative data → extracts entities → builds graph relationships → geocodes locations → visualizes trajectories.

```text
┌─────────────────────────────┐
│  Raw Narrative Dataset      │
│ (Interview / Report Data)   │
└──────────────┬──────────────┘
               │
               ▼
      ┌────────────────────┐
      │ NLP Processing     │
      │ • Entity Extraction│
      │ • Relation Mapping │
      └────────┬───────────┘
               │
               ▼
     ┌───────────────────────┐
     │ Graph Neural Network  │
     │ • Graph Construction  │
     │ • Node Classification │
     └──────────┬────────────┘
                │
                ▼
     ┌───────────────────────┐
     │ GIS Visualizer (Map)  │
     │ • Trajectories        │
     │ • Heatmaps            │
     └───────────────────────┘
```

---

## 🧾 NLP Pipeline

| Step | Module                             | Description                                               |
| ---- | ---------------------------------- | --------------------------------------------------------- |
| 1    | **Preprocessing**                  | Tokenization, sentence segmentation, and normalization    |
| 2    | **Named Entity Recognition (NER)** | Extract Victims, Traffickers, Chiefs, and Locations       |
| 3    | **Coreference Resolution**         | Merge repeated mentions and pronouns                      |
| 4    | **Relation Extraction**            | Identify links between entities (e.g., Victim–Trafficker) |
| 5    | **Output Structuring**             | Export structured entity data to JSON                     |

**Example Output:**

```json
{
  "Victim": "Amina Yusuf",
  "Traffickers": ["Hassan Ali", "Fatou Keita"],
  "Locations": ["Tripoli", "Agadez", "Sabha"],
  "Chief": "Ibrahim Musa",
  "Time Spent (days)": [5, 10, 3]
}
```

---

## 🧮 Graph Neural Network (GNN)

| Concept   | Description                                       |
| --------- | ------------------------------------------------- |
| **Nodes** | Victims, Traffickers, Chiefs, Locations           |
| **Edges** | Relationships or movements                        |
| **Goal**  | Predict community affiliations or influence ranks |

### 📘 Model Equation

The **Graph Convolutional Network (GCN)** layer is defined as:

$$
H^{(l+1)} = \sigma \left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)
$$

Where:

* ( \tilde{A} = A + I ) is adjacency with self-loops
* ( H^{(l)} ) is the node embedding matrix
* ( W^{(l)} ) are learnable weights
* ( \sigma ) is a ReLU activation

---

## 🌍 GIS Visualization

**Purpose:** Map and animate victim movements based on extracted location sequences.

| Feature                  | Description                                              |
| ------------------------ | -------------------------------------------------------- |
| **Gazetteer Matching**   | Integrates GeoNames and custom gazetteers                |
| **Fuzzy Resolution**     | Handles misspelled / partial names                       |
| **Heatmap Layer**        | Visualizes trafficking intensity                         |
| **Trajectory Animation** | Uses `TimestampedGeoJson` to animate spatial transitions |

**Example Path:**
`Eritrea → Ethiopia → Sudan → Libya → Italy`

Each leg of the route is assigned a duration via the `Time Spent (days)` column.

---

## 🗂️ Dataset Schema

| Column                   | Description                |
| ------------------------ | -------------------------- |
| Serialized ID            | Record ID                  |
| Unique ID                | Case ID                    |
| Location                 | Base location              |
| City / Locations Crossed | Full migration route       |
| Time Spent (days)        | Duration per stop          |
| Perpetrators (NLP)       | Extracted perpetrators     |
| Chiefs (NLP)             | Extracted chiefs           |
| Gender of Victim         | Gender metadata            |
| Nationality of Victim    | Country of origin          |
| Borders Crossed          | Number of border crossings |

---

## 🧠 Algorithms Overview

### NLP Relation Extraction

```python
for text in dataset:
    entities = nlp_model(text)
    victims = extract_victims(entities)
    traffickers = extract_traffickers(entities)
    relations = build_relations(victims, traffickers)
```

### GNN Training

```python
for epoch in range(epochs):
    out = model(graph.x, graph.edge_index)
    loss = criterion(out[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
```

### GIS Trajectory Builder

```python
def build_timestamped_geojson(df, place_col, time_col, default_days=7):
    coords = resolve_locations(df[place_col])
    for a, b in zip(coords[:-1], coords[1:]):
        add_segment(a, b, duration=default_days)
```

---

## 💻 Frontend (Streamlit Interface)

| Page                          | Description                                    |
| ----------------------------- | ---------------------------------------------- |
| `5_NLP_Processing.py`         | Run entity extraction and display results      |
| `6_Graph_Network_Analyzer.py` | Visualize and analyze trafficking graphs       |
| `8_Map_GIS_Visualizer.py`     | Display geospatial trajectories and animations |
| `Admin_File_Manager.py`       | Manage datasets and gazetteers                 |

---

## ⚙️ Installation & Execution

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Mehulupase01/Human-Trafficking-Detection-An-Integrated-Pipeline-Utilizing-NLP-GNN-GIS.git
cd Human-Trafficking-Detection-An-Integrated-Pipeline-Utilizing-NLP-GNN-GIS
```

### 2️⃣ Create an Environment

```bash
conda create -n trafficking python=3.10
conda activate trafficking
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run frontend/streamlit_app.py
```

---

## 🧭 Typical Workflow

| Step | Description                                 |
| ---- | ------------------------------------------- |
| 1    | Upload preprocessed dataset                 |
| 2    | Run NLP pipeline for entity extraction      |
| 3    | Build and train GNN                         |
| 4    | Open GIS Visualizer to explore trajectories |
| 5    | Export visualizations as HTML               |

---

## 📈 Results Snapshot

| Metric               | Example                         |
| -------------------- | ------------------------------- |
| Nodes                | 3,241                           |
| Edges                | 7,835                           |
| Communities          | 12                              |
| Top Nodes            | `Tripoli`, `Khartoum`, `Agadez` |
| Mean Travel Duration | 48.7 days                       |

---

## 🧾 Citation

If you use this project, please cite:

> **Upase, Mehul (2025).**
> *Human Trafficking Detection: An Integrated Pipeline Utilizing NLP, Graph Neural Networks, and GIS Framework.*
> Leiden University, Master’s Thesis Repository.

```bibtex
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

Developed as part of the **Master’s Thesis** at **Leiden University**.
Gratitude to the **Human Trafficking Data Lab**, supervisors, and reviewers for their invaluable input.

> Built with ❤️ using **Python, PyTorch, Streamlit, spaCy, and Folium.**

```