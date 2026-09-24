GreenQO: A Value-Stabilized Graph Neural Network Framework for Scale-Aware Energy-Efficient Query Optimization

Overview

This repository contains the machine learning models, feature extraction pipelines, training/inference scripts, and plotting utilities required to reproduce the experimental results presented in our EDBT submission.
System Architecture & Prerequisites

    Host Environment: Windows Host with PostgreSQL installed.

    ML Environment: WSL (Windows Subsystem for Linux) Machine Learning Environment.

    Development/Execution Tool: VS Code (integrated terminal used for running scripts).

Stage 1: Telemetry Collection

Metrics were initially collected using external telemetry tools and exported to the WSL Machine Learning Environment execution workspace.

Stage 2: Data Generation, Training, and Evaluation
The repository is structured into distinct folders corresponding to data preparation, baseline models, GNN models, and final evaluation. Follow the steps below sequentially.
Datasets & Workloads

    Datasets: Contains tpch_1gb_energy.xlsx, tpch_3gb_energy.xlsx, and tpch_5gb_energy.xlsx.

    Workloads: Located in tpch-dbgen/TPCH_queries.

Database Setup & Loading

PostgreSQL must be hosted on your Windows machine, and TPC-H data (1 GB, 3 GB, and 5 GB) must be loaded independently for this phase.

    Generate execution plans without execution time by running:
    ./run_tpch_explain_without_execution.sh

Phase 1 (MLP Baseline)

Navigate to the MLP phase folder and execute the scripts in order:

    Dataset Builder: python3 dataset_builder.py

    MLP Model Definition: python3 model_MLP.py

    Train Energy Model: python3 train_energy_MLP.py

    Train Time Model: python3 train_time_MLP.py

    Check Weights: python3 check_weights.py

    Evaluate Model: python3 evaluate_MLP.py

    Compute Mean Metrics: python3 mean_MLP_energy_time.py

Phase 2 (GNN / GreenQO)

Navigate to the GNN phase folder and execute the scripts in order:

    Graph Builder: python3 graph_builder.py

    GNN Model Definition: python3 model_GNN.py

    Train Energy Model: python3 train_energy_GNN.py

    Train Time Model: python3 train_time_GNN.py

    Evaluate Models:

        python3 evaluate_energy_GNN.py

        python3 evaluate_time_GNN.py

    Compute Mean Metrics: python3 mean_GNN_energy_time.py

Final Evaluation & Plotting

    Optimizer Selector: Automatically evaluates both eco and performance modes:
    python3 optimizer_selector.py 
    Generate Baselines: python3 generate_paper_baselines.py
    Generate Plots: python3 generate_plot.py
