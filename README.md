# `susubert`: BERT for Product Matching

Welcome to susubert, a robust pipeline designed for training BERT models on pairwise product deduplication/product matching. This repository utilizes Kubeflow pipelines and Google Cloud Functions to efficiently manage data and training processes, making it ideal for handling dynamic datasets in retail and e-commerce environments.

## Features

- **Kubeflow Pipelines:** Utilize scalable machine learning pipelines capable of handling complex workflows. Find these in the `pipelines/` directory.
- **Google Cloud Functions:** Automate tasks related to data updates and model training with two key functions:
  - `update_clusters` in the `functions/` directory: Monitors for new product additions or conditions warranting retraining, and triggers the matching pipeline.
  - `update_training` in the `functions/` directory: Triggers retraining of the model when new master products are added to the database.
- **Scripts for Training and Utility:** Essential scripts required for processing data and training the models are available in the `src/` directory.


