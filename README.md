End-to-End Machine Learning Project Pipeline

This repository contains a modular machine learning pipeline structured for reusability,  
scalability, and clarity. It follows best practices for code organization using `config`,  
`logging`, `exception handling`, and a clean `training_pipeline.py` script to control the workflow.

---

Problem Statement

The goal is to predict student performance based on various features like gender, race/ethnicity,  
parental education, lunch type, and test preparation.  

The objective is not just to build a predictive model but to demonstrate a clean, modular pipeline  
that could be easily adapted to other tabular ML problems.

---

Pipeline Overview

The project is broken into four main stages. Each file defines what to do,  
but only one file (`training_pipeline.py`) controls when and how everything runs.

**So the build process is:**

1. Create `data_ingestion.py` → define class + method  
2. Then create `data_transformation.py` → define class + method  
3. Then create `model_trainer.py` → define class + method  
4. Finally — write `training_pipeline.py` to call all those in order

---

**Author:** Adham Khedr  
**Computer Engineering Student — American University in Cairo**  
**GitHub:** [@AdhamKhedr](https://github.com/AdhamKhedr)
