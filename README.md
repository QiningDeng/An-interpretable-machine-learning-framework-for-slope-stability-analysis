# 📚 Introduction  

This interpretable learning framework applies a **stacked ensemble strategy** to construct predictive models and uses the **SHAP method** to explain model results.  

- **Training dataset**: Derived from Python-based calculations using the strength reduction method. See [related project](https://github.com/QiningDeng/The-programming-performance-of-ChatGPT-in-Slope-stability-analysis).  
- **Ensemble structure**:  
  - Base learners: Elastic Net Regression (ENR), Support Vector Regression (SVR), Decision Tree Regression (DTR), K-Nearest Neighbor Regression (KNNR), and Multilayer Perceptron Regression (MLPR).  
  - Meta learner: An improved Transformer model adapted for tabular data.  
- **Interpretability**: SHAP visualizations reveal feature contributions to predictions and highlight pairwise feature relationships.  

---

# ⚙️ Usage Example  

We provide a trained sample model and its corresponding interpretability analysis:  

- **Trained model**:  
  Located in the folder `04.Final model ensemble construction`. Includes both the packed model files and Python scripts for direct prediction.  

- **SHAP analysis**:  
  All SHAP results and related scripts are available in the folder `05.SHAP analysis`.  

- **Reproducibility**:  
  The original database and training code are included in the repository for full replication. Please download the complete `.zip` package to use.  

---

# ✨ Contact  

If you have any questions, please feel free to reach out via email:  

📧 1974739605ngyx@gmail.com  
