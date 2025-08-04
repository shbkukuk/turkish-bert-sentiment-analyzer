
## How to Run the Notebook

### Prerequisites
- Python 3.11.5 or higher
- Jupyter Notebook or JupyterLab
- Required Python packages (see requirements.txt)

### Installation Steps

1. **Unzip the Folder**:
   ```bash
   unzip <directory_path>
   cd setur_case
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   Or if you prefer JupyterLab:
   ```bash
   jupyter lab
   ```

4. **Open the notebook**:
   Navigate to and open `turkish_comment_analysis.ipynb`

5. **Run the analysis**:
   - Execute cells sequentially from top to bottom
   - The notebook is designed to run end-to-end
   - Total runtime: approximately 15-30 minutes (depending on hardware)

### Required Files
Ensure the following files are in the project directory:
- `setur_complaints_new.csv` or `setur_complaints_new.json` (dataset)
- `turkish_comment_analysis.ipynb` (main notebook)
- `requirements.txt` (dependencies)

### Key Dependencies
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `matplotlib` & `seaborn`: Data visualization
- `transformers`: BERT models for NLP
- `torch`: Deep learning framework
- `scikit-learn`: Machine learning utilities
- `wordcloud`: Word cloud generation
- `nltk`: Natural language processing toolkit

### Troubleshooting
- If you encounter memory issues with BERT models, consider reducing batch sizes or using CPU instead of GPU
- For Turkish text processing issues, ensure proper UTF-8 encoding
- Missing packages can be installed individually using `pip install <package-name>`

