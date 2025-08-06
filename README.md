
# Turkish Comment Analysis - End-to-End NLP Pipeline

This project implements a comprehensive Natural Language Processing (NLP) pipeline for analyzing Turkish customer complaints from Setur tourism company. The analysis combines traditional text processing methods with state-of-the-art BERT-based approaches to extract insights from customer feedback.

## 📋 Project Overview

This project performs a complete analysis of Turkish tourism complaints with the following components:

1. **Data Collection & Preprocessing**
   - Scraped 948+ Turkish complaints from Setur tourism company
   - Applied comprehensive text cleaning and Turkish-specific preprocessing
   - Handled Turkish character encoding and normalization

2. **Advanced NLP Analysis**
   - **Morphological Analysis**: Used Turkish BERT model (`dbmdz/bert-base-turkish-cased`) for POS tagging and lemmatization
   - **Sentiment Analysis**: Dual approach with lexicon-based and BERT-based methods
   - **Topic Modeling**: BERT-enhanced clustering to discover complaint themes
   - **Linguistic Feature Extraction**: Generated contextual embeddings and morphological features

3. **Key Findings**
   - **Sentiment Distribution**: 97.8% negative sentiment (BERT), 52.1% neutral (Lexicon)
   - **Main Topics**: Reservation issues, service quality, payment problems, accommodation issues
   - **Method Comparison**: 26.6% agreement between lexicon and BERT sentiment analysis
   - **Language Insights**: Comprehensive Turkish morphological analysis with 4,217 tokens processed

### Technologies Used

- **Turkish BERT**: `dbmdz/bert-base-turkish-cased` for advanced language understanding
- **Machine Learning**: K-means clustering, TF-IDF vectorization, semantic embeddings
- **Text Processing**: Custom Turkish lexicons, negation handling, morphological analysis
- **Visualization**: Comprehensive charts, word clouds, and statistical plots
- **API Integration**: Support for ChatGPT API for enhanced analysis capabilities

## 🤖 ChatGPT API Integration

This project supports integration with OpenAI's ChatGPT API for enhanced analysis capabilities:

### Features Available via ChatGPT API

1. **Advanced Sentiment Analysis**
   - More nuanced understanding of Turkish sentiment
   - Context-aware emotion detection
   - Handling of complex linguistic patterns

2. **Enhanced Topic Classification**
   - Semantic topic labeling and description
   - Custom prompt-based categorization
   - Multi-level topic hierarchies

3. **Automated Insights Generation**
   - Natural language summaries of findings
   - Business recommendations based on complaints
   - Trend analysis and pattern detection

### API Setup Instructions

1. **Get OpenAI API Key**:
   ```cmd
   # Set environment variable
   set OPENAI_API_KEY=your_api_key_here
   ```

2. **Install OpenAI Library**:
   ```cmd
   pip install openai
   ```

3. **Configure API in Notebook**:
   ```python
   import openai
   import os
   
   # Setup API key
   openai.api_key = os.getenv("OPENAI_API_KEY")
   
   # Example usage for Turkish sentiment analysis
   def analyze_with_chatgpt(text):
       response = openai.ChatCompletion.create(
           model="gpt-3.5-turbo",
           messages=[{
               "role": "system",
               "content": "You are an expert in Turkish sentiment analysis for tourism complaints."
           }, {
               "role": "user", 
               "content": f"Bu Türkçe şikayet metninin duygusunu analiz edin: {text}"
           }],
           temperature=0.3
       )
       return response.choices[0].message.content
   ```

4. **Prompt Engineering for Turkish Analysis**:
   - Custom prompts for Turkish language nuances
   - Tourism-specific context understanding
   - Multi-step analysis workflows

### What the ChatGPT API Integration Does

The ChatGPT API integration adds a powerful layer of analysis to complement our local BERT models:

#### 1. **Enhanced Sentiment Analysis**
```python
# Example: Analyzing Turkish complaint sentiment
prompt = "Bu şikayet metninin duygusunu analiz edin: [metin]"
# Returns: "Negative - Customer expresses frustration with service quality"
```
- **Context-Aware**: Understands nuanced Turkish expressions and cultural context
- **Reasoning**: Provides explanations for sentiment classifications
- **Confidence Scoring**: Offers detailed confidence levels for each prediction

#### 2. **Intelligent Topic Extraction**
```python
# Example: Extracting main complaint topics
prompt = "Bu şikayetin ana konusunu belirleyin: [metin]"  
# Returns: "Rezervasyon - Otel rezervasyonu ile ilgili sorunlar"
```
- **Semantic Understanding**: Goes beyond keyword matching
- **Hierarchical Topics**: Creates topic hierarchies (main → sub-topics)
- **Turkish-Specific**: Handles tourism industry terminology in Turkish

#### 3. **Automated Text Summarization**
```python
# Example: Summarizing long complaints
prompt = "Bu şikayeti 2-3 cümlede özetleyin: [metin]"
# Returns: "Müşteri otel rezervasyonunda sorun yaşadı. Ödeme alındı ancak rezervasyon onaylanmadı."
```
- **Key Point Extraction**: Identifies most important complaint aspects
- **Concise Summaries**: Creates actionable business insights
- **Multi-language**: Processes Turkish text with high accuracy

#### 4. **Cross-Validation with Local Models**
- **Agreement Analysis**: Compares ChatGPT results with BERT predictions
- **Confidence Weighting**: Uses agreement levels to improve accuracy
- **Error Detection**: Identifies cases where models disagree for manual review

### Benefits of API Integration

- **Complementary Analysis**: Combines local BERT models with cloud-based GPT
- **Enhanced Accuracy**: Cross-validation between different AI approaches  
- **Flexible Processing**: Custom prompts for specific analysis requirements
- **Scalable Solution**: Cloud processing for large datasets
- **Business Intelligence**: Generates actionable insights in natural language

### 🔧 Practical Implementation in the Project

The ChatGPT API integration is implemented in the notebook with the following workflow:

#### Step 1: Setup and Configuration
```python
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_response_from_openai(prompt):
    """Get response from OpenAI API using the provided prompt"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150
    )
    return response.choices[0].message.content
```

#### Step 2: Multi-Analysis Function
```python
def analyze_with_chatgpt(texts, max_samples=10):
    """
    Analyze Turkish complaints using ChatGPT API for:
    - Sentiment Analysis
    - Topic Extraction  
    - Text Summarization
    """
    results = {
        "sentiment_analysis": [],
        "topic_extraction": [],
        "summarization": [],
        "original_texts": []
    }
    
    sample_texts = texts[:max_samples]
    
    for i, text in enumerate(sample_texts):
        truncated_text = text[:200]  # Limit token usage
        
        # 1. Sentiment Analysis
        sentiment_prompt = f"{truncated_text} Bu şikayetin duygusunu belirtin (pozitif/negatif/nötr)"
        sentiment_result = get_response_from_openai(sentiment_prompt)
        
        # 2. Topic Extraction
        topic_prompt = f"{truncated_text} Şikayet konusunu belirtin (rezervasyon/hizmet/ürün)"
        topic_result = get_response_from_openai(topic_prompt)
        
        # 3. Summarization
        summary_prompt = f"Bu şikayeti özetleyin: {truncated_text}"
        summary_result = get_response_from_openai(summary_prompt)
        
        results["sentiment_analysis"].append(sentiment_result)
        results["topic_extraction"].append(topic_result)
        results["summarization"].append(summary_result)
        results["original_texts"].append(truncated_text)
    
    return results
```

#### Step 3: Integration with BERT Results
```python
def compare_analysis_methods():
    """Compare ChatGPT results with local BERT analysis"""
    
    # Get BERT predictions
    bert_sentiments = sentiment_df['bert_sentiment'].tolist()
    
    # Get ChatGPT analysis
    chatgpt_results = analyze_with_chatgpt(df['cleaned_text'].tolist(), max_samples=20)
    
    # Compare and visualize results
    comparison_df = pd.DataFrame({
        'text': chatgpt_results['original_texts'],
        'bert_sentiment': bert_sentiments[:len(chatgpt_results['original_texts'])],
        'chatgpt_sentiment': chatgpt_results['sentiment_analysis'],
        'chatgpt_topic': chatgpt_results['topic_extraction'],
        'chatgpt_summary': chatgpt_results['summarization']
    })
    
    return comparison_df
```

### 📊 Expected Results from API Integration

When you run the ChatGPT analysis, you'll get enhanced insights like:

**Example Output:**
```
🔄 Analyzing 10 samples with ChatGPT...
   Processing sample 1/10...
   Processing sample 2/10...
   ...
✅ Analysis completed!

Results Summary:
├── Sentiment Analysis: 8 Negative, 2 Neutral
├── Main Topics: Rezervasyon (60%), Hizmet Kalitesi (40%)
└── Key Issues: Ödeme sorunları, rezervasyon iptalleri, müşteri hizmetleri yetersizliği
```

### 💡 Business Value Added

1. **Deeper Insights**: Understands context better than keyword-based approaches
2. **Actionable Summaries**: Generates business-ready complaint summaries
3. **Quality Assurance**: Cross-validates local model predictions
4. **Scalable Analysis**: Processes Turkish text with human-level understanding

## 🚀 How to Run the Project

### Prerequisites

- **Python 3.11.5** or higher
- **Jupyter Notebook** or JupyterLab
- **8GB+ RAM** (recommended for BERT models)
- **Internet connection** (for downloading BERT models)

### Installation Steps

1. **Unzip and Navigate to Project**:
   ```cmd
   cd setur_case
   ```

2. **Install Dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```
   
   **Key packages installed**:
   - `transformers` - Hugging Face BERT models
   - `torch` - Deep learning framework
   - `pandas`, `numpy` - Data processing
   - `matplotlib`, `seaborn` - Visualization
   - `scikit-learn` - Machine learning utilities
   - `openai` - ChatGPT API integration (optional)

3. **Launch Jupyter Notebook**:
   ```cmd
   jupyter notebook
   ```
   Or for JupyterLab:
   ```cmd
   jupyter lab
   ```

4. **Open and Run the Analysis**:
   - Navigate to `turkish_comment_analysis.ipynb`
   - Execute cells sequentially from top to bottom
   - **Total runtime**: 15-30 minutes (depending on hardware)

### Alternative: Run Data Scraper (Optional)

If you want to collect fresh data:
```cmd
python prepare_dataset.py
```
This will scrape new complaints from the source website.

## 📁 Project Structure

```
setur_case/
├── turkish_comment_analysis.ipynb    # Main analysis notebook
├── prepare_dataset.py                # Data scraping script
├── setur_complaints_new.csv          # Dataset (948 complaints)
├── setur_complaints_new.json         # Dataset (JSON format)
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── Case_Report.pdf                   # Detailed analysis report
└── figures/                          # Generated visualizations
    ├── text_preprocessing.png
    ├── topic_model_result.png
    ├── top_words.png
    ├── top_words_of_topics.png
    └── turkish_comment.png
```

## 🔍 Analysis Pipeline

### 1. Data Loading & Exploration
- Load 948 Turkish complaints from CSV/JSON
- Explore data structure and basic statistics
- Handle missing values and data quality issues

### 2. Text Preprocessing & Cleaning
- Turkish-specific text normalization
- Remove HTML tags, URLs, and special characters
- Handle Turkish character encoding (ç, ş, ğ, ü, ö, ı)
- Tokenization optimized for Turkish morphology

### 3. Linguistic Analysis (Turkish BERT)
- Load `dbmdz/bert-base-turkish-cased` model
- **POS Tagging**: 94.8% nouns, 3.1% adjectives, 2.1% verbs
- **Lemmatization**: 1,988 unique lemmas identified
- **Embeddings**: 4,217 contextual word embeddings generated
- **Morphological Features**: Turkish-specific linguistic analysis

### 4. Sentiment Analysis (Dual Approach)
- **Lexicon-based**: Custom Turkish sentiment dictionaries
  - 51 positive words, 56 negative words, 26 negation words
  - Handles Turkish negation patterns (değil, yok, etc.)
- **BERT-based**: Fine-tuned transformer model
  - 97.8% negative, 2.2% positive sentiment detected
  - High confidence scores (avg. 0.99)
- **ChatGPT API**: Enhanced contextual analysis (optional)
  - Advanced sentiment reasoning for complex cases
  - Cross-validation with local models

### 5. Topic Modeling (BERT-Enhanced)
- **Method**: K-means clustering on BERT embeddings
- **Topics Discovered**:
  - **Rezervasyon & Ödeme** (40.9% of complaints)
  - **Genel Değerlendirme** (14.8%)
  - **Hizmet Kalitesi** (8.6%)
  - **Konaklama** (varies by analysis)

### 6. Visualization & Insights
- Word frequency analysis by sentiment
- Topic distribution charts
- Sentiment comparison (Lexicon vs BERT)
- Interactive visualizations and statistical summaries

## 📊 Key Results

### Sentiment Analysis Results
- **BERT Model**: 97.8% negative, 2.2% positive
- **Lexicon Model**: 52.1% neutral, 26.6% negative, 21.3% positive  
- **Method Agreement**: 26.6% (indicates complexity of Turkish sentiment)

### Topic Analysis Results
- **6 Main Topics** identified through clustering
- **Reservation Issues** dominate complaints (40.9%)
- **Service Quality** and **Payment Problems** are secondary concerns
- **BERT Embeddings** provide semantic clustering superior to TF-IDF

### Technical Performance
- **Processed**: 948 complaints, 4,217 tokens
- **Model**: 12-layer Turkish BERT transformer
- **Embeddings**: 768-dimensional contextual vectors
- **Clustering**: 6-topic K-means with 85%+ coherence

## 🛠️ Troubleshooting

### Common Issues

1. **Memory Issues with BERT**:
   - Reduce batch size in notebook
   - Use CPU instead of GPU: Set `device = "cpu"`
   - Close other applications

2. **Turkish Encoding Issues**:
   - Ensure UTF-8 encoding for all text files
   - Check locale settings: `locale.getdefaultlocale()`

3. **Missing Packages**:
   ```cmd
   pip install transformers torch pandas numpy matplotlib seaborn scikit-learn
   ```

4. **BERT Model Download Issues**:
   - Ensure internet connection
   - Models download automatically on first run
   - ~500MB download for Turkish BERT model

### Performance Optimization
- **GPU Usage**: Automatic if CUDA available
- **Memory Management**: Models loaded once and reused
- **Batch Processing**: Comments processed in optimized batches

## 📈 Expected Runtime

| Component | Time (CPU) | Time (GPU) |
|-----------|------------|------------|
| Data Loading | 30s | 30s |
| Preprocessing | 2-3 min | 2-3 min |
| BERT Analysis | 15-20 min | 5-7 min |
| Sentiment Analysis | 8-10 min | 3-4 min |
| Topic Modeling | 5-8 min | 2-3 min |
| **Total** | **30-41 min** | **12-17 min** |

## 🎯 Business Value

This analysis provides actionable insights for Setur tourism:

1. **Customer Pain Points**: Identified reservation and payment as primary issues
2. **Service Improvement**: Specific areas needing attention (accommodation, communication)  
3. **Sentiment Trends**: Nearly universal negative sentiment requires immediate action
4. **Topic Prioritization**: Data-driven focus on reservation system improvements

## 📚 References & Citations

- **Turkish BERT Model**: `dbmdz/bert-base-turkish-cased` (Hugging Face)
- **NLP Framework**: Transformers library (Hugging Face)
- **Data Source**: Customer complaints from tourism platform
- **Analysis Methods**: Combined lexicon-based and transformer-based approaches

---

*This project demonstrates advanced NLP techniques applied to Turkish language processing, combining traditional methods with cutting-edge transformer models for comprehensive text analysis.*

