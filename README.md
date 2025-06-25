# Trading Sentiment Analysis Dashboard

A comprehensive Flask web application that analyzes the correlation between market sentiment (Fear & Greed Index) and trading performance. Upload your trading data and sentiment data to gain insights into how market emotions affect your trading results.

## 🚀 Features

- **Data Upload & Processing**: Upload CSV files for sentiment data and trading records
- **Automated Data Merging**: Intelligent matching of sentiment and trading data by date
- **Interactive Visualizations**: 
  - PnL over time charts
  - Sentiment distribution analysis
  - Performance by sentiment classification
  - Correlation heatmaps
  - Cumulative PnL tracking
- **Machine Learning Predictions**: Train models to predict profitable trades based on sentiment
- **Comprehensive Analytics**: Detailed insights including win rates, leverage analysis, and time-based performance
- **Cross-Platform Compatibility**: Works on Windows, macOS, and Linux

## 📋 Requirements

### Python Version
- Python 3.7 or higher

### Dependencies
```
Flask==2.3.3
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
scikit-learn==1.3.0
werkzeug==2.3.7
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/trading-sentiment-analysis.git
cd trading-sentiment-analysis
```

2. **Create a virtual environment**
```bash
python -m venv venv
```

3. **Activate the virtual environment**

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Create required directories**
```bash
mkdir uploads models static/plots
```

## 🚀 Usage

1. **Start the application**
```bash
python app.py
```

2. **Open your browser**
Navigate to `http://localhost:5000`

3. **Upload your data**
- **Sentiment File**: CSV with columns like `date`, `value`, `classification`
- **Trader File**: CSV with columns like `date`, `closedPnL`, `account`, `leverage`

4. **View results**
After processing, you'll be redirected to the dashboard with:
- Key performance metrics
- Interactive charts
- ML model results
- Downloadable insights

## 📊 Data Format Requirements

### Sentiment Data CSV
```csv
date,value,classification
2023-01-01,25,Fear
2023-01-02,75,Greed
2023-01-03,10,Extreme Fear
```

**Required columns:**
- `date`: Date in YYYY-MM-DD format
- `value`: Numerical sentiment value (0-100)
- `classification`: Text classification (Fear, Greed, Extreme Fear, etc.)

### Trading Data CSV
```csv
date,account,closedPnL,leverage,side
2023-01-01,TRADER001,150.50,10,BUY
2023-01-02,TRADER001,-75.25,5,SELL
2023-01-03,TRADER002,200.00,15,BUY
```

**Required columns:**
- `date`: Date in YYYY-MM-DD format
- `closedPnL` (or similar): Profit/Loss amount
- `account` (optional): Trader identifier
- `leverage` (optional): Position leverage
- Additional trading columns are automatically detected

## 🔧 Configuration

### File Size Limits
- Default maximum file size: 100MB
- Configurable in `app.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
```

### Processing Timeout
- Default timeout: 10 minutes for large files
- Configurable in the `process_with_timeout` function

### Supported File Types
- CSV files only
- UTF-8 encoding recommended

## 📈 Analytics Features

### Key Metrics
- Total PnL and average returns
- Win rate and trade distribution
- Sentiment correlation analysis
- Leverage impact assessment
- Time-based performance patterns

### Visualizations
- **Daily PnL Chart**: Track performance over time
- **Sentiment Distribution**: Pie chart of market emotions
- **PnL by Sentiment**: Bar chart showing performance by market mood
- **Correlation Matrix**: Heatmap of variable relationships
- **Cumulative PnL**: Running total of profits/losses

### Machine Learning
- **Random Forest Classifier**: Predicts trade profitability
- **Feature Importance**: Shows which factors matter most
- **Model Accuracy**: Performance metrics and validation

## 🗂️ Project Structure

```
trading-sentiment-analysis/
│
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── templates/            # HTML templates
│   ├── index.html        # Upload page
│   └── dashboard.html    # Results dashboard
│
├── static/               # Static files
│   ├── style.css         # Styling
│   └── plots/           # Generated visualizations
│
├── uploads/              # Temporary file storage
├── models/              # Saved ML models
└── venv/                # Virtual environment (created during setup)
```

## 🐛 Troubleshooting

### Common Issues

**1. "Object of type int64 is not JSON serializable"**
- This is handled automatically by the `make_json_serializable` function
- If you encounter this, ensure all numpy/pandas data is converted before session storage

**2. "Could not detect PnL column"**
- Check that your trader file contains profit/loss data
- Ensure files aren't swapped (sentiment data in trader field)
- Verify column names contain keywords like 'pnl', 'profit', 'closedPnL'

**3. "Files too large to process"**
- Reduce file size or increase timeout in `process_with_timeout`
- Consider processing data in smaller chunks

**4. Matplotlib threading warnings**
- These are harmless warnings due to Flask's threading
- The app uses 'Agg' backend to prevent GUI issues

### File Format Issues
- Ensure dates are in YYYY-MM-DD format
- Check for missing or null values in key columns
- Verify CSV encoding (UTF-8 recommended)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with Flask web framework
- Data visualization powered by Matplotlib, Seaborn, and Plotly
- Machine learning capabilities via scikit-learn
- Inspired by the need to understand sentiment-driven trading patterns

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/trading-sentiment-analysis/issues) page
2. Create a new issue with detailed description
3. Include sample data format and error messages

## 🔮 Future Enhancements

- [ ] Real-time data integration
- [ ] Additional ML algorithms
- [ ] Export functionality for reports
- [ ] Multi-timeframe analysis
- [ ] Portfolio-level analytics
- [ ] API endpoints for programmatic access

---

**Made with ❤️ for traders who want to understand market psychology's impact on their performance.**