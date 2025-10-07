# ☀️ Solar Pile Optimization Analysis

A Streamlit web application for analyzing solar array pile data to determine optimal racking lines and calculate required pile lengths.

## Features

- Upload Excel files containing solar pile data
- Compare three optimization methods:
  - Simple Line of Best Fit (LOBF)
  - Refined LOBF with vertical shift
  - Dynamic Fixed pile configuration
- Apply North-South constraints between adjacent rows
- Interactive visualizations with Plotly
- Cost analysis with customizable weights
- Export results and analysis

## Usage

1. Upload your Excel file (Grading.xlsx format)
2. Configure optimization parameters in the sidebar
3. Review the analysis results and visualizations
4. Export data as needed

## Data Format

Your Excel file should contain the following columns:
- Easting
- Northing 
- EG (Existing Grade)
- Point Number
- Row information
- Pile information

## Deployment

This app is deployed on Streamlit Community Cloud. Visit the live application at: [Your App URL]

## Local Development

To run locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Version

Version 1.0 - Solar Pile Optimization Analysis Tool