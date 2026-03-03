# PDF Difference Finder

A Streamlit application that compares PDF files and automatically highlights differences with revision clouds for engineering drawings and technical documents.

**Repository:** [https://github.com/FOXFOX2046/PDF-Difference-Finder](https://github.com/FOXFOX2046/PDF-Difference-Finder)

## Features

- **Single Pair Mode**: Compare two PDFs side-by-side with automatic difference highlighting
- **Batch Mode**: Process multiple PDF pairs at once and download results as ZIP
- **Revision Clouds**: Red engineering-style clouds mark all differences (editable in Acrobat/Bluebeam)
- **Visual Overlay**: Green semi-transparent overlay shows changed areas
- **Export Options**: Download highlight PDFs and annotated PDFs (with editable clouds)
- **JPEG Compression**: Optional compression to reduce output file size
- **Page-by-Page View**: Select specific page for single-pair comparison

## Installation

1. Clone the repository:
```bash
git clone https://github.com/FOXFOX2046/PDF-Difference-Finder.git
cd PDF-Difference-Finder
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Choose **Single Pair** or **Batch Mode** in the sidebar

3. **Single Pair**: Upload PDF A and PDF B, select page, adjust sensitivity

4. **Batch Mode**: Upload multiple PDFs in each slot; pairs are processed sequentially

5. Download highlight PDFs and annotated PDFs (with revision clouds) from the sidebar

## Project Structure

```
PDF-Difference-Finder/
├── app.py                  # Main Streamlit app
├── MadFoxLogo.png          # Sidebar app icon
├── requirements.txt        # Dependencies
├── .streamlit/
│   └── config.toml         # Streamlit config
└── src/core/
    ├── pdf_render.py      # PDF → images
    ├── diff_mask.py       # Difference mask generation
    ├── regions.py         # Region detection
    ├── annotate.py        # Overlay + revision clouds
    ├── export.py          # PDF/PNG export
    ├── pdf_annotate.py    # PDF cloud annotations
    └── security.py        # Security helpers
```

## Requirements

- Python 3.8+
- Streamlit 1.28+
- OpenCV, Pillow, PyMuPDF
- Runs locally (no cloud/API needed)

## Deployment

- Configure `.streamlit/config.toml` for production (port, CORS, upload limits)
- Deploy to Streamlit Cloud, Docker, or any Python host
- Default: `http://localhost:8501`

## License

See repository for details.
