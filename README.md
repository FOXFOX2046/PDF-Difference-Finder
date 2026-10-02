# PDF Difference Finder

A Streamlit application that compares PDF files and automatically highlights differences with revision clouds for engineering drawings and technical documents.

**[Live App](https://pdfdiffdiff.streamlit.app/)** · **Repository:** [GitHub](https://github.com/FOXFOX2046/PDF-Difference-Finder)

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
- Streamlit 1.64+
- OpenCV, Pillow, PyMuPDF
- Runs locally (no cloud/API needed)

## Deployment

- Configure `.streamlit/config.toml` for production (port, CORS, upload limits)
- Deploy to Streamlit Cloud, Docker, or any Python host
- Default: `http://localhost:8501`

For a reverse proxy or custom domain, set `STREAMLIT_BROWSER_SERVER_ADDRESS`
to the public hostname and `STREAMLIT_BROWSER_SERVER_PORT` to the public port
(usually `443` for HTTPS). Keep CORS and XSRF protection enabled. Community
Cloud supplies its deployment settings; do not hardcode a localhost browser
address in the shared configuration. Uploads are limited to 50 MB per PDF.

## Verification

Run the upload regression checks with `python -m unittest discover -s tests -v`.
Uploads can be compared again after changing controls, and replacing a PDF
with different content under the same filename clears the previous results.

## License

See repository for details.
