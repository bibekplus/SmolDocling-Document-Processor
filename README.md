# SmolDocling Document Processor

SmolDocling Document Processor is a lightweight application that processes document images or PDFs and converts them into structured formats such as Markdown, HTML, or JSON. It leverages the **SmolDocling** Visual Language Model (VLM) for document understanding, making it ideal for extracting semantic meaning from diverse document types.

---

## Features

- **Input Options**:
  - Upload PDF or image files.
  - Provide a URL to a remote PDF or image.

- **Output Formats**:
  - **Markdown**: For easy viewing and copy-pasting.
  - **HTML**: Preserves rich layout for web rendering.
  - **JSON**: Ideal for developers and downstream automation.

- **Capabilities**:
  - Understands full pages of diverse document types (e.g., academic papers, business forms, patents).
  - Extracts:
    - Paragraphs, headers, and footers.
    - Tables and their structure (including merged cells and headers).
    - Code blocks with indentation.
    - Equations (LaTeX format).
    - Charts and captions.
    - Lists and nested lists.
  - Maintains spatial layout and reading order.
  - Outputs results in structured **DocTags**, convertible into Markdown, HTML, and JSON.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bibekplus/SmolDocling-Document-Processor.git
   cd SmolDocling-Document-Processor

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt

3. Ensure you have the necessary backend setup for MLX (optimized for Apple Silicon but works on other platforms).

<hr>

## Usage
1. Run the application:

   ```bash
   python main.py


2. Open the Gradio interface in your browser.

3. Upload a document or provide a URL, select the desired output format, and click Process Document.

4. View the structured output, preview the document, and download the results.

## Requirements
 - Python 3.8 or higher.
 - Dependencies listed in requirements.txt (e.g., gradio, torch, pdf2image, mlx-vlm).

## Notes
- This app is optimized for Apple Silicon (Metal backend with MLX) but works on other machines with appropriate setup.
- SmolDocling is a compact (256MB) Visual Language Model designed for document understanding.

## License
This project is licensed under the MIT License. See the LICENSE file for details.