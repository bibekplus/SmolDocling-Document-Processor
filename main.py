import gradio as gr
import os
import tempfile
from pathlib import Path
from io import BytesIO
from urllib.parse import urlparse
import requests
import webbrowser
from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes
import torch
import json
# Import MLX VLM and Docling related modules
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config, stream_generate
from docling_core.types.doc import ImageRefMode
from docling_core.types.doc.document import DocTagsDocument, DoclingDocument

def load_input_resource(input_path):
    """Load image or PDF file from path or URL and return list of images."""
    images = []
    
    if urlparse(input_path).scheme != "":  # it is a URL
        response = requests.get(input_path, stream=True, timeout=10)
        response.raise_for_status()
        content = BytesIO(response.content)
        
        # Check if it's a PDF by examining magic numbers
        content.seek(0)
        if content.read(4) == b"%PDF":
            content.seek(0)
            # Convert PDF pages to images
            pdf_images = convert_from_bytes(content.read())
            images.extend(pdf_images)
        else:
            content.seek(0)
            images.append(Image.open(content))
    else:
        # Local file
        file_path = Path(input_path)
        if file_path.suffix.lower() == ".pdf":
            # Convert PDF pages to images
            pdf_images = convert_from_path(str(file_path))
            images.extend(pdf_images)
        else:
            images.append(Image.open(file_path))
            
    return images

def load_model():
    """Load the SmolDocling model with MLX optimizations"""
    import mlx.core as mx
    
    # Force Metal backend (Apple GPU)
    mx.set_default_device(mx.gpu)
    
    model_path = "ds4sd/SmolDocling-256M-preview-mlx-bf16"
    model, processor = load(model_path)
    
    # Use more memory-efficient precision
    model.eval()
    mx.eval(model.parameters())
    
    config = load_config(model_path)
    
    print(f"Running on {mx.default_device().type}")  # Verify device
    return model, processor, config
    

def process_document(file_obj, url_input, export_format):
    """Process a document with SmolDocling and return the results."""
    try:
        # Load the model
        model, processor, config = load_model()
        
        # Determine the input source
        if file_obj is not None:
            # Save the uploaded file to a temporary location
            temp_dir = tempfile.mkdtemp()
            
            # Get the file name from the upload
            file_name = getattr(file_obj, 'name', 'uploaded_file')
            
            # Handle different types of file objects that gradio might provide
            temp_path = os.path.join(temp_dir, file_name)
            
            # Different handling based on file object type
            if hasattr(file_obj, 'read'):
                # If it's a file-like object with read method
                with open(temp_path, "wb") as f:
                    f.write(file_obj.read())
            else:
                # If it's already a path (in newer Gradio versions)
                if isinstance(file_obj, str):
                    temp_path = file_obj
                else:
                    # For Gradio's file component that returns tuple (path, name)
                    temp_path = file_obj if isinstance(file_obj, str) else file_obj.name
            
            input_path = temp_path
        elif url_input.strip():
            input_path = url_input.strip()
        else:
            return "Please provide either a file upload or a URL", None, None
        
        # Get images from input file
        images = load_input_resource(input_path)
        if not images:
            return "No images could be extracted from the provided file or URL", None, None
        
        # Set up the prompt
        prompt = "Convert this page to docling."
        formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)
        
        # Process each image and generate output
        all_outputs = []
        all_images = []
        processing_log = ""
        
        for i, image in enumerate(images):
            processing_log += f"Processing page {i+1}/{len(images)}...\n\n"
            processing_log += "DocTags:\n\n"
            
            output = ""
            all_images.append(image)
            
            for token in stream_generate(
                model, processor, formatted_prompt, [image], max_tokens=4096, verbose=False
            ):
                output += token.text
                if "</doctag>" in token.text:
                    break
                
            all_outputs.append(output)
            processing_log += output + "\n\n"
        
        # Create DoclingDocument
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(all_outputs, all_images)
        doc = DoclingDocument(name="ProcessedDocument")
        doc.load_from_doctags(doctags_doc)
        
        # Export based on selected format
        if export_format == "Markdown":
            result = doc.export_to_markdown()
        elif export_format == "HTML":
            html_output = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
            html_path = Path(html_output.name)
            doc.save_as_html(html_path, image_mode=ImageRefMode.EMBEDDED)
            with open(html_path, "r") as f:
                result = f.read()
        elif export_format == "JSON":
            doc_dict = doc.export_to_dict()
            result = json.dumps(doc_dict, indent=4)
        else:
            result = "Invalid export format selected"
            
        # Return the first image as a preview and the processing log
        return result, images[0] if images else None, processing_log
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"Error processing document: {str(e)}\n\nDetails:\n{error_details}", None, error_details

def render_output(result, export_format):
    """Render the processed result based on export format."""
    if export_format == "Markdown":
        # For markdown, show the rendered markdown component.
        return gr.update(value=result, visible=True), gr.update(visible=False), gr.update(visible=False)
    elif export_format == "HTML":
        # For HTML, render as an embedded web component.
        return gr.update(visible=False), gr.update(value=result, visible=True), gr.update(visible=False)
    elif export_format == "JSON":
        # For JSON, parse it into an object so that gr.JSON can render it as an expandable tree.
        try:
            json_obj = json.loads(result)
        except Exception as e:
            json_obj = {"error": "Invalid JSON", "detail": str(e)}
        return gr.update(visible=False), gr.update(visible=False), gr.update(value=json_obj, visible=True)
    else:
        # Fallback: hide all rendered views.
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def prepare_download(result, export_format):
    """Prepare a downloadable file for the processed output."""
    if export_format == "Markdown":
        ext = ".md"
    elif export_format == "HTML":
        ext = ".html"
    elif export_format == "JSON":
        ext = ".json"
    else:
        ext = ".txt"
    # Create a temporary file with the correct file type.
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_file.write(result.encode("utf-8"))
    temp_file.close()
    # Return update objects for the download buttons.
    return gr.update(value=temp_file.name), gr.update(value=temp_file.name)

# Create the Gradio interface
with gr.Blocks(title="SmolDocling Document Processing") as app:
    # Add custom CSS for border styling in the output sections.
    gr.HTML(
        """
        <style>
        #raw_output_box, #formatted_output_box {
            border: 1px solid #ccc;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        """
    )
    
    gr.Markdown("""
    # 📄 SmolDocling Document Processor
    
    Upload a document image or PDF, or provide a URL, to convert it into a structured format using SmolDocling.
    
    SmolDocling is a compact (256MB) Visual Language Model designed for document understanding. It can analyze document layouts,
    identify structural elements, and generate structured representations that preserve the document's semantic meaning.
    """)
    
    lang=None
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload PDF or Image")
            url_input = gr.Textbox(label="Or enter a URL to a PDF or Image")
            export_format = gr.Radio(
                choices=["Markdown", "HTML", "JSON"],
                label="Export Format",
                value="Markdown"
            )
            submit_button = gr.Button("Process Document", variant="primary")
        if export_format == "Markdown":
            lang = "markdown"
        elif export_format == "HTML":
            lang = "html"
        elif export_format == "JSON":
            lang = "json"
        with gr.Column(scale=2):
            with gr.Tab("Raw Output"):
                with gr.Column(elem_id="raw_output_box"):
                    # Display the raw output in a code block.
                    output_text = gr.Code(label="Structured Output", language=lang, lines=20, max_lines=20)
                    download_raw = gr.DownloadButton("Download Raw Output")
            with gr.Tab("Document Preview"):
                preview_image = gr.Image(label="Document Preview", type="pil")
            with gr.Tab("Log"):
                # Display the log in a code block.
                log_output = gr.Code(label="Processing Log", language="html", lines=20, max_lines=20)
            with gr.Tab("Formatted Output"):
                with gr.Column(elem_id="formatted_output_box"):
                    rendered_markdown = gr.Markdown(visible=False, label="Markdown Render")
                    rendered_html = gr.HTML(visible=False, label="HTML Render")
                    rendered_json = gr.JSON(visible=False, label="JSON Render")
                    download_formatted = gr.DownloadButton("Download Formatted Output")
    
    gr.Markdown("""
    ### 💡 What SmolDocling can do:

- Understand full pages of diverse document types: academic papers, business forms, patents, etc.
- Extract:
  - Paragraphs, headers, and footers
  - Tables and their structure (including merged cells, headers)
  - Code blocks with indentation
  - Equations (LaTeX format)
  - Charts and captions
  - Lists and nested lists
- Maintain spatial layout and reading order
- Output results in structured **DocTags**, which are convertible into Markdown, HTML, and JSON.

### 📥 Input Options:
- Upload PDF or image files
- Enter a URL to a remote PDF or image

### 📤 Output Formats:
- **Markdown**: For easy viewing and copy-pasting
- **HTML**: Preserves rich layout for web rendering
- **JSON**: Ideal for developers and downstream automation

**Note**: This app is optimized for Apple Silicon (Metal backend with MLX), but works on any macOS machine with appropriate setup.

---
    """)
    
    # Set up event handlers with chained callbacks:
    submit_button.click(
        process_document,
        inputs=[file_input, url_input, export_format],
        outputs=[output_text, preview_image, log_output]
    ).then(
        render_output,
        inputs=[output_text, export_format],
        outputs=[rendered_markdown, rendered_html, rendered_json]
    ).then(
        prepare_download,
        inputs=[output_text, export_format],
        outputs=[download_raw, download_formatted]
    )

if __name__ == "__main__":
    app.launch()
