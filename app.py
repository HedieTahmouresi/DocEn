"""Interactive Web GUI Application for AI Document Scanner.

Fulfills Phase 08 Web GUI requirement.
Features:
- Upload raw smartphone photos.
- Display corner detection overlay (TL Red, TR Green, BR Blue, BL Yellow).
- Display perspective-rectified crop.
- Compare original raw image and enhanced clean scan side-by-side.
- 1-Click download of the final clean scan.
"""

import os

# Normalize socks:// proxy schemes for httpx compatibility before importing gradio
for k in ["all_proxy", "ALL_PROXY", "http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"]:
    if k in os.environ and os.environ[k].startswith("socks://"):
        os.environ[k] = os.environ[k].replace("socks://", "socks5://")

import logging
from pathlib import Path
import tempfile
import cv2
import gradio as gr
import numpy as np
from PIL import Image

from src.pipeline.scanner import EndToEndScannerPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global scanner pipeline instance
_pipeline_instance = None


def get_pipeline():
    global _pipeline_instance
    if _pipeline_instance is None:
        logger.info("Initializing End-to-End Scanner Pipeline for Web App...")
        _pipeline_instance = EndToEndScannerPipeline()
    return _pipeline_instance


def process_document(image):
    """Process uploaded raw photo through end-to-end scanner pipeline."""
    if image is None:
        return None, None, None, None

    pipe = get_pipeline()
    results = pipe.scan(image)

    orig_rgb = results["original"]
    overlay_rgb = results["corner_overlay"]
    rectified_rgb = results["rectified"]
    enhanced_rgb = results["enhanced"]

    # Save final enhanced scan to temporary file for 1-click download
    temp_dir = Path(tempfile.gettempdir()) / "doc_scanner_app"
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_file = temp_dir / "clean_document_scan.png"

    bgr_img = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_file), bgr_img)

    return overlay_rgb, rectified_rgb, enhanced_rgb, str(output_file)


# Find sample images if available
sample_images = []
sample_dir = Path("data/real_photos/raw")
if sample_dir.exists():
    sample_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
    sample_images = [str(f) for f in sample_files[:5]]


css_style = """
.container { max-width: 1200px; margin: 0 auto; }
.header { text-align: center; margin-bottom: 20px; }
.output-box { border-radius: 8px; overflow: hidden; }
"""

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 📄 CamScanner AI — Deep Learning Document Scanner
        Upload any raw smartphone photo of a document to automatically detect page corners, 
        rectify perspective distortion, and restore a clean, high-contrast digital scan.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Upload Raw Smartphone Photo",
                type="numpy",
                sources=["upload", "clipboard"],
            )
            scan_button = gr.Button("🚀 Scan Document", variant="primary", size="lg")

            if sample_images:
                gr.Examples(
                    examples=sample_images,
                    inputs=input_image,
                    label="Sample Raw Photos",
                )

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("✨ Enhanced Clean Scan"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Original Raw Photo")
                            raw_preview = gr.Image(label="Original Photo", interactive=False)
                        with gr.Column():
                            gr.Markdown("### Restored Clean Scan")
                            enhanced_preview = gr.Image(label="Enhanced Clean Scan", interactive=False)
                    download_btn = gr.File(label="📥 Download Clean Scan PNG", interactive=False)

                with gr.TabItem("🔍 Intermediate Stages"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 1. Detected Corners Overlay")
                            corner_preview = gr.Image(label="Corner Overlay (TL: Red, TR: Green, BR: Blue, BL: Yellow)", interactive=False)
                        with gr.Column():
                            gr.Markdown("### 2. Perspective Rectified Crop")
                            rectified_preview = gr.Image(label="512x512 Rectified Crop", interactive=False)

    def on_scan(img):
        if img is None:
            return None, None, None, None, None
        overlay, rectified, enhanced, download_path = process_document(img)
        return img, enhanced, overlay, rectified, download_path

    scan_button.click(
        fn=on_scan,
        inputs=[input_image],
        outputs=[
            raw_preview,
            enhanced_preview,
            corner_preview,
            rectified_preview,
            download_btn,
        ],
    )

    input_image.change(
        fn=on_scan,
        inputs=[input_image],
        outputs=[
            raw_preview,
            enhanced_preview,
            corner_preview,
            rectified_preview,
            download_btn,
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=css_style)
