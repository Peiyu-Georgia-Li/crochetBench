# CrochetPARADE Screenshot Automation

This directory contains scripts to automate capturing screenshots from CrochetPARADE, a crochet pattern renderer.

## Scripts

Two implementations are provided:

1. **`capture_crochetparade_playwright.py`** (Recommended)
   - Uses Playwright for browser automation
   - More reliable and easier to set up
   - Better screenshot capabilities

2. **`capture_crochetparade_screenshot.py`**
   - Uses Selenium for browser automation
   - Alternative if you prefer Selenium

## Installation

### For Playwright (Recommended)

```bash
# Install Python dependencies
pip install -r requirements_screenshot.txt

# Install Playwright browsers
playwright install chromium
```

### For Selenium

```bash
# Install Python dependencies
pip install selenium Pillow

# Install ChromeDriver
# macOS with Homebrew:
brew install chromedriver

# Or download from: https://chromedriver.chromium.org/
```

## Usage

### Basic Usage (Playwright)

```bash
python capture_crochetparade_playwright.py example_pattern_work.txt
```

This will:
1. Load the pattern from `example_pattern_work.txt`
2. Open CrochetPARADE in the browser
3. Input the pattern text
4. Trigger the rendering calculation (Shift+Enter)
5. Wait for rendering to complete
6. Capture a screenshot of the 3D canvas
7. Save it as `crochet_screenshot.png`

### Advanced Options

```bash
# Specify output filename
python capture_crochetparade_playwright.py example_pattern_work.txt -o output.png

# Increase wait time for complex patterns (in seconds)
python capture_crochetparade_playwright.py example_pattern_work.txt -w 20 -a 10

# Capture full page instead of just the canvas
python capture_crochetparade_playwright.py example_pattern_work.txt --full-page

# Run with visible browser (for debugging)
python capture_crochetparade_playwright.py example_pattern_work.txt --no-headless

# Specify CrochetPARADE directory
python capture_crochetparade_playwright.py example_pattern_work.txt -d /path/to/CrochetPARADE
```

### Using Selenium Version

The Selenium version has the same command-line arguments:

```bash
python capture_crochetparade_screenshot.py example_pattern_work.txt -o output.png
```

## Command-Line Arguments

- `pattern_file`: Path to the txt file containing crochet pattern (required)
- `-o, --output`: Output screenshot filename (default: `crochet_screenshot.png`)
- `-d, --crochetparade-dir`: Path to CrochetPARADE directory (default: `benchmark_task/CrochetPARADE`)
- `-w, --wait-time`: Time to wait for rendering in seconds (default: 10)
- `-a, --additional-wait`: Additional wait time after rendering (default: 5)
- `--full-page`: Capture full page instead of cropping to canvas
- `--no-headless`: Run browser in visible mode (Playwright only)

## Batch Processing

To process multiple pattern files:

```bash
# Using bash
for file in patterns/*.txt; do
    output="screenshots/$(basename "$file" .txt).png"
    python capture_crochetparade_playwright.py "$file" -o "$output" -w 15
done
```

Or create a Python script:

```python
from pathlib import Path
import asyncio
from capture_crochetparade_playwright import capture_crochetparade_screenshot

async def batch_process():
    pattern_dir = Path("patterns")
    output_dir = Path("screenshots")
    output_dir.mkdir(exist_ok=True)
    
    for pattern_file in pattern_dir.glob("*.txt"):
        output_file = output_dir / f"{pattern_file.stem}.png"
        print(f"Processing {pattern_file.name}...")
        await capture_crochetparade_screenshot(
            pattern_file=str(pattern_file),
            output_file=str(output_file),
            wait_time=15,
        )

if __name__ == "__main__":
    asyncio.run(batch_process())
```

## Troubleshooting

### Pattern not rendering
- Increase wait time: `-w 20 -a 10`
- Check pattern syntax in CrochetPARADE manual
- Run with `--no-headless` to see what's happening

### Canvas not found
- Ensure CrochetPARADE directory path is correct
- Check that all JavaScript files are present in the CrochetPARADE directory

### Empty or black screenshot
- Increase additional wait time: `-a 10`
- The pattern might take longer to render

### Browser errors
- For Playwright: Run `playwright install chromium` again
- For Selenium: Ensure ChromeDriver version matches your Chrome version

## How It Works

1. **Load Pattern**: Reads the pattern text from the input file
2. **Open Browser**: Launches a headless Chromium browser
3. **Load CrochetPARADE**: Opens the `index.html` file from the CrochetPARADE directory
4. **Input Pattern**: Fills the textarea with the pattern text
5. **Trigger Calculation**: Simulates pressing Shift+Enter to start the rendering
6. **Wait for Rendering**: Waits for the specified time for the 3D model to generate
7. **Capture Screenshot**: Takes a screenshot of the canvas element (or full page)
8. **Save**: Saves the screenshot to the specified output file

## Notes

- The default wait time (10s + 5s) works for most small to medium patterns
- Complex patterns with many stitches may require longer wait times
- The script captures the right bottom corner (canvas area) by default
- Use `--full-page` to capture the entire interface including controls
- Both scripts default to `headless=False` because CrochetPARADE's WebGL canvas has not
  rendered reliably for us in headless Chromium on macOS. On Linux, headless mode with a
  software GL fallback (e.g. `--headless=new` plus `--use-gl=swiftshader`, or running under
  `xvfb-run`) has been more reliable for batch jobs — verify a few renders by eye before
  trusting a large unattended batch.

## Computing similarity to the reference image

Once you have rendered screenshots for a set of patterns, `calculate_dino_similarity.py`
(at the repo root) compares each rendered image against its reference product photo using
DINO (ViT-Small, CLS-token cosine similarity):

```bash
python calculate_dino_similarity.py \
    --patterns-json split_dataset/test.json \
    --base-dir /path/to/valid_<model>_img_dirs \
    --models qwen claude gemini
```

`batch_compare_images.py` computes the same DINO score alongside LPIPS, SSIM, a Wasserstein
color-distribution distance, and CLIP similarity, and writes one CSV per model.
