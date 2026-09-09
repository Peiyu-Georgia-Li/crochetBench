# Quick Start: CrochetPARADE Screenshot Automation

This guide will help you quickly get started with automatically capturing screenshots from CrochetPARADE.

## Installation (One-time Setup)

```bash
# 1. Install Python dependencies
pip install playwright

# 2. Install Chromium browser for Playwright
playwright install chromium
```

That's it! You're ready to go.

## Basic Usage

### Option 1: Shell Script (Easiest)

```bash
./screenshot_crochet.sh example_pattern_work.txt
```

This creates `crochet_screenshot.png` with the rendered pattern.

### Option 2: Python Script

```bash
python capture_crochetparade_playwright.py example_pattern_work.txt
```

### Option 3: Command with Custom Output

```bash
./screenshot_crochet.sh example_pattern_work.txt -o my_output.png -w 15
```

## Common Use Cases

### 1. Simple Screenshot
```bash
./screenshot_crochet.sh pattern.txt
```

### 2. Complex Pattern (needs more time)
```bash
./screenshot_crochet.sh complex_pattern.txt -w 20 -a 10
```

### 3. Batch Process Multiple Files
```bash
# Create output directory
mkdir -p screenshots

# Process all txt files
for file in patterns/*.txt; do
    output="screenshots/$(basename "$file" .txt).png"
    ./screenshot_crochet.sh "$file" -o "$output"
done
```

### 4. Debug Mode (see what's happening)
```bash
./screenshot_crochet.sh pattern.txt -v --full-page
```

## Parameters Explained

- **`-w, --wait TIME`**: How long to wait for rendering (default: 10 seconds)
  - Use 15-30 for complex patterns
  - Use 5-10 for simple patterns
  
- **`-a, --additional TIME`**: Extra wait after rendering (default: 5 seconds)
  - Ensures rendering is fully complete
  
- **`-o, --output FILE`**: Where to save the screenshot
  
- **`--full-page`**: Capture entire window instead of just the canvas
  
- **`-v, --visible`**: Show the browser (for debugging)

## Troubleshooting

### "Module not found: playwright"
```bash
pip install playwright
playwright install chromium
```

### Empty or Black Screenshot
Increase wait time:
```bash
./screenshot_crochet.sh pattern.txt -w 20 -a 10
```

### Pattern Not Rendering
Check your pattern syntax or run in visible mode:
```bash
./screenshot_crochet.sh pattern.txt -v
```

## What Gets Captured?

By default, the script captures only the **3D canvas area** (right bottom corner) showing the rendered crochet pattern. This is the main visualization area where your pattern appears.

To capture the full interface including all controls:
```bash
./screenshot_crochet.sh pattern.txt --full-page
```

## File Structure

```
crochetBench-main/
├── screenshot_crochet.sh              # Easy shell script wrapper
├── capture_crochetparade_playwright.py # Main Python script (recommended)
├── capture_crochetparade_screenshot.py # Selenium alternative
├── example_batch_screenshot.py         # Batch processing examples
├── requirements_screenshot.txt         # Python dependencies
├── QUICKSTART_SCREENSHOT.md           # This file
├── README_SCREENSHOT.md               # Detailed documentation
└── benchmark_task/
    └── CrochetPARADE/
        └── index.html                 # The CrochetPARADE application
```

## Next Steps

- Read [README_SCREENSHOT.md](README_SCREENSHOT.md) for detailed documentation
- Check [example_batch_screenshot.py](example_batch_screenshot.py) for programmatic usage
- Experiment with different wait times for your patterns

## Example Output

When you run:
```bash
./screenshot_crochet.sh example_pattern_work.txt
```

You'll see:
```
Capturing screenshot from CrochetPARADE...
Loading pattern from: example_pattern_work.txt
CrochetPARADE location: .../CrochetPARADE/index.html
Opening: file:///.../index.html
Waiting for page to load...
Entering pattern text...
Triggering calculation (Shift+Enter)...
Waiting for 3D canvas...
Waiting 10s for rendering to complete...
Additional wait: 5s...
Capturing screenshot...
Screenshot saved (canvas only): crochet_screenshot.png
Done!
Success! Screenshot saved to: crochet_screenshot.png
```

And you'll have a PNG file with the rendered 3D crochet pattern!
