# Batch Screenshot Generator for Valid Patterns

Automatically generate screenshots for all `*valid.txt` files in `*_dsl_patterns` directories.

## Output Structure

Screenshots are organized by model in separate directories:
- `blip_img/` - Screenshots from blip_dsl_patterns
- `claude_img/` - Screenshots from claude_dsl_patterns  
- `dsvl_img/` - Screenshots from dsvl_dsl_patterns
- `gemini_img/` - Screenshots from gemini_dsl_patterns
- `gemma_img/` - Screenshots from gemma_dsl_patterns
- `gpt4v_img/` - Screenshots from gpt4v_dsl_patterns
- `qwen_img/` - Screenshots from qwen_dsl_patterns

Output filenames match the input filenames:
- `claude_dsl_patterns/Pattern_last_valid.txt` → `claude_img/Pattern_last_valid.png`

## Usage

### Basic Usage (Process all files)
```bash
source env/bin/activate
python batch_screenshot_valid_patterns.py
```

Or use the shell wrapper:
```bash
./batch_screenshot_valid.sh
```

### Test First (Process only a few files)
```bash
# Test with first 3 files
python batch_screenshot_valid_patterns.py -l 3
```

### Custom Wait Times
```bash
# Wait 15s for rendering + 8s additional
python batch_screenshot_valid_patterns.py -w 15 -a 8
```

### Regenerate All (Don't skip existing)
```bash
python batch_screenshot_valid_patterns.py --no-skip
```

## Options

- `-w, --wait-time SECONDS` - Time to wait for rendering (default: 10)
- `-a, --additional-wait SECONDS` - Additional wait after rendering (default: 5)
- `-l, --limit N` - Only process first N files (for testing)
- `--no-skip` - Regenerate all screenshots, don't skip existing ones

## Notes

- The script automatically skips files that already have screenshots (use `--no-skip` to regenerate)
- Browser window will appear for each screenshot (WebGL requires visible window on macOS)
- Total time: ~15-20 seconds per screenshot
- For 239 files: approximately 1-1.5 hours total
- Processing happens sequentially to avoid browser conflicts

## Examples

```bash
# Test run with 5 files
python batch_screenshot_valid_patterns.py -l 5

# Process all with faster timing
python batch_screenshot_valid_patterns.py -w 8 -a 3

# Full batch processing
python batch_screenshot_valid_patterns.py
```

## Resuming

If the script is interrupted, simply run it again. It will automatically skip files that were already processed and continue from where it left off.
