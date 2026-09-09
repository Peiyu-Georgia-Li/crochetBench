#!/usr/bin/env python3
"""
Script to automate CrochetPARADE screenshot capture.
Takes a txt file with crochet instructions, loads it into CrochetPARADE,
waits for the rendering to complete, and captures a screenshot of the 
simulated product in the right bottom corner.
"""

import argparse
import time
import os
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains


def capture_crochetparade_screenshot(
    pattern_file: str,
    output_file: str,
    crochetparade_dir: str = None,
    wait_time: int = 10,
    additional_wait: int = 5,
    crop_to_canvas: bool = True,
):
    """
    Captures a screenshot of CrochetPARADE rendering.
    
    Args:
        pattern_file: Path to the txt file containing crochet pattern
        output_file: Path where the screenshot should be saved
        crochetparade_dir: Path to CrochetPARADE directory (default: benchmark_task/CrochetPARADE)
        wait_time: Maximum time to wait for rendering (seconds)
        additional_wait: Additional wait time after rendering completes (seconds)
        crop_to_canvas: If True, crops to just the 3D canvas area
    """
    
    # Set default CrochetPARADE directory if not provided
    if crochetparade_dir is None:
        script_dir = Path(__file__).parent
        crochetparade_dir = script_dir / "benchmark_task" / "CrochetPARADE"
    else:
        crochetparade_dir = Path(crochetparade_dir)
    
    # Validate paths
    pattern_file = Path(pattern_file)
    if not pattern_file.exists():
        raise FileNotFoundError(f"Pattern file not found: {pattern_file}")
    
    index_html = crochetparade_dir / "index.html"
    if not index_html.exists():
        raise FileNotFoundError(f"CrochetPARADE index.html not found: {index_html}")
    
    # Read pattern content
    with open(pattern_file, 'r', encoding='utf-8') as f:
        pattern_text = f.read()
    
    print(f"Loading pattern from: {pattern_file}")
    print(f"CrochetPARADE location: {index_html}")
    
    # Set up Chrome options
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')
    
    # Initialize the driver
    driver = webdriver.Chrome(options=options)
    
    try:
        # Load the CrochetPARADE page
        file_url = f"file://{index_html.absolute()}"
        print(f"Opening: {file_url}")
        driver.get(file_url)
        
        # Wait for page to load
        wait = WebDriverWait(driver, 20)
        
        # Wait for textarea to be present
        print("Waiting for page to load...")
        textarea = wait.until(
            EC.presence_of_element_located((By.ID, "inputText"))
        )
        
        # Clear any existing content and input the pattern
        print("Entering pattern text...")
        textarea.clear()
        textarea.send_keys(pattern_text)
        
        # Trigger the calculation by pressing Shift+Enter
        print("Triggering calculation (Shift+Enter)...")
        ActionChains(driver).key_down(Keys.SHIFT).send_keys(Keys.ENTER).key_up(Keys.SHIFT).perform()
        
        # Wait for the 3D canvas to be ready
        print("Waiting for 3D canvas...")
        canvas = wait.until(
            EC.presence_of_element_located((By.TAG_NAME, "canvas"))
        )
        
        # Wait for rendering to complete
        # We'll wait for a specific time and also check if calculations are done
        print(f"Waiting {wait_time}s for rendering to complete...")
        time.sleep(wait_time)
        
        # Additional wait to ensure rendering is fully complete
        if additional_wait > 0:
            print(f"Additional wait: {additional_wait}s...")
            time.sleep(additional_wait)
        
        # Take screenshot
        print("Capturing screenshot...")
        if crop_to_canvas:
            # Get canvas location and size
            location = canvas.location
            size = canvas.size
            
            # Take full page screenshot first
            driver.save_screenshot('temp_full.png')
            
            # Crop to canvas using PIL
            try:
                from PIL import Image
                
                # Open the screenshot
                image = Image.open('temp_full.png')
                
                # Define the crop area (right bottom corner area)
                # The canvas is typically in the right side of the page
                left = location['x']
                top = location['y']
                right = left + size['width']
                bottom = top + size['height']
                
                # Crop the image
                cropped = image.crop((left, top, right, bottom))
                cropped.save(output_file)
                
                # Clean up temp file
                os.remove('temp_full.png')
                print(f"Screenshot saved (cropped to canvas): {output_file}")
                
            except ImportError:
                print("PIL not available, saving full screenshot instead")
                os.rename('temp_full.png', output_file)
                print(f"Screenshot saved (full page): {output_file}")
        else:
            driver.save_screenshot(output_file)
            print(f"Screenshot saved (full page): {output_file}")
        
    finally:
        # Clean up
        driver.quit()
        print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Capture screenshot of CrochetPARADE rendering"
    )
    parser.add_argument(
        "pattern_file",
        help="Path to the txt file containing crochet pattern (e.g., example_pattern_work.txt)"
    )
    parser.add_argument(
        "-o", "--output",
        default="crochet_screenshot.png",
        help="Output screenshot filename (default: crochet_screenshot.png)"
    )
    parser.add_argument(
        "-d", "--crochetparade-dir",
        default=None,
        help="Path to CrochetPARADE directory (default: benchmark_task/CrochetPARADE)"
    )
    parser.add_argument(
        "-w", "--wait-time",
        type=int,
        default=10,
        help="Time to wait for rendering in seconds (default: 10)"
    )
    parser.add_argument(
        "-a", "--additional-wait",
        type=int,
        default=5,
        help="Additional wait time after rendering (default: 5)"
    )
    parser.add_argument(
        "--full-page",
        action="store_true",
        help="Capture full page instead of cropping to canvas"
    )
    
    args = parser.parse_args()
    
    capture_crochetparade_screenshot(
        pattern_file=args.pattern_file,
        output_file=args.output,
        crochetparade_dir=args.crochetparade_dir,
        wait_time=args.wait_time,
        additional_wait=args.additional_wait,
        crop_to_canvas=not args.full_page,
    )


if __name__ == "__main__":
    main()
