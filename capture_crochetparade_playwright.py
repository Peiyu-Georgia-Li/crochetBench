#!/usr/bin/env python3
"""
Script to automate CrochetPARADE screenshot capture using Playwright.
Takes a txt file with crochet instructions, loads it into CrochetPARADE,
waits for the rendering to complete, and captures a screenshot of the 
simulated product in the right bottom corner.

This version uses Playwright which may be more reliable than Selenium.
"""

import argparse
import asyncio
import time
from pathlib import Path
from playwright.async_api import async_playwright
import http.server
import socketserver
import threading


async def capture_crochetparade_screenshot(
    pattern_file: str,
    output_file: str,
    crochetparade_dir: str = None,
    wait_time: int = 10,
    additional_wait: int = 10,
    crop_to_canvas: bool = True,
    headless: bool = False,  # WebGL doesn't render properly in headless mode on macOS
):
    """
    Captures a screenshot of CrochetPARADE rendering using Playwright.
    
    Args:
        pattern_file: Path to the txt file containing crochet pattern
        output_file: Path where the screenshot should be saved
        crochetparade_dir: Path to CrochetPARADE directory (default: benchmark_task/CrochetPARADE)
        wait_time: Maximum time to wait for rendering (seconds)
        additional_wait: Additional wait time after rendering completes (seconds)
        crop_to_canvas: If True, crop screenshot to canvas area only
        headless: Run in headless mode (may not work on macOS)
    """
    # Convert output_file to absolute path
    output_file = str(Path(output_file).resolve())
    print(f"Output will be saved to: {output_file}")
    
    if crochetparade_dir is None:
        # Default to benchmark_task/CrochetPARADE relative to this script
        script_dir = Path(__file__).parent
        crochetparade_dir = script_dir / "benchmark_task" / "CrochetPARADE"
    else:
        crochetparade_dir = Path(crochetparade_dir)
    
    crochetparade_dir = crochetparade_dir.resolve()
    index_html = crochetparade_dir / "index.html"
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
    
    # Start a local HTTP server to avoid CORS issues
    # Use a random available port to avoid conflicts
    import random
    port = random.randint(8000, 9000)
    Handler = http.server.SimpleHTTPRequestHandler
    
    # Change to CrochetPARADE directory for serving
    import os
    original_dir = os.getcwd()
    os.chdir(crochetparade_dir)
    
    httpd = socketserver.TCPServer(("", port), Handler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    print(f"Started local HTTP server on port {port}")
    
    try:
        await asyncio.sleep(0.5)  # Give server time to start
    except:
        pass
    
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()
        
        # Set up console logging to capture errors and track messages
        console_messages = []
        def log_console(msg):
            console_messages.append((time.time(), msg.text, msg.type))
            print(f"Browser console [{msg.type}]: {msg.text}")
        
        page.on("console", log_console)
        page.on("pageerror", lambda err: print(f"Browser error: {err}"))
        
        try:
            # Load the CrochetPARADE page via HTTP server
            http_url = f"http://localhost:{port}/index.html"
            print(f"Opening: {http_url}")
            await page.goto(http_url)
            
            # Wait for textarea to be present
            print("Waiting for page to load...")
            await page.wait_for_selector("#inputText", timeout=20000)
            
            # Clear any existing content and input the pattern
            print("Entering pattern text...")
            textarea = await page.query_selector("#inputText")
            await textarea.fill(pattern_text)
            
            # Wait a moment for the text to be processed
            await asyncio.sleep(1)
            
            # Check if button exists and is visible
            print("Looking for Calculate button...")
            calc_button = await page.query_selector('button[id="3dbutton"]')
            if calc_button:
                is_visible = await calc_button.is_visible()
                print(f"Calculate button found, visible: {is_visible}")
                print("Clicking Calculate button...")
                await calc_button.click()
                print("Button clicked successfully")
            else:
                print("Warning: Calculate button not found, trying Shift+Enter...")
                await textarea.press("Shift+Enter")
            
            # Give a moment for the calculation to start
            await asyncio.sleep(2)
            
            # Wait for the 3D canvas to be ready (look for the specific 3D render canvas)
            # Use attribute selector because the ID starts with a number
            print("Waiting for 3D canvas to be created...")
            print("(This may take a while for complex patterns)")
            
            # Check all canvas elements to debug
            all_canvases = await page.query_selector_all('canvas')
            print(f"Found {len(all_canvases)} canvas elements on page")
            
            canvas_found = False
            try:
                await page.wait_for_selector('canvas[id="3DRender"]', timeout=60000, state='attached')
                print("3D canvas found!")
                canvas_found = True
            except Exception as e:
                print(f"Warning: Could not find 3D canvas after 60s: {e}")
                # List all canvas IDs for debugging
                canvas_ids = await page.evaluate('''() => {
                    const canvases = document.querySelectorAll('canvas');
                    return Array.from(canvases).map(c => c.id || 'no-id');
                }''')
                print(f"Canvas IDs found: {canvas_ids}")
                print("Attempting to continue anyway...")
            
            # If canvas was found, wait for WebGL rendering to complete
            # The canvas uses THREE.js (WebGL), so we need to wait for the rendering iterations
            if canvas_found:
                print("Waiting for 3D rendering calculations to complete...")
                print("(Monitoring console logs to detect when calculation finishes)")
                
                # Monitor for calculation completion (no new logs for a period)
                max_wait_iterations = 60  # 60 * 2 = 120 seconds max
                stable_count = 0
                
                for i in range(max_wait_iterations):
                    initial_count = len(console_messages)
                    await asyncio.sleep(2)
                    
                    # Check if log count has stabilized (no new messages)
                    if len(console_messages) == initial_count:
                        stable_count += 1
                        if stable_count >= 3:  # No logs for 6 seconds
                            print(f"Calculation appears complete (no logs for {stable_count*2}s)")
                            break
                    else:
                        stable_count = 0
                        if i % 5 == 0:  # Print progress every 10 seconds
                            print(f"Still calculating... ({i*2}s elapsed)")
                else:
                    print(f"Warning: Max wait time reached ({max_wait_iterations*2}s)")
                
                print("Calculation complete, waiting for rendering to finalize...")
                
                # Wait for rendering to complete
                print(f"Waiting {wait_time}s for 3D rendering to display...")
                await asyncio.sleep(wait_time)
                
                # Additional wait to ensure rendering is fully complete
                if additional_wait > 0:
                    print(f"Additional wait: {additional_wait}s...")
                    await asyncio.sleep(additional_wait)
                
                # Force rendering frames to complete and trigger explicit render
                print("Finalizing rendering...")
                await page.evaluate('''async () => {
                    // Force several animation frames to ensure rendering is complete
                    for (let i = 0; i < 10; i++) {
                        await new Promise(resolve => requestAnimationFrame(resolve));
                    }
                    
                    // Try to access the renderer and force a render if available
                    if (window.renderer && window.scene && window.camera) {
                        window.renderer.render(window.scene, window.camera);
                    }
                }''')
                await asyncio.sleep(2)
            
            # Capture screenshot using Playwright's viewport screenshot
            # This captures what's actually visible on screen (including WebGL rendering)
            print("Capturing screenshot from visible browser window...")
            
            if crop_to_canvas:
                # Get canvas bounding box to crop viewport screenshot
                canvas = await page.query_selector('canvas[id="3DRender"]')
                if canvas:
                    bbox = await canvas.bounding_box()
                    if bbox:
                        print(f"Canvas found at position ({bbox['x']}, {bbox['y']}) size {bbox['width']}x{bbox['height']}")
                        await page.screenshot(path=output_file, clip={
                            'x': bbox['x'],
                            'y': bbox['y'],
                            'width': bbox['width'],
                            'height': bbox['height']
                        })
                        print(f"Screenshot saved (canvas area): {output_file}")
                    else:
                        print("Could not get canvas bounding box, taking full viewport")
                        await page.screenshot(path=output_file, full_page=False)
                        print(f"Screenshot saved (full viewport): {output_file}")
                else:
                    print("Canvas not found, taking full viewport")
                    await page.screenshot(path=output_file, full_page=False)
                    print(f"Screenshot saved (full viewport): {output_file}")
            else:
                await page.screenshot(path=output_file, full_page=True)
                print(f"Screenshot saved (full page): {output_file}")
            
        finally:
            # Clean up
            await browser.close()
            httpd.shutdown()
            os.chdir(original_dir)
            print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Capture screenshot of CrochetPARADE rendering using Playwright"
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
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (invisible, may not work on macOS)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(capture_crochetparade_screenshot(
        pattern_file=args.pattern_file,
        output_file=args.output,
        crochetparade_dir=args.crochetparade_dir,
        wait_time=args.wait_time,
        additional_wait=args.additional_wait,
        crop_to_canvas=not args.full_page,
        headless=args.headless,  # Default is False (visible)
    ))


if __name__ == "__main__":
    main()
