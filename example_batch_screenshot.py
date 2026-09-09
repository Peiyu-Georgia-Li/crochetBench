#!/usr/bin/env python3
"""
Example script showing how to batch process multiple pattern files
to generate screenshots.
"""

import asyncio
from pathlib import Path
from capture_crochetparade_playwright import capture_crochetparade_screenshot


async def process_single_pattern(pattern_file: Path, output_dir: Path, wait_time: int = 10):
    """Process a single pattern file and save screenshot."""
    output_file = output_dir / f"{pattern_file.stem}.png"
    
    print(f"\n{'='*60}")
    print(f"Processing: {pattern_file.name}")
    print(f"Output: {output_file}")
    print(f"{'='*60}")
    
    try:
        await capture_crochetparade_screenshot(
            pattern_file=str(pattern_file),
            output_file=str(output_file),
            wait_time=wait_time,
            additional_wait=5,
            crop_to_canvas=True,
            headless=True,
        )
        return True
    except Exception as e:
        print(f"Error processing {pattern_file.name}: {e}")
        return False


async def batch_process_patterns(
    pattern_dir: str = "patterns",
    output_dir: str = "screenshots",
    wait_time: int = 10,
    pattern: str = "*.txt"
):
    """
    Batch process all pattern files in a directory.
    
    Args:
        pattern_dir: Directory containing pattern files
        output_dir: Directory to save screenshots
        wait_time: Time to wait for rendering (seconds)
        pattern: Glob pattern for finding pattern files (default: "*.txt")
    """
    pattern_dir = Path(pattern_dir)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all pattern files
    pattern_files = list(pattern_dir.glob(pattern))
    
    if not pattern_files:
        print(f"No pattern files found in {pattern_dir} matching {pattern}")
        return
    
    print(f"Found {len(pattern_files)} pattern files")
    print(f"Output directory: {output_dir}")
    
    # Process each pattern
    success_count = 0
    for pattern_file in pattern_files:
        success = await process_single_pattern(pattern_file, output_dir, wait_time)
        if success:
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Successfully processed: {success_count}/{len(pattern_files)}")
    print(f"{'='*60}")


async def example_single_file():
    """Example: Process a single pattern file."""
    print("Example 1: Processing single file")
    
    await capture_crochetparade_screenshot(
        pattern_file="example_pattern_work.txt",
        output_file="output_single.png",
        wait_time=10,
        additional_wait=5,
    )


async def example_batch_processing():
    """Example: Batch process multiple files."""
    print("Example 2: Batch processing")
    
    # Assuming you have pattern files in a 'patterns' directory
    await batch_process_patterns(
        pattern_dir="data/test_patterns",
        output_dir="screenshots",
        wait_time=15,  # Increase for complex patterns
    )


async def example_custom_settings():
    """Example: Custom settings for a complex pattern."""
    print("Example 3: Custom settings for complex pattern")
    
    await capture_crochetparade_screenshot(
        pattern_file="complex_pattern.txt",
        output_file="complex_output.png",
        wait_time=30,  # Longer wait for complex patterns
        additional_wait=10,
        crop_to_canvas=True,
        headless=True,
    )


if __name__ == "__main__":
    import sys
    
    # You can run different examples by passing arguments
    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "single":
            asyncio.run(example_single_file())
        elif example == "batch":
            asyncio.run(example_batch_processing())
        elif example == "complex":
            asyncio.run(example_custom_settings())
        else:
            print(f"Unknown example: {example}")
            print("Available examples: single, batch, complex")
    else:
        # Default: process the example file
        print("Running default example...")
        print("Usage: python example_batch_screenshot.py [single|batch|complex]")
        print()
        asyncio.run(example_single_file())
