#!/usr/bin/env python3
"""
Batch screenshot generator for all *valid.txt files in *_dsl_patterns directories.
"""

import asyncio
import sys
import os
from pathlib import Path
from capture_crochetparade_playwright import capture_crochetparade_screenshot
import time
import argparse


def find_valid_pattern_files(base_dir: Path):
    """Find all *valid.txt files in *_dsl_patterns directories."""
    valid_files = []
    
    # Find all *_dsl_patterns directories
    for dsl_dir in base_dir.glob("*_dsl_patterns"):
        if dsl_dir.is_dir():
            # Find all *valid.txt files in this directory
            for pattern_file in dsl_dir.glob("*valid.txt"):
                valid_files.append(pattern_file)
    
    return sorted(valid_files)


async def process_pattern_file(
    pattern_file: Path,
    base_dir: Path,
    wait_time: int = 10,
    additional_wait: int = 5,
    skip_existing: bool = True
):
    """Process a single pattern file and generate screenshot."""
    
    # Generate output filename
    # e.g., claude_dsl_patterns/Pattern_last_valid.txt -> claude_img/Pattern_last_valid.png
    dsl_dir_name = pattern_file.parent.name  # e.g., "claude_dsl_patterns"
    model_name = dsl_dir_name.replace("_dsl_patterns", "")  # e.g., "claude"
    
    # Create model-specific output directory with _img suffix
    model_output_dir = base_dir / f"{model_name}_img"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output filename - same as input but with .png extension
    output_file = model_output_dir / pattern_file.with_suffix(".png").name
    
    # Skip if file exists and skip_existing is True
    if skip_existing and output_file.exists():
        print(f"⏭️  Skipping {pattern_file.name} (already exists)")
        return {"status": "skipped", "file": str(pattern_file), "output": str(output_file)}
    
    try:
        print(f"\n📸 Processing: {pattern_file.parent.name}/{pattern_file.name}")
        print(f"   Output: {output_file}")
        
        start_time = time.time()
        
        await capture_crochetparade_screenshot(
            pattern_file=str(pattern_file),
            output_file=str(output_file),
            wait_time=wait_time,
            additional_wait=additional_wait,
            crop_to_canvas=True,
            headless=False
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Success! ({elapsed:.1f}s)")
        
        return {"status": "success", "file": str(pattern_file), "output": str(output_file), "time": elapsed}
        
    except Exception as e:
        print(f"❌ Error processing {pattern_file.name}: {e}")
        return {"status": "error", "file": str(pattern_file), "error": str(e)}


async def process_all_patterns(
    base_dir: Path,
    wait_time: int = 10,
    additional_wait: int = 5,
    skip_existing: bool = True,
    limit: int = None
):
    """Process all pattern files."""
    
    print("🔍 Finding all *valid.txt files in *_dsl_patterns directories...")
    pattern_files = find_valid_pattern_files(base_dir)
    
    if limit:
        pattern_files = pattern_files[:limit]
    
    # Get list of unique model names for directory creation
    models = set()
    for pf in pattern_files:
        model_name = pf.parent.name.replace("_dsl_patterns", "")
        models.add(model_name)
    
    print(f"📋 Found {len(pattern_files)} pattern files to process")
    print(f"📁 Output directories: {', '.join(sorted([f'{m}_img/' for m in models]))}")
    print(f"⏱️  Wait times: {wait_time}s render + {additional_wait}s additional")
    if skip_existing:
        print("⏭️  Skipping existing screenshots")
    print()
    
    results = []
    
    for i, pattern_file in enumerate(pattern_files, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(pattern_files)}] Processing {pattern_file.parent.name}/{pattern_file.name}")
        print(f"{'='*80}")
        
        result = await process_pattern_file(
            pattern_file,
            base_dir,
            wait_time,
            additional_wait,
            skip_existing
        )
        results.append(result)
        
        # Small delay between files to avoid issues
        await asyncio.sleep(1)
    
    # Print summary
    print(f"\n\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}")
    
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    
    print(f"✅ Successful: {success_count}")
    print(f"⏭️  Skipped: {skipped_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📁 Total: {len(results)}")
    
    if error_count > 0:
        print(f"\n❌ Failed files:")
        for result in results:
            if result["status"] == "error":
                print(f"   - {Path(result['file']).name}: {result['error']}")
    
    total_time = sum(r.get("time", 0) for r in results)
    print(f"\n⏱️  Total processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate screenshots for all *valid.txt files in *_dsl_patterns directories.\n"
                    "Output will be saved to <model>_img/ directories (e.g., claude_img/, blip_img/, etc.)"
    )
    parser.add_argument(
        "-w", "--wait-time",
        type=int,
        default=300,
        help="Time to wait for rendering in seconds (default: 10)"
    )
    parser.add_argument(
        "-a", "--additional-wait",
        type=int,
        default=5,
        help="Additional wait time after rendering (default: 5)"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip existing screenshots (regenerate all)"
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        help="Limit number of files to process (for testing)"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    
    asyncio.run(process_all_patterns(
        base_dir=base_dir,
        wait_time=args.wait_time,
        additional_wait=args.additional_wait,
        skip_existing=not args.no_skip,
        limit=args.limit
    ))


if __name__ == "__main__":
    main()
