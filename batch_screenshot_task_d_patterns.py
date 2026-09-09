#!/usr/bin/env python3
"""
Batch screenshot generator for all *_full.txt files in task_d_new_*_dsl_patterns directories.
"""

import asyncio
import sys
import os
from pathlib import Path
from capture_crochetparade_playwright import capture_crochetparade_screenshot
import time
import argparse


def find_pattern_files(base_dir: Path):
    """Find all *_full.txt files in task_d_new_*_dsl_patterns directories."""
    pattern_files = []
    
    # Find all task_d_new_*_dsl_patterns directories
    for dsl_dir in base_dir.glob("task_d_new_*_dsl_patterns"):
        if dsl_dir.is_dir():
            # Find all *_full.txt files in this directory
            for pattern_file in dsl_dir.glob("*_full.txt"):
                pattern_files.append(pattern_file)
    
    return sorted(pattern_files)


async def process_pattern_file(
    pattern_file: Path,
    base_dir: Path,
    wait_time: int = 10,
    additional_wait: int = 5,
    skip_existing: bool = True,
    semaphore: asyncio.Semaphore = None,
    file_index: int = 0,
    total_files: int = 0
):
    """Process a single pattern file and generate screenshot."""
    
    # Acquire semaphore if parallel processing
    if semaphore:
        async with semaphore:
            return await _process_pattern_file_impl(
                pattern_file, base_dir, wait_time, additional_wait, 
                skip_existing, file_index, total_files
            )
    else:
        return await _process_pattern_file_impl(
            pattern_file, base_dir, wait_time, additional_wait, 
            skip_existing, file_index, total_files
        )


async def _process_pattern_file_impl(
    pattern_file: Path,
    base_dir: Path,
    wait_time: int,
    additional_wait: int,
    skip_existing: bool,
    file_index: int,
    total_files: int
):
    """Internal implementation of pattern file processing."""
    
    # Generate output filename
    # e.g., task_d_new_qwen_dsl_patterns/Pattern_full.txt -> task_d_new_qwen_img/Pattern_full.png
    dsl_dir_name = pattern_file.parent.name  # e.g., "task_d_new_qwen_dsl_patterns"
    
    # Extract model name from directory name
    # task_d_new_qwen_dsl_patterns -> qwen
    if dsl_dir_name.startswith("task_d_new_") and dsl_dir_name.endswith("_dsl_patterns"):
        model_name = dsl_dir_name.replace("task_d_new_", "").replace("_dsl_patterns", "")
    else:
        model_name = dsl_dir_name.replace("_dsl_patterns", "")
    
    # Create model-specific output directory with task_d_new_<model>_img format
    model_output_dir = base_dir / f"task_d_new_{model_name}_img"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output filename - same as input but with .png extension
    output_file = model_output_dir / pattern_file.with_suffix(".png").name
    
    # Skip if file exists and skip_existing is True
    if skip_existing and output_file.exists():
        if total_files > 0:
            print(f"⏭️  [{file_index}/{total_files}] Skipping {pattern_file.name} (already exists)")
        else:
            print(f"⏭️  Skipping {pattern_file.name} (already exists)")
        return {"status": "skipped", "file": str(pattern_file), "output": str(output_file)}
    
    try:
        if total_files > 0:
            print(f"\n📸 [{file_index}/{total_files}] Processing: {pattern_file.parent.name}/{pattern_file.name}")
        else:
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
        if total_files > 0:
            print(f"✅ [{file_index}/{total_files}] Success! ({elapsed:.1f}s)")
        else:
            print(f"✅ Success! ({elapsed:.1f}s)")
        
        return {"status": "success", "file": str(pattern_file), "output": str(output_file), "time": elapsed}
        
    except Exception as e:
        if total_files > 0:
            print(f"❌ [{file_index}/{total_files}] Error processing {pattern_file.name}: {e}")
        else:
            print(f"❌ Error processing {pattern_file.name}: {e}")
        return {"status": "error", "file": str(pattern_file), "error": str(e)}


async def process_all_patterns(
    base_dir: Path,
    wait_time: int = 10,
    additional_wait: int = 5,
    skip_existing: bool = True,
    limit: int = None,
    parallel: int = 1
):
    """Process all pattern files."""
    
    print("🔍 Finding all *_full.txt files in task_d_new_*_dsl_patterns directories...")
    pattern_files = find_pattern_files(base_dir)
    
    if limit:
        pattern_files = pattern_files[:limit]
    
    # Get list of unique model names for directory creation
    models = set()
    for pf in pattern_files:
        dsl_dir_name = pf.parent.name
        if dsl_dir_name.startswith("task_d_new_") and dsl_dir_name.endswith("_dsl_patterns"):
            model_name = dsl_dir_name.replace("task_d_new_", "").replace("_dsl_patterns", "")
        else:
            model_name = dsl_dir_name.replace("_dsl_patterns", "")
        models.add(model_name)
    
    print(f"📋 Found {len(pattern_files)} pattern files to process")
    print(f"📁 Output directories: {', '.join(sorted([f'task_d_new_{m}_img/' for m in models]))}")
    print(f"⏱️  Wait times: {wait_time}s render + {additional_wait}s additional")
    if parallel > 1:
        print(f"🚀 Parallel processing: {parallel} concurrent tasks")
    if skip_existing:
        print("⏭️  Skipping existing screenshots")
    print()
    
    results = []
    
    if parallel > 1:
        # Parallel processing with semaphore
        semaphore = asyncio.Semaphore(parallel)
        tasks = []
        
        for i, pattern_file in enumerate(pattern_files, 1):
            task = process_pattern_file(
                pattern_file,
                base_dir,
                wait_time,
                additional_wait,
                skip_existing,
                semaphore=semaphore,
                file_index=i,
                total_files=len(pattern_files)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
    else:
        # Sequential processing
        for i, pattern_file in enumerate(pattern_files, 1):
            print(f"\n{'='*80}")
            print(f"[{i}/{len(pattern_files)}] Processing {pattern_file.parent.name}/{pattern_file.name}")
            print(f"{'='*80}")
            
            result = await process_pattern_file(
                pattern_file,
                base_dir,
                wait_time,
                additional_wait,
                skip_existing,
                file_index=i,
                total_files=len(pattern_files)
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
        description="Batch generate screenshots for all *_full.txt files in task_d_new_*_dsl_patterns directories.\n"
                    "Output will be saved to task_d_new_<model>_img/ directories."
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
        "--no-skip",
        action="store_true",
        help="Don't skip existing screenshots (regenerate all)"
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        help="Limit number of files to process (for testing)"
    )
    parser.add_argument(
        "-p", "--parallel",
        type=int,
        default=1,
        help="Number of parallel browser instances (default: 1, sequential)"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    
    asyncio.run(process_all_patterns(
        base_dir=base_dir,
        wait_time=args.wait_time,
        additional_wait=args.additional_wait,
        skip_existing=not args.no_skip,
        limit=args.limit,
        parallel=args.parallel
    ))


if __name__ == "__main__":
    main()
