# Example usage:
# python cli.py --project-type "Hats" --pages 3
import argparse
from data_scraping import download_yarnspirations

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="A CLI tool for data scraping")
    scrape_parser.add_argument("--project-type", type=str, required=True, help="Project type (e.g., Rugs, Scarves, Blankets)")
    scrape_parser.add_argument("--pages", type=int, required=True, help="Number of pages to scrape")

    # Parse arguments
    args = parser.parse_args()

    print(f"Scraping {args.pages} pages of {args.project_type} from Yarnspirations...")
    download_yarnspirations(args.project_type, args.pages)

if __name__ == "__main__":
    main()

