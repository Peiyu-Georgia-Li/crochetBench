import argparse
from data_scraping import download_yarnspirations

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="A CLI tool for data scraping")

    # Add subcommands: 'scrape' for scraping, 'upload' for uploading
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand for scraping data
    scrape_parser = subparsers.add_parser("scrape", help="Scrape data from Yarnspirations")
    scrape_parser.add_argument("--project-type", type=str, required=True, help="Project type (e.g., Rugs, Scarves, Blankets)")
    scrape_parser.add_argument("--pages", type=int, required=True, help="Number of pages to scrape")

    # Parse arguments
    args = parser.parse_args()

    # Handle 'scrape' command
    if args.command == "scrape":
        print(f"Scraping {args.pages} pages of {args.project_type} from Yarnspirations...")
        download_yarnspirations(args.project_type, args.pages)

if __name__ == "__main__":
    main()

# Example usage:
# python cli.py scrape --project-type "Rugs" --pages 3