import requests
from bs4 import BeautifulSoup
import time
import os
import re
from urllib.parse import urljoin, urlparse

class PaulGrahamScraper:
    def __init__(self):
        self.base_url = "http://www.paulgraham.com/"
        self.essays_dir = "pg_essays"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def create_essays_directory(self):
        """Create directory to store essays if it doesn't exist"""
        if not os.path.exists(self.essays_dir):
            os.makedirs(self.essays_dir)
            
    def get_essay_links(self):
        """Scrape the main page to get all essay links"""
        try:
            response = self.session.get(self.base_url + "articles.html")
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links that point to essays
            essay_links = []
            
            # Look for links in the main content area
            # PG's site structure: essays are typically linked from articles.html
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Filter for essay links (typically .html files, exclude external links)
                if (href.endswith('.html') and 
                    not href.startswith('http') and 
                    href != 'articles.html' and
                    not href.startswith('#')):
                    
                    full_url = urljoin(self.base_url, href)
                    title = link.get_text().strip()
                    essay_links.append({
                        'url': full_url,
                        'filename': href,
                        'title': title
                    })
            
            return essay_links
            
        except requests.RequestException as e:
            print(f"Error fetching essay links: {e}")
            return []
    
    def clean_filename(self, filename):
        """Clean filename to be filesystem-safe"""
        # Remove or replace problematic characters
        cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
        return cleaned[:100]  # Limit length
    
    def scrape_essay(self, essay_info):
        """Scrape a single essay and save it"""
        try:
            print(f"Scraping: {essay_info['title']}")
            
            response = self.session.get(essay_info['url'])
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract the main content
            # PG's essays are typically in a table structure or simple HTML
            # We'll try to get the main text content
            
            # Method 1: Look for the main content (common patterns on PG's site)
            content = None
            
            # Try to find content in common containers
            possible_containers = [
                soup.find('table'),  # Many PG essays use table layouts
                soup.find('body'),
                soup
            ]
            
            for container in possible_containers:
                if container:
                    # Remove script and style elements
                    for script in container(["script", "style"]):
                        script.decompose()
                    
                    # Get text content
                    text = container.get_text()
                    
                    # Clean up the text
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    content = ' '.join(chunk for chunk in chunks if chunk)
                    
                    if len(content) > 500:  # Ensure we got substantial content
                        break
            
            if content:
                # Create filename
                safe_title = self.clean_filename(essay_info['title'])
                filename = f"{safe_title}.txt"
                filepath = os.path.join(self.essays_dir, filename)
                
                # Save the essay
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"Title: {essay_info['title']}\n")
                    f.write(f"URL: {essay_info['url']}\n")
                    f.write(f"{'='*50}\n\n")
                    f.write(content)
                
                print(f"Saved: {filename}")
                return True
            else:
                print(f"No content found for: {essay_info['title']}")
                return False
                
        except requests.RequestException as e:
            print(f"Error scraping {essay_info['title']}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error scraping {essay_info['title']}: {e}")
            return False
    
    def scrape_all_essays(self, delay=1):
        """Scrape all essays with a delay between requests"""
        self.create_essays_directory()
        
        print("Getting essay links...")
        essay_links = self.get_essay_links()
        
        if not essay_links:
            print("No essay links found. Check the scraping logic.")
            return
        
        print(f"Found {len(essay_links)} essays to scrape")
        
        successful = 0
        for i, essay in enumerate(essay_links, 1):
            print(f"\nProgress: {i}/{len(essay_links)}")
            
            if self.scrape_essay(essay):
                successful += 1
            
            # Be respectful - add delay between requests
            if i < len(essay_links):
                time.sleep(delay)
        
        print(f"\nScraping complete! Successfully scraped {successful}/{len(essay_links)} essays")
        print(f"Essays saved in '{self.essays_dir}' directory")

# Usage example
if __name__ == "__main__":
    scraper = PaulGrahamScraper()
    
    # Option 1: Scrape all essays
    scraper.scrape_all_essays(delay=2)  # 2 second delay between requests
    
    # Option 2: Just get the list of essays first to see what's available
    # links = scraper.get_essay_links()
    # for link in links[:5]:  # Show first 5
    #     print(f"Title: {link['title']}")
    #     print(f"URL: {link['url']}")
    #     print("---")