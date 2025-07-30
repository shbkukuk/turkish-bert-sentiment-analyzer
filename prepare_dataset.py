import requests
from bs4 import BeautifulSoup
import json
import csv
import time
import re
from urllib.parse import urljoin, urlparse
import logging
import sqlite3

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SeturComplaintScraper:
    def __init__(self):
        self.base_url = "https://www.sikayetvar.com"
        self.setur_url = "https://www.sikayetvar.com/setur-turizm"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'tr-TR,tr;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.complaints_data = []

    def get_page_content(self, url, max_retries=3):
        """Get page content with retry mechanism"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
                    return None

    def extract_complaint_details(self, complaint_url):
        """Extract detailed information from individual complaint page"""
        response = self.get_page_content(complaint_url)
        if not response:
            return {}
        
        soup = BeautifulSoup(response.content, 'html.parser')
        details = {}
        
        try:
            # Extract rating if exists
            rating_element = soup.find('div', class_='rating') or soup.find('span', class_='rating')
            if rating_element:
                stars = rating_element.find_all('i', class_='star-full') or rating_element.find_all('span', class_='star')
                details['rating'] = len(stars) if stars else None            # Extract full complaint text - updated selector for complaint detail
            complaint_detail = soup.find('div', class_='complaint-detail-description')
            if complaint_detail:
                # Get all paragraphs within the complaint detail
                paragraphs = complaint_detail.find_all('p')
                if paragraphs:
                    complaint_text = '\n'.join([p.get_text(strip=True) for p in paragraphs])
                    details['full_complaint'] = complaint_text
                else:
                    details['full_complaint'] = complaint_detail.get_text(strip=True)
            else:
                # Fallback to original selectors
                complaint_text = soup.find('div', class_='complaint-text') or soup.find('div', class_='complaint-content')
                if complaint_text:
                    details['full_complaint'] = complaint_text.get_text(strip=True)
            
            # Extract view count from detail page
            view_element = soup.find('span', class_='js-view-count count js-increment-view')
            if view_element:
                details['views_detail'] = view_element.get_text(strip=True)
            
            # Extract company response - updated selector for complaint reply
            response_wrapper = soup.find('div', class_='complaint-reply-wrapper ga-v ga-c')
            if response_wrapper:
                response_message = response_wrapper.find('p', class_='message')
                if response_message:
                    details['company_response'] = response_message.get_text(strip=True)
            else:
                # Fallback to original selectors
                response_section = soup.find('div', class_='company-response') or soup.find('section', class_='response')
                if response_section:
                    details['company_response'] = response_section.get_text(strip=True)
            
            # Extract comments and replies
            comments = []
            comment_sections = soup.find_all('div', class_='comment') or soup.find_all('article', class_='comment')
            
            for comment in comment_sections:
                comment_data = {}
                
                # Comment author
                author = comment.find('a', class_='username') or comment.find('span', class_='comment-author')
                if author:
                    comment_data['author'] = author.get_text(strip=True)
                
                # Comment text
                comment_text = comment.find('div', class_='comment-text') or comment.find('p', class_='comment-content')
                if comment_text:
                    comment_data['text'] = comment_text.get_text(strip=True)
                
                # Comment time
                comment_time = comment.find('time') or comment.find('span', class_='time')
                if comment_time:
                    comment_data['time'] = comment_time.get_text(strip=True)
                
                # Replies to this comment
                replies = []
                reply_elements = comment.find_all('div', class_='reply') or comment.find_all('div', class_='comment-reply')
                for reply in reply_elements:
                    reply_data = {}
                    reply_author = reply.find('a', class_='username') or reply.find('span', class_='reply-author')
                    if reply_author:
                        reply_data['author'] = reply_author.get_text(strip=True)
                    
                    reply_text = reply.find('div', class_='reply-text') or reply.find('p', class_='reply-content')
                    if reply_text:
                        reply_data['text'] = reply_text.get_text(strip=True)
                    
                    reply_time = reply.find('time') or reply.find('span', class_='time')
                    if reply_time:
                        reply_data['time'] = reply_time.get_text(strip=True)
                    
                    replies.append(reply_data)
                
                comment_data['replies'] = replies
                comments.append(comment_data)
            
            details['comments'] = comments
            
        except Exception as e:
            logger.error(f"Error extracting details from {complaint_url}: {e}")
        
        return details

    def extract_complaints_from_page(self, page_url):
        """Extract complaint data from a single page"""
        response = self.get_page_content(page_url)
        if not response:
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        complaints = []
        
        # Find complaint cards
        complaint_cards = soup.find_all('article', class_='card-v2') or soup.find_all('div', class_='complaint-card')
        
        logger.info(f"Found {len(complaint_cards)} complaints on page: {page_url}")
        
        for card in complaint_cards:
            try:
                complaint_data = {}
                
                # Extract complaint ID
                complaint_id = card.get('data-id')
                if complaint_id:
                    complaint_data['id'] = complaint_id
                
                # Extract complaint title and URL
                title_link = card.find('a', class_='complaint-layer') or card.find('h2').find('a') if card.find('h2') else None
                if title_link:
                    complaint_data['title'] = title_link.get_text(strip=True)
                    complaint_data['url'] = urljoin(self.base_url, title_link.get('href'))
                
                # Extract user information
                profile_section = card.find('header', class_='profile-details') or card.find('div', class_='profile')
                if profile_section:
                    # Username
                    username_link = profile_section.find('a', class_='username')
                    if username_link:
                        complaint_data['username'] = username_link.get_text(strip=True)
                    
                    # Post time
                    time_element = profile_section.find('div', class_='js-tooltip time') or profile_section.find('time')
                    if time_element:
                        complaint_data['time'] = time_element.get_text(strip=True)
                      # View count - updated selector
                    view_count = profile_section.find('span', class_='js-view-count count js-increment-view')
                    if view_count:
                        complaint_data['views'] = view_count.get_text(strip=True)
                
                # Extract complaint description/preview
                description = card.find('a', class_='complaint-description') or card.find('div', class_='complaint-text')
                if description:
                    complaint_data['description'] = description.get_text(strip=True)
                
                # Extract support count (upvotes)
                upvoter_count = card.get('data-upvoter-count')
                if upvoter_count:
                    complaint_data['supported'] = upvoter_count
                
                # Get detailed information from complaint page
                if 'url' in complaint_data:
                    logger.info(f"Extracting details for complaint: {complaint_data['title']}")
                    details = self.extract_complaint_details(complaint_data['url'])
                    complaint_data.update(details)
                    
                    # Add a small delay to be respectful
                    time.sleep(1)
                
                complaints.append(complaint_data)
                
            except Exception as e:
                logger.error(f"Error processing complaint card: {e}")
                continue
        
        return complaints

    def get_total_pages(self):
        """Get total number of pages"""
        response = self.get_page_content(self.setur_url)
        if not response:
            return 1
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for pagination
        pagination = soup.find('div', class_='pagination') or soup.find('nav', class_='pagination')
        if pagination:
            page_links = pagination.find_all('a')
            if page_links:
                # Get the last page number
                last_page = 1
                for link in page_links:
                    text = link.get_text(strip=True)
                    if text.isdigit():
                        last_page = max(last_page, int(text))
                return last_page
        
        return 1

    def scrape_all_complaints(self, max_pages=None):
        """Scrape all complaints from Setur tourism page"""
        logger.info("Starting complaint scraping...")
        
        # Get total pages if not specified
        if max_pages is None:
            total_pages = self.get_total_pages()
            logger.info(f"Found {total_pages} total pages")
        else:
            total_pages = max_pages
        
        all_complaints = []
        
        for page_num in range(1, total_pages + 1):
            if page_num == 1:
                page_url = self.setur_url
            else:
                page_url = f"{self.setur_url}?page={page_num}"
            
            logger.info(f"Scraping page {page_num} of {total_pages}")
            
            complaints = self.extract_complaints_from_page(page_url)
            all_complaints.extend(complaints)
            
            logger.info(f"Extracted {len(complaints)} complaints from page {page_num}")
            
            # Add delay between pages
            time.sleep(2)
        
        self.complaints_data = all_complaints
        logger.info(f"Total complaints extracted: {len(all_complaints)}")
        return all_complaints

    def save_to_json(self, filename="setur_complaints.json"):
        """Save complaints data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.complaints_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data saved to {filename}")

    def save_to_csv(self, filename="setur_complaints.csv"):
        """Save complaints data to CSV file"""
        if not self.complaints_data:
            logger.warning("No data to save")
            return
        
        # Flatten the data for CSV
        flattened_data = []
        for complaint in self.complaints_data:
            row = {
                'id': complaint.get('id', ''),
                'title': complaint.get('title', ''),
                'username': complaint.get('username', ''),
                'time': complaint.get('time', ''),
                'views': complaint.get('views', ''),
                'views_detail': complaint.get('views_detail', ''),
                'supported': complaint.get('supported', ''),
                'description': complaint.get('description', ''),
                'full_complaint': complaint.get('full_complaint', ''),
                'rating': complaint.get('rating', ''),
                'company_response': complaint.get('company_response', ''),
                'url': complaint.get('url', ''),
                'comments_count': len(complaint.get('comments', [])),
                'comments': json.dumps(complaint.get('comments', []), ensure_ascii=False)
            }
            flattened_data.append(row)
        
        fieldnames = flattened_data[0].keys() if flattened_data else []
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened_data)
        
        logger.info(f"Data saved to {filename}")
    def save_to_sql(self, filename="setur_complaints.sql"):
        def save_to_sql(self, filename="setur_complaints.sql"):
            """Save complaints data to a SQLite database file"""
            if not self.complaints_data:
                logger.warning("No data to save")
                return

            conn = sqlite3.connect(filename)
            c = conn.cursor()

            # Create main complaints table
            c.execute('''
                CREATE TABLE IF NOT EXISTS complaints (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    username TEXT,
                    time TEXT,
                    views TEXT,
                    supported TEXT,
                    description TEXT,
                    full_complaint TEXT,
                    rating INTEGER,
                    company_response TEXT,
                    url TEXT
                )
            ''')

            # Create comments table
            c.execute('''
                CREATE TABLE IF NOT EXISTS comments (
                    complaint_id TEXT,
                    author TEXT,
                    text TEXT,
                    time TEXT,
                    FOREIGN KEY (complaint_id) REFERENCES complaints(id)
                )
            ''')

            # Create replies table
            c.execute('''
                CREATE TABLE IF NOT EXISTS replies (
                    complaint_id TEXT,
                    comment_index INTEGER,
                    author TEXT,
                    text TEXT,
                    time TEXT,
                    FOREIGN KEY (complaint_id) REFERENCES complaints(id)
                )
            ''')

            # Insert complaints and related comments/replies
            for complaint in self.complaints_data:
                c.execute('''
                    INSERT OR REPLACE INTO complaints
                    (id, title, username, time, views, supported, description, full_complaint, rating, company_response, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    complaint.get('id', ''),
                    complaint.get('title', ''),
                    complaint.get('username', ''),
                    complaint.get('time', ''),
                    complaint.get('views', ''),
                    complaint.get('supported', ''),
                    complaint.get('description', ''),
                    complaint.get('full_complaint', ''),
                    complaint.get('rating', None),
                    complaint.get('company_response', ''),
                    complaint.get('url', '')
                ))

                comments = complaint.get('comments', [])
                for idx, comment in enumerate(comments):
                    c.execute('''
                        INSERT INTO comments
                        (complaint_id, author, text, time)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        complaint.get('id', ''),
                        comment.get('author', ''),
                        comment.get('text', ''),
                        comment.get('time', '')
                    ))
                    replies = comment.get('replies', [])
                    for reply in replies:
                        c.execute('''
                            INSERT INTO replies
                            (complaint_id, comment_index, author, text, time)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (
                            complaint.get('id', ''),
                            idx,
                            reply.get('author', ''),
                            reply.get('text', ''),
                            reply.get('time', '')
                        ))

            conn.commit()
            conn.close()
            logger.info(f"Data saved to {filename}")

def main():
    scraper = SeturComplaintScraper()

    # Scrape first 41 pages (you can change this or set to None for all pages)
    complaints = scraper.scrape_all_complaints(max_pages=41)
    
    # Save data in both formats
    scraper.save_to_json("setur_complaints.json")
    scraper.save_to_csv("setur_complaints.csv")
    
    print(f"\nScraping completed! Extracted {len(complaints)} complaints.")
    print("Files saved:")
    print("- setur_complaints.json")
    print("- setur_complaints.csv")
    
    # Print sample data
    if complaints:
        print("\nSample complaint data:")
        sample = complaints[0]
        for key, value in sample.items():
            if key != 'comments':  # Skip comments for brevity
                print(f"{key}: {value}")

if __name__ == "__main__":
    main()