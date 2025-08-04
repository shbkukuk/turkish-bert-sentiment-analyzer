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
            # Extract complaint text - comprehensive search for complaint content
            complaint_text = None
            
            # Try multiple selectors for complaint text
            complaint_selectors = [
                'div.complaint-detail-description',
                'div.complaint-text', 
                'div.complaint-content',
                'div.complaint-detail',
                'section.complaint-content',
                'article.complaint-text',
                'div.complaint-thank-message',
                'p.complaint-description',
                'div.complaint-description',
                'div.complaint-preview'
            ]
            for selector in complaint_selectors:
                complaint_element = soup.select_one(selector)
                if complaint_element:
                    # Get all paragraphs within the complaint detail
                    paragraphs = complaint_element.find_all('p')
                    if paragraphs:
                        complaint_text = '\n'.join([p.get_text(strip=True) for p in paragraphs])
                    else:
                        complaint_text = complaint_element.get_text(strip=True)
                    break
            
            # If still no complaint text found, try more specific selectors
            if not complaint_text:
                # Try to find complaint text in specific nested structures
                specific_selectors = [
                    'div.complaint-detail-description p',
                    'div.complaint-content p',
                    'div.complaint-text p',
                    'div.complaint-detail p',
                    'p.complaint-description',
                    'div.complaint-description p',
                    'div.complaint-preview p'
                ]
                for selector in specific_selectors:
                    elements = soup.select(selector)
                    if elements:
                        complaint_text = '\n'.join([elem.get_text(strip=True) for elem in elements])
                        break
            
            # If still no complaint text, try even more broad selectors
            if not complaint_text:
                broad_selectors = [
                    'div[class*="complaint"] p',
                    'div[class*="detail"] p',
                    'div[class*="description"] p',
                    'article p',
                    'main p'
                ]
                for selector in broad_selectors:
                    elements = soup.select(selector)
                    if elements:
                        # Filter out very short texts (likely not the main complaint)
                        filtered_texts = [elem.get_text(strip=True) for elem in elements if len(elem.get_text(strip=True)) > 20]
                        if filtered_texts:
                            complaint_text = '\n'.join(filtered_texts)
                            break
            
            # Last resort: try to find any substantial text content
            if not complaint_text:
                # Look for any paragraph with substantial content
                all_paragraphs = soup.find_all('p')
                substantial_paragraphs = []
                for p in all_paragraphs:
                    text = p.get_text(strip=True)
                    # Look for paragraphs with substantial content (more than 50 characters)
                    # and exclude navigation/footer content
                    if (len(text) > 50 and 
                        not any(word in text.lower() for word in ['menu', 'copyright', 'footer', 'navigation', 'cookie', 'gizlilik'])):
                        substantial_paragraphs.append(text)
                
                if substantial_paragraphs:
                    # Take the longest paragraph as it's likely the complaint text
                    complaint_text = max(substantial_paragraphs, key=len)
            
            details['complaint_text'] = complaint_text or ''
            if details['complaint_text'] == '':
                logger.warning(f"No complaint text found for {complaint_url}")
                # Add debug information about what was found on the page
                logger.debug(f"Available divs: {[div.get('class') for div in soup.find_all('div') if div.get('class')]}")
            else:
                logger.debug(f"Found complaint text ({len(details['complaint_text'])} chars) for {complaint_url}")
                
            
            # Extract view count from detail page with multiple selectors
            view_selectors = [
                'span.js-view-count.count.js-increment-view',
                'span.view-count',
                'div.view-count',
                'span.count'
            ]
            
            for selector in view_selectors:
                view_element = soup.select_one(selector)
                if view_element:
                    details['view'] = view_element.get_text(strip=True)
                    break
            
            # Extract user information from detail page
            user_selectors = [
                'a.username',
                'span.username',
                'div.username',
                'header.profile-details a',
                'div.profile a'
            ]
            
            for selector in user_selectors:
                user_element = soup.select_one(selector)
                if user_element:
                    details['user_id'] = user_element.get_text(strip=True)
                    break
            
            # Extract timestamp with multiple selectors
            timestamp_selectors = [
                'div.js-tooltip.time',
                'time',
                'span.time',
                'div.time',
                'span.date'
            ]
            
            for selector in timestamp_selectors:
                timestamp_element = soup.select_one(selector)
                if timestamp_element:
                    details['timestamp'] = timestamp_element.get_text(strip=True)
                    break
            
            # Extract complaint answer container (company response)
            answer_selectors = [
                'div.complaint-answer-container',
                'div.complaint-reply-wrapper',
                'div.company-response',
                'section.response',
                'div.response-container'
            ]
            
            for selector in answer_selectors:
                answer_element = soup.select_one(selector)
                if answer_element:
                    # Look for message within the container
                    message_element = answer_element.find('p', class_='message') or answer_element
                    details['complaint_answer_container'] = message_element.get_text(strip=True)
                    break
            
            # Extract comments with comprehensive selectors
            comments = []
            comment_selectors = [
                'div.comment',
                'article.comment',
                'div.comment-item',
                'section.comment'
            ]
            
            comment_sections = []
            for selector in comment_selectors:
                found_comments = soup.select(selector)
                if found_comments:
                    comment_sections = found_comments
                    break
            
            for comment in comment_sections:
                comment_data = {}
                
                # Comment author with multiple selectors
                author_selectors = ['a.username', 'span.comment-author', 'div.author', 'span.username']
                for selector in author_selectors:
                    author = comment.select_one(selector)
                    if author:
                        comment_data['author'] = author.get_text(strip=True)
                        break
                
                # Comment text with multiple selectors
                text_selectors = ['div.comment-text', 'p.comment-content', 'div.comment-body', 'span.comment-text']
                for selector in text_selectors:
                    comment_text = comment.select_one(selector)
                    if comment_text:
                        comment_data['text'] = comment_text.get_text(strip=True)
                        break
                
                # Comment time with multiple selectors
                time_selectors = ['time', 'span.time', 'div.time', 'span.date']
                for selector in time_selectors:
                    comment_time = comment.select_one(selector)
                    if comment_time:
                        comment_data['time'] = comment_time.get_text(strip=True)
                        break
                
                # Replies to this comment
                replies = []
                reply_selectors = ['div.reply', 'div.comment-reply', 'article.reply']
                
                reply_elements = []
                for selector in reply_selectors:
                    found_replies = comment.select(selector)
                    if found_replies:
                        reply_elements = found_replies
                        break
                
                for reply in reply_elements:
                    reply_data = {}
                    
                    # Reply author
                    for selector in author_selectors:
                        reply_author = reply.select_one(selector)
                        if reply_author:
                            reply_data['author'] = reply_author.get_text(strip=True)
                            break
                    
                    # Reply text
                    reply_text_selectors = ['div.reply-text', 'p.reply-content', 'div.reply-body']
                    for selector in reply_text_selectors:
                        reply_text = reply.select_one(selector)
                        if reply_text:
                            reply_data['text'] = reply_text.get_text(strip=True)
                            break
                    
                    # Reply time
                    for selector in time_selectors:
                        reply_time = reply.select_one(selector)
                        if reply_time:
                            reply_data['time'] = reply_time.get_text(strip=True)
                            break
                    
                    if reply_data:  # Only add if we found some data
                        replies.append(reply_data)
                
                comment_data['replies'] = replies
                if comment_data:  # Only add if we found some data
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
                
                # Extract complaint ID with multiple selectors
                complaint_id = card.get('data-id') or card.get('id')
                if not complaint_id:
                    # Try to extract from URL or other attributes
                    url_element = card.find('a')
                    if url_element and url_element.get('href'):
                        href = url_element.get('href')
                        # Extract ID from URL if possible
                        id_match = re.search(r'/(\d+)', href)
                        if id_match:
                            complaint_id = id_match.group(1)
                
                if complaint_id:
                    complaint_data['id'] = complaint_id
                
                # Extract complaint title and URL with multiple selectors
                title_selectors = [
                    'a.complaint-layer',
                    'h2 a',
                    'h3 a',
                    'a[href*="complaint"]',
                    'a[href*="sikayet"]'
                ]
                
                title_link = None
                for selector in title_selectors:
                    title_link = card.select_one(selector)
                    if title_link:
                        break
                
                if title_link:
                    complaint_data['title'] = title_link.get_text(strip=True)
                    complaint_data['url'] = urljoin(self.base_url, title_link.get('href'))
                
                # Extract user information with comprehensive selectors
                user_selectors = [
                    'header.profile-details',
                    'div.profile',
                    'div.user-info',
                    'section.profile'
                ]
                
                profile_section = None
                for selector in user_selectors:
                    profile_section = card.select_one(selector)
                    if profile_section:
                        break
                
                if profile_section:
                    # Username/User ID with multiple selectors
                    username_selectors = ['a.username', 'span.username', 'div.username']
                    for selector in username_selectors:
                        username_link = profile_section.select_one(selector)
                        if username_link:
                            complaint_data['user_id'] = username_link.get_text(strip=True)
                            break
                    
                    # Timestamp with multiple selectors
                    time_selectors = [
                        'div.js-tooltip.time',
                        'time',
                        'span.time',
                        'div.time',
                        'span.date'
                    ]
                    for selector in time_selectors:
                        time_element = profile_section.select_one(selector)
                        if time_element:
                            complaint_data['timestamp'] = time_element.get_text(strip=True)
                            break
                    
                    # View count with multiple selectors
                    view_selectors = [
                        'span.js-view-count.count.js-increment-view',
                        'span.view-count',
                        'div.view-count',
                        'span.count'
                    ]
                    for selector in view_selectors:
                        view_count = profile_section.select_one(selector)
                        if view_count:
                            complaint_data['view'] = view_count.get_text(strip=True)
                            break
                
                # Extract complaint description/preview text from listing page
                description_selectors = [
                    'p.complaint-description',
                    'div.complaint-description',
                    'a.complaint-description',
                    'div.complaint-text',
                    'div.complaint-preview',
                    'div.complaint-detail-description',
                    'p.complaint-summary',
                    'div.complaint-thank-message',
                ]
                
                for selector in description_selectors:
                    description = card.select_one(selector)
                    if description:
                        complaint_data['complaint_text_preview'] = description.get_text(strip=True)
                        # Also set as main complaint text if we don't have a URL to visit detail page
                        if 'url' not in complaint_data:
                            complaint_data['complaint_text'] = description.get_text(strip=True)
                        break
                
                # Extract support count (upvotes) - keeping this as additional info
                upvoter_count = card.get('data-upvoter-count')
                if upvoter_count:
                    complaint_data['supported'] = upvoter_count
                
                # Get detailed information from complaint page
                if 'url' in complaint_data:
                    logger.info(f"Extracting details for complaint: {complaint_data.get('title', 'Unknown')}")
                    details = self.extract_complaint_details(complaint_data['url'])
                    complaint_data.update(details)
                    
                    # If detail page didn't provide complaint text, use the preview text
                    if not details.get('complaint_text') and complaint_data.get('complaint_text_preview'):
                        complaint_data['complaint_text'] = complaint_data['complaint_text_preview']
                    
                    # Add a small delay to be respectful
                    time.sleep(1)
                else:
                    # If no URL available, try to get complaint text from various sources
                    complaint_text = complaint_data.get('complaint_text', '')
                    
                    if not complaint_text:
                        # Try to get complaint text from thank message
                        thank_message = card.select_one('div.complaint-thank-message p')
                        if thank_message:
                            complaint_text = thank_message.get_text(strip=True)
                    
                    if not complaint_text:
                        # Try to get complaint text from thank message div
                        thank_div = card.select_one('div.complaint-thank-message')
                        if thank_div:
                            complaint_text = thank_div.get_text(strip=True)
                    
                    complaint_data['complaint_text'] = complaint_text
                
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
        if pagination and hasattr(pagination, 'find_all'):
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
    def search_complaints_is_empty(self, complaints):
        """Remove complaints with empty text and log warnings"""
        filtered_complaints = []
        removed_count = 0
        
        for complaint in complaints:
            if not complaint.get('complaint_text'):
                logger.warning(f"User ID {complaint.get('user_id', 'Unknown')} Complaint ID {complaint.get('id', 'Unknown')} has no text.")
                removed_count += 1
            else:
                filtered_complaints.append(complaint)
        
        if removed_count == 0:    
            logger.info("All complaints have text content.")
        else:
            logger.info(f"Removed {removed_count} complaints without text content.")
        
        return filtered_complaints

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
            complaints = self.search_complaints_is_empty(complaints)
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
        
        # Flatten the data for CSV with updated field names
        flattened_data = []
        for complaint in self.complaints_data:
            row = {
                'id': complaint.get('id', ''),
                'title': complaint.get('title', ''),
                'complaint_text': complaint.get('complaint_text', ''),
                'user_id': complaint.get('user_id', ''),
                'timestamp': complaint.get('timestamp', ''),
                'view': complaint.get('view', ''),
                'url': complaint.get('url', ''),
                'complaint_answer_container': complaint.get('complaint_answer_container', ''),
                'comments_count': len(complaint.get('comments', [])),
                'comments': json.dumps(complaint.get('comments', []), ensure_ascii=False),
                # Keep some additional fields that might be useful
                'complaint_text_preview': complaint.get('complaint_text_preview', ''),
                'supported': complaint.get('supported', '')
            }
            flattened_data.append(row)
        
        fieldnames = flattened_data[0].keys() if flattened_data else []
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened_data)
        
        logger.info(f"Data saved to {filename}")

    def save_to_sql(self, filename="setur_complaints.sql"):
        """Save complaints data to a SQLite database file"""
        if not self.complaints_data:
            logger.warning("No data to save")
            return

        conn = sqlite3.connect(filename)
        c = conn.cursor()

        # Create main complaints table with updated field names
        c.execute('''
            CREATE TABLE IF NOT EXISTS complaints (
                id TEXT PRIMARY KEY,
                title TEXT,
                complaint_text TEXT,
                user_id TEXT,
                timestamp TEXT,
                view TEXT,
                url TEXT,
                complaint_answer_container TEXT,
                complaint_text_preview TEXT,
                supported TEXT
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

        # Insert complaints and related comments/replies with updated field names
        for complaint in self.complaints_data:
            c.execute('''
                INSERT OR REPLACE INTO complaints
                (id, title, complaint_text, user_id, timestamp, view, url, complaint_answer_container, complaint_text_preview, supported)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                complaint.get('id', ''),
                complaint.get('title', ''),
                complaint.get('complaint_text', ''),
                complaint.get('user_id', ''),
                complaint.get('timestamp', ''),
                complaint.get('view', ''),
                complaint.get('url', ''),
                complaint.get('complaint_answer_container', ''),
                complaint.get('complaint_text_preview', ''),
                complaint.get('supported', '')
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
    scraper.save_to_json("setur_complaints_new.json")
    scraper.save_to_csv("setur_complaints_new.csv")
    
    print(f"\nScraping completed! Extracted {len(complaints)} complaints.")
    print("Files saved:")
    print("- setur_complaints_new.json")
    print("- setur_complaints_new.csv")

    # Print sample data
    if complaints:
        print("\nSample complaint data:")
        sample = complaints[0]
        for key, value in sample.items():
            if key != 'comments':  # Skip comments for brevity
                print(f"{key}: {value}")

if __name__ == "__main__":
    main()