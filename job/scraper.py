import json
import time
import logging
import requests
import feedparser
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse
from typing import List, Dict, Optional, Callable, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

class NewsScraper:
    def __init__(self, 
                 site_name: str, 
                 feed_url: str, 
                 parser_func: Callable[[BeautifulSoup, Any], Dict[str, Any]],
                 url_modifier: Optional[Callable[[str], str]] = None,
                 sleep_time: int = 1):
        self.site_name = site_name
        self.feed_url = feed_url
        self.parser_func = parser_func
        self.url_modifier = url_modifier
        self.sleep_time = sleep_time
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _fetch_url(self, url: str) -> Optional[str]:
        try:
            r = self.session.get(url, timeout=20)
            r.raise_for_status()
            r.encoding = r.apparent_encoding if r.encoding == 'ISO-8859-1' else r.encoding
            return r.text
        except Exception as e:
            logger.error(f"[{self.site_name}] Failed to fetch {url}: {e}")
            return None

    @staticmethod
    def _format_date(entry: Any) -> str:
        """Extracts and formats date from RSS entry if available."""
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            return time.strftime("%Y-%m-%d", entry.published_parsed)
        return ""

    def run(self, target_date: str) -> List[Dict]:
        """
        Fetches feed and returns articles only matching the target_date.
        :param target_date: Date string in 'YYYY-MM-DD' format.
        """
        logger.info(f"Starting scrape for: {self.site_name} (Filter: {target_date})")
        feed = feedparser.parse(self.feed_url)
        results = []

        if not feed.entries:
            logger.warning(f"No entries found for {self.site_name}")
            return results

        for entry in feed.entries:
            # 1. Check RSS Date First (Optimization)
            rss_date = self._format_date(entry)
            
            # If RSS date is present and NOT today, skip immediately
            if rss_date and rss_date != target_date:
                continue

            # 2. Prepare URL
            raw_url = entry.link
            url = self.url_modifier(raw_url) if self.url_modifier else raw_url
            rss_title = getattr(entry, "title", "No Title")

            # 3. Fetch Content
            html_content = self._fetch_url(url)
            if not html_content:
                continue

            try:
                soup = BeautifulSoup(html_content, "html.parser")
                parsed_data = self.parser_func(soup, entry)
                
                # Determine final date (prefer HTML date, fallback to RSS)
                final_date = parsed_data.get("published") or rss_date

                # 4. Final Date Check
                # (Essential if RSS had no date, but HTML parsing found one)
                if final_date != target_date:
                    continue

                article = {
                    "source": self.site_name,
                    "title": parsed_data.get("title") or rss_title,
                    "link": url,
                    "published": final_date,
                    "content": parsed_data.get("content", "")
                }
                
                if article["content"]:
                    results.append(article)
                    logger.info(f"✔ [{self.site_name}] Extracted: {article['title']}")
                else:
                    logger.warning(f"⚠ [{self.site_name}] Empty content for: {url}")

            except Exception as e:
                logger.error(f"✘ [{self.site_name}] Error parsing {url}: {e}")
            
            time.sleep(self.sleep_time)
        
        return results

# --- Site-Specific Logic (Same as before) ---

def bbc_url_converter(url: str) -> str:
    p = urlparse(url)
    path = p.path.replace("/trad", "/simp")
    return urlunparse((p.scheme, p.netloc, path, "", "", ""))

def parse_chinanews(soup: BeautifulSoup, entry: Any) -> Dict[str, Any]:
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else None
    content_div = soup.find("div", class_="left_zw")
    paragraphs = [p.get_text(strip=True) for p in content_div.find_all("p")] if content_div else []
    return {"title": title, "content": "\n".join(filter(None, paragraphs))}

def parse_bbc(soup: BeautifulSoup, entry: Any) -> Dict[str, Any]:
    title_el = soup.select_one("main h1")
    title = title_el.get_text(strip=True) if title_el else None
    time_el = None
    for t in soup.select("main time[datetime]"):
        if not t.find_parent("figure"):
            time_el = t
            break
    published = time_el["datetime"][:10] if time_el else None # Slice to YYYY-MM-DD
    paras = []
    for p in soup.select("main p"):
        if p.find_parent("figure") or p.find_parent("section", attrs={"data-testid": "byline"}):
            continue
        text = p.get_text(" ", strip=True)
        if text and not text.startswith("图像来源"):
            paras.append(text)
    return {"title": title, "published": published, "content": "\n".join(paras)}

def parse_huxiu(soup: BeautifulSoup, entry: Any) -> Dict[str, Any]:
    content_div = soup.find("div", class_="article__content")
    paragraphs = []
    if content_div:
        all_p = content_div.find_all("p")
        paragraphs = [p.get_text(strip=True) for p in all_p[1:]]
    return {"content": "\n".join(filter(None, paragraphs))}

def parse_guokr(soup: BeautifulSoup, entry: Any) -> Dict[str, Any]:
    content_div = soup.find("div", class_="styled__ArticleContent-sc-1ctyfcr-4") 
    paragraphs = [p.get_text(strip=True) for p in content_div.find_all("p")] if content_div else []
    return {"content": "\n".join(filter(None, paragraphs))}

def parse_chuapp(soup: BeautifulSoup, entry: Any) -> Dict[str, Any]:
    content_div = soup.find("div", class_="the-content")
    paragraphs = []
    if content_div:
        for p in content_div.find_all("p"):
            if not p.find_parent("figure"):
                paragraphs.append(p.get_text(strip=True))
    return {"content": "\n".join(filter(None, paragraphs))}

# --- Main Execution ---

def scrape():
    # 1. Determine "Today"
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")         # Format: 2023-10-27 (for filtering)
    filename_date = now.strftime("%y_%m_%d")     # Format: 23_10_27 (for filename)
    output_filename = f"news_{filename_date}.json"

    logger.info(f"Target Date: {today_str}")
    logger.info(f"Output File: {output_filename}")

    tasks = [
        {"site_name": "ChinaNews", "feed_url": "https://www.chinanews.com.cn/rss/life.xml", "parser_func": parse_chinanews},
        {"site_name": "BBC Chinese", "feed_url": "https://feeds.bbci.co.uk/zhongwen/trad/rss.xml", "parser_func": parse_bbc, "url_modifier": bbc_url_converter},
        {"site_name": "Huxiu", "feed_url": "https://rss.huxiu.com/", "parser_func": parse_huxiu},
        {"site_name": "Guokr", "feed_url": "https://rss.aishort.top/?type=guokr", "parser_func": parse_guokr},
        {"site_name": "Chuapp", "feed_url": "http://www.chuapp.com/feed", "parser_func": parse_chuapp}
    ]

    all_news_data = []

    for task in tasks:
        scraper = NewsScraper(**task)
        # Pass the date filter to the run method
        site_results = scraper.run(target_date=today_str)
        all_news_data.extend(site_results)

    # Save aggregated data
    if all_news_data:
        try:
            # with open(output_filename, "w", encoding="utf-8") as f:
            #     json.dump(all_news_data, f, ensure_ascii=False, indent=4)
            return all_news_data
            logger.info(f"Successfully saved {len(all_news_data)} articles to {output_filename}")
        except IOError as e:
            logger.error(f"Failed to save aggregated file: {e}")
    else:
        logger.warning(f"No news found for {today_str}. JSON file was not created.")
        return []

if __name__ == "__main__":
    scrape()