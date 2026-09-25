from scrape import main as scrape_books
from database import build_database
from queries import main as run_queries

if __name__ == "__main__":
    scrape_books()
    build_database()
    run_queries()
