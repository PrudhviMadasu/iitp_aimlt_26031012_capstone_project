# Module 1 — Data Pipeline

This is the data-engineering part of the Zepto capstone. It scrapes a bookstore catalog, cleans the records, converts the prices to INR, writes everything into a normalized SQLite database, and checks the output with both SQL and pandas.

## What it does

`data_pipeline.ipynb` scrapes book listings from [books.toscrape.com](http://books.toscrape.com/), a public sandbox site made for scraping practice. It walks through five categories — Mystery, Fiction, Fantasy, Romance, and Sequential Art — and follows pagination within each one. That gives us 255 books, which is comfortably above the 60-book minimum and spread across more than three categories.

For each book, it captures the title, price, star rating, availability, and category directly from the listing pages. There is no need to open each book detail page separately, since the listing page already has the fields we need.

The notebook then:
- cleans `price` into a float `price_gbp`, converts the star rating word into an integer from 1 to 5, and turns the stock text into a boolean `in_stock`
- converts `price_gbp` into `price_inr` using a fixed project rate
- writes the data into `books.db` with a normalized `categories` / `books` schema
- runs five SQL queries covering filters, sorting, limits, distinct values, ranges, and joins
- compares the SQL output against a pandas `pd.merge` version to confirm both methods line up

## Setup

```bash
cd data_pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running it

Re-run the whole pipeline (scrape → clean → build `books.db` → queries) and bake the outputs into the
notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace data_pipeline.ipynb
```

Or open it interactively:

```bash
jupyter notebook data_pipeline.ipynb
```

Either way it scrapes live from books.toscrape.com, so it needs network access and takes a bit under a
minute (sequential requests, ~20 pages total, with a small delay between requests — it's a small public
site so there's no need for concurrency or elaborate rate-limiting).

`books.db` is committed as produced by the last run, but running the notebook again regenerates it from
scratch (the notebook deletes any existing `books.db` before rebuilding, so it's safe to re-run).

## Currency conversion

`price_inr = price_gbp * 105.50`. That rate (1 GBP = 105.50 INR) is a fixed, project-defined constant
for this capstone — not a live FX rate. It's hardcoded rather than pulled from an API because the spec
calls for a fixed baseline rate here; hitting a live currency API is explicitly the stretch goal and was
skipped as out of scope for this module.

## Design decisions

I kept this module deliberately simple and stable because the goal was to build a reliable ETL pipeline without adding a lot of moving parts.

- I used a single fixed GBP-to-INR conversion rate instead of a live FX API. That makes the output reproducible and keeps the project easy to explain and evaluate.
- I scraped a public demo bookstore rather than a production catalog. That keeps the setup lightweight while still giving me realistic enough data to test cleaning, validation, and database loading.
- I split category names into a separate table and linked them to books with foreign keys. I wanted the schema to reflect a real-world normalized design rather than storing duplicated category text in every row.
- I checked the results in both SQL and pandas. That gave me a practical double-check that the database and the cleaned data were behaving consistently.
- I handled missing values conservatively. Numeric fields use median imputation when needed, while unreliable non-numeric values are dropped instead of being guessed.

## Cleaning / missing-data decisions

- **`price_gbp` and `rating`** are numeric, so if a value cannot be parsed, I fill it with the median for that column instead of dropping the whole row. That keeps the dataset stable without inventing a fake price or rating.
- **`title` and `in_stock`** are not good candidates for imputation. There is no meaningful median title, and guessing stock status would just hide real data issues, so rows with unusable values there are dropped.
- In practice, all 255 scraped rows parsed cleanly in this run, so there were no real drops or imputations to show. The logic is still implemented in the notebook so the pipeline fails gracefully if the source site changes and produces messy rows later.

## Schema

The database is normalized into two tables:

```
categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE NOT NULL)
books(book_id INTEGER PRIMARY KEY, title TEXT, price_gbp REAL, price_inr REAL,
      rating INTEGER, in_stock INTEGER, category_id INTEGER REFERENCES categories(category_id))
```

This keeps category names in one place instead of repeating them across every book row. The books table then points back to the category table with a real foreign key, which is what the join query depends on. SQLite does not have a native boolean type, so `in_stock` is stored as `0` and `1` and converted back to Python booleans when it is read back into pandas.

## Files

- `data_pipeline.ipynb` — the pipeline, already executed with real output cells
- `books.db` — SQLite database produced by the notebook's last run
- `requirements.txt` — exact packages this module needs
- `.venv/` — local virtualenv (not committed; see Setup above)
