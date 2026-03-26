import csv

from common.utilities.web_crawler.crawler_data import Company


def load_companies_from_csv(csv_path: str, limit: int | None = None) -> list[Company]:
    companies = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            companies.append(
                Company(
                    company_identifier=row["CompanyName"],
                    company_identifier_type="name",
                    company_name=row["CompanyName"],
                    root_url=row["InternetAddress"].strip(),
                )
            )
            if limit and len(companies) >= limit:
                break
    return companies
