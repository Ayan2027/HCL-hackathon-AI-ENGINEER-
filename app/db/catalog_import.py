"""
Import the ShopMate product catalogue into SQLite - safely and repeatably.

    python -m app.db.catalog_import --dry-run     # validate files + DB, change nothing
    python -m app.db.catalog_import               # back up the DB, then import

Source of truth: data/shopmate_products.json  ({"products": [...]})
Cross-check:     data/shopmate_products.csv   (every field compared with the JSON)

What the import does, in ONE transaction (all or nothing):
  1. Validates every JSON record (required fields, unique IDs, types). Any invalid
     record aborts the whole import - nothing is half-loaded.
  2. Upserts the catalogue products (unchanged rows are not rewritten, so a second
     run changes nothing).
  3. Older products that are NOT in the new catalogue:
       - referenced by an order  -> kept as an ARCHIVED historical record
                                    (is_active = 0: never searched, listed or recommended)
       - not referenced anywhere -> removed
     Orders and customers are never modified or deleted, and no old product ID is
     mapped onto an unrelated new one.
  4. Rebuilds the FTS5 keyword index from the active catalogue only.
Then it writes a validation report (data/catalog_import_report.json) and prints it.

Missing values stay missing: no rating -> NULL, no brand -> NULL. Nothing is invented.
`original_values` and `quality_flags` are kept for provenance only.
"""
import argparse
import csv
import json
import logging
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from app.config import CATEGORIES, settings
from app.db import repository as repo
from app.db.database import get_connection, init_db
from app.schemas import ProductIn

logger = logging.getLogger("shopsense.catalog_import")

REQUIRED_FIELDS = ["product_id", "name", "category", "price", "stock_count", "description"]
EXPECTED_FIELDS = [
    "product_id", "name", "category", "subcategory", "brand", "price", "retail_price", "stock_count",
    "attributes", "description", "rating", "image_url", "source_product_id", "data_source",
    "stock_is_simulated", "description_language", "quality_flags", "original_values", "embedding_text",
]


class CatalogImportError(Exception):
    """The source files failed validation; the database was not changed."""


# ---------------------------------------------------------------------------
# Reading and validating the source files
# ---------------------------------------------------------------------------
def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise CatalogImportError(f"Catalogue JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("products") if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise CatalogImportError(f"{path.name}: expected a list or {{'products': [...]}}")
    return records


def record_to_product(record: Dict[str, Any]) -> ProductIn:
    """Map one ShopMate record onto the product model. No value is guessed or filled in."""
    brand = record.get("brand")
    rating = record.get("rating")
    return ProductIn(
        product_id=str(record["product_id"]).strip(),
        name=str(record["name"]).strip(),
        category=str(record["category"]).strip(),
        subcategory=record.get("subcategory"),
        brand=None if _blank(brand) else str(brand).strip(),
        price=record.get("price"),
        stock_qty=record.get("stock_count") or 0,
        rating=None if _blank(rating) else rating,
        attributes=record.get("attributes") or {},
        description=record.get("description"),
        language=record.get("description_language") or "en",
        retail_price=record.get("retail_price"),
        image_url=None if _blank(record.get("image_url")) else str(record["image_url"]).strip(),
        source_product_id=record.get("source_product_id"),
        data_source=record.get("data_source"),
        stock_is_simulated=bool(record.get("stock_is_simulated")),
        quality_flags=list(record.get("quality_flags") or []),
        original_values=dict(record.get("original_values") or {}),
        embedding_text=record.get("embedding_text"),
        is_active=True,
    )


def validate_records(records: List[Dict[str, Any]]) -> Tuple[List[ProductIn], Dict[str, Any]]:
    """Validate every record. Raises CatalogImportError listing ALL problems if any record fails."""
    errors: List[str] = []
    products: List[ProductIn] = []
    ids = [str(r.get("product_id", "")).strip() for r in records]
    duplicates = sorted(pid for pid, n in Counter(ids).items() if n > 1)
    if duplicates:
        errors.append(f"duplicate product_id values: {duplicates[:10]}")

    missing_field_counts: Counter = Counter()
    for record in records:
        pid = record.get("product_id", "?")
        for field in EXPECTED_FIELDS:
            if field not in record:
                missing_field_counts[field] += 1
        absent = [f for f in REQUIRED_FIELDS if _blank(record.get(f))]
        if absent:
            errors.append(f"{pid}: missing required {absent}")
            continue
        if not isinstance(record.get("stock_count"), int) or record["stock_count"] < 0:
            errors.append(f"{pid}: stock_count must be a non-negative integer")
            continue
        try:
            products.append(record_to_product(record))
        except ValidationError as exc:
            errors.append(f"{pid}: {exc.errors()[0]['loc']} {exc.errors()[0]['msg']}")
    if errors:
        raise CatalogImportError("Catalogue validation failed:\n  " + "\n  ".join(errors[:50]))

    stats = catalogue_stats(records)
    stats["fields_absent_from_some_records"] = dict(missing_field_counts)
    return products, stats


def catalogue_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Statistics recomputed from the file (reported, never hard-coded)."""
    ids = [r.get("product_id") for r in records]
    categories = Counter(r.get("category") for r in records)
    return {
        "records": len(records),
        "unique_product_ids": len(set(ids)),
        "category_distribution": dict(categories),
        "unexpected_categories": sorted(c for c in categories if c not in CATEGORIES),
        "description_languages": dict(Counter(r.get("description_language") for r in records)),
        "missing_rating": sum(1 for r in records if _blank(r.get("rating"))),
        "missing_brand": sum(1 for r in records if _blank(r.get("brand"))),
        "zero_stock": sum(1 for r in records if (r.get("stock_count") or 0) == 0),
        "missing_image_url": sum(1 for r in records if _blank(r.get("image_url"))),
        "missing_embedding_text": sum(1 for r in records if _blank(r.get("embedding_text"))),
        "stock_simulated": sum(1 for r in records if r.get("stock_is_simulated")),
    }


def _csv_value_matches(csv_value: str, json_value: Any) -> bool:
    if isinstance(json_value, bool):
        return csv_value.strip().lower() in ({"true", "1"} if json_value else {"false", "0", ""})
    if isinstance(json_value, (dict, list)):
        if not csv_value.strip():
            return not json_value
        try:
            return json.loads(csv_value) == json_value
        except json.JSONDecodeError:
            return False
    if json_value is None:
        return csv_value.strip() == ""
    if isinstance(json_value, (int, float)):
        try:
            return float(csv_value) == float(json_value)
        except ValueError:
            return False
    return csv_value == json_value


def cross_validate_csv(path: Path, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare the CSV with the JSON field by field. The JSON stays canonical either way."""
    if not path.exists():
        return {"checked": False, "reason": f"{path.name} not found"}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {str(r["product_id"]): r for r in records}
    csv_ids = [row.get("product_id") for row in rows]
    mismatches: Counter = Counter()
    examples: Dict[str, str] = {}
    for row in rows:
        record = by_id.get(row.get("product_id"))
        if record is None:
            continue
        for field, value in row.items():
            if field in record and not _csv_value_matches(value or "", record[field]):
                mismatches[field] += 1
                examples.setdefault(field, row["product_id"])
    return {
        "checked": True,
        "rows": len(rows),
        "ids_only_in_csv": sorted(set(csv_ids) - set(by_id))[:20],
        "ids_only_in_json": sorted(set(by_id) - set(csv_ids))[:20],
        "field_mismatches": dict(mismatches),
        "mismatch_examples": examples,
        "columns_not_in_csv": [f for f in EXPECTED_FIELDS if rows and f not in rows[0]],
        "reconciled": not mismatches and set(csv_ids) == set(by_id) and len(rows) == len(records),
    }


# ---------------------------------------------------------------------------
# Database inspection
# ---------------------------------------------------------------------------
def _counts(conn: sqlite3.Connection) -> Dict[str, int]:
    out = {}
    for table in ["products", "products_fts", "customers", "orders", "sessions", "audit_log"]:
        out[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    out["products_active"] = conn.execute("SELECT COUNT(*) FROM products WHERE is_active = 1").fetchone()[0]
    out["products_archived"] = conn.execute("SELECT COUNT(*) FROM products WHERE is_active = 0").fetchone()[0]
    return out


def order_reference_report(conn: sqlite3.Connection, catalogue_ids: set) -> Dict[str, Any]:
    """Classify every product referenced by an order against the new catalogue."""
    rows = conn.execute(
        """SELECT o.product_id, COUNT(*) AS n, p.product_id AS existing, p.name, p.is_active
           FROM orders o LEFT JOIN products p ON p.product_id = o.product_id
           GROUP BY o.product_id ORDER BY o.product_id"""
    ).fetchall()
    in_catalogue, historical, missing = [], [], []
    for r in rows:
        if r["product_id"] in catalogue_ids:
            in_catalogue.append(r["product_id"])
        elif r["existing"] is not None:
            historical.append({"product_id": r["product_id"], "name": r["name"], "orders": r["n"]})
        else:
            missing.append({"product_id": r["product_id"], "orders": r["n"]})
    return {
        "orders": conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0],
        "distinct_products_referenced": len(rows),
        "resolved_in_new_catalogue": len(in_catalogue),
        "not_in_new_catalogue_kept_as_archived_record": len(historical),
        "orders_pointing_at_archived_records": sum(h["orders"] for h in historical),
        "missing_product_row_entirely": missing,
        "unmapped_product_ids": [h["product_id"] for h in historical],
        "mapping_policy": "No mapping between old and new product IDs is invented. Unmapped order lines "
                          "keep pointing at an archived snapshot of the original product.",
    }


def backup_database(db_path: Path, backup_dir: Path) -> Optional[Path]:
    """Consistent copy using SQLite's online backup API (safe with WAL)."""
    if not db_path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / f"{db_path.stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.db"
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(target))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return target


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------
def _row_matches(conn: sqlite3.Connection, product: ProductIn) -> bool:
    row = conn.execute("SELECT * FROM products WHERE product_id = ?", (product.product_id,)).fetchone()
    if row is None:
        return False
    values = repo.product_db_values(product)
    return all(row[k] == v for k, v in values.items())


def import_catalog(
    json_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    dry_run: bool = False,
    backup: bool = True,
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    json_path = Path(json_path or settings.catalog_json)
    csv_path = Path(csv_path or settings.catalog_csv)
    db_path = Path(db_path or settings.db_path)

    records = load_json_records(json_path)
    products, stats = validate_records(records)          # raises before touching the DB
    csv_report = cross_validate_csv(csv_path, records)
    catalogue_ids = {p.product_id for p in products}

    init_db(db_path)                                      # adds new columns in place if missing
    conn = get_connection(db_path)
    report: Dict[str, Any] = {
        "source_json": str(json_path), "source_csv": str(csv_path), "dry_run": dry_run,
        "catalogue": stats, "csv_cross_check": csv_report,
    }
    try:
        before = _counts(conn)
        report["before"] = before
        existing_ids = {r[0] for r in conn.execute("SELECT product_id FROM products")}
        referenced = {r[0] for r in conn.execute("SELECT DISTINCT product_id FROM orders")}
        legacy = existing_ids - catalogue_ids
        to_archive = sorted(legacy & referenced)
        to_remove = sorted(legacy - referenced)
        unchanged = [p for p in products if _row_matches(conn, p)]
        report["plan"] = {
            "insert": len(catalogue_ids - existing_ids),
            "update": len((catalogue_ids & existing_ids)) - len(unchanged),
            "unchanged": len(unchanged),
            "archive_legacy_referenced_by_orders": len(to_archive),
            "remove_legacy_unreferenced": len(to_remove),
        }

        if dry_run:
            report["order_references"] = order_reference_report(conn, catalogue_ids)
            return report

        if backup:
            conn.close()
            report["backup"] = str(backup_database(db_path, settings.backup_dir) or "")
            conn = get_connection(db_path)

        unchanged_ids = {p.product_id for p in unchanged}
        conn.execute("BEGIN IMMEDIATE")
        try:
            for product in products:
                if product.product_id not in unchanged_ids:
                    repo._write_product(conn, product)            # noqa: SLF001 - same package
            for pid in to_archive:
                conn.execute("UPDATE products SET is_active = 0 WHERE product_id = ? AND is_active != 0", (pid,))
            for pid in to_remove:
                conn.execute("DELETE FROM products WHERE product_id = ?", (pid,))
            fts_rows = repo.rebuild_fts(conn)
            fk_problems = conn.execute("PRAGMA foreign_key_check").fetchall()
            if fk_problems:
                raise CatalogImportError(f"foreign key check failed: {[tuple(r) for r in fk_problems[:5]]}")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        after = _counts(conn)
        report["after"] = after
        report["fts_rows_indexed"] = fts_rows
        report["order_references"] = order_reference_report(conn, catalogue_ids)
        report["checks"] = {
            "active_products_equal_catalogue": after["products_active"] == len(catalogue_ids),
            "fts_rows_equal_active_products": after["products_fts"] == after["products_active"],
            "customers_preserved": after["customers"] == before["customers"],
            "orders_preserved": after["orders"] == before["orders"],
            "every_order_has_a_product_row": not report["order_references"]["missing_product_row_entirely"],
        }
    finally:
        conn.close()

    report_path = Path(report_path) if report_path else db_path.parent / "catalog_import_report.json"
    try:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_file"] = str(report_path)
    except OSError as exc:                     # the report is informational only
        logger.warning("Could not write import report: %s", exc)
    return report


def catalogue_is_current(db_path: Optional[Path] = None) -> bool:
    """True if every product in the catalogue JSON is active in the database (no stale catalogue)."""
    try:
        records = load_json_records(settings.catalog_json)
    except (CatalogImportError, json.JSONDecodeError):
        return False
    ids = {str(r.get("product_id")) for r in records}
    conn = get_connection(db_path)
    try:
        active = {r[0] for r in conn.execute("SELECT product_id FROM products WHERE is_active = 1")}
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()
    return ids == active


def _print_summary(report: Dict[str, Any]) -> None:
    c = report["catalogue"]
    print(f"Source: {report['source_json']}")
    print(f"  records={c['records']} unique_ids={c['unique_product_ids']} "
          f"missing_rating={c['missing_rating']} missing_brand={c['missing_brand']} zero_stock={c['zero_stock']}")
    print(f"  categories={c['category_distribution']}")
    print(f"  languages={c['description_languages']}  unexpected_categories={c['unexpected_categories']}")
    x = report["csv_cross_check"]
    print(f"CSV cross-check: {'reconciled' if x.get('reconciled') else x}")
    print(f"Before: {report['before']}")
    print(f"Plan:   {report['plan']}")
    if "after" in report:
        print(f"After:  {report['after']}")
        print(f"FTS rows indexed: {report['fts_rows_indexed']}")
        print(f"Checks: {report['checks']}")
        print(f"Backup: {report.get('backup') or 'none (new database)'}")
    o = report["order_references"]
    print(f"Order references: {o['orders']} orders -> {o['distinct_products_referenced']} products; "
          f"{o['resolved_in_new_catalogue']} in new catalogue; "
          f"{o['not_in_new_catalogue_kept_as_archived_record']} unmapped (kept as archived records); "
          f"{len(o['missing_product_row_entirely'])} missing entirely")
    if o["unmapped_product_ids"]:
        print(f"  Unmapped product IDs: {', '.join(o['unmapped_product_ids'])}")
    if report.get("report_file"):
        print(f"Full report: {report['report_file']}")
    if not report["dry_run"]:
        print("Next: rebuild the vector index with  python -m scripts.build_index  (if vector search is on).")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and import the ShopMate catalogue into SQLite")
    parser.add_argument("--json", type=Path, help="catalogue JSON (default: settings.catalog_json)")
    parser.add_argument("--csv", type=Path, help="catalogue CSV for cross-validation")
    parser.add_argument("--dry-run", action="store_true", help="validate and show the plan; change nothing")
    parser.add_argument("--no-backup", action="store_true", help="skip the automatic database backup")
    args = parser.parse_args()
    try:
        report = import_catalog(args.json, args.csv, dry_run=args.dry_run, backup=not args.no_backup)
    except CatalogImportError as exc:
        print(f"IMPORT ABORTED - database not changed.\n{exc}")
        return 1
    _print_summary(report)
    checks = report.get("checks", {})
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
