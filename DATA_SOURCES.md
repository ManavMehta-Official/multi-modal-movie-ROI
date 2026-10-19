# Data provenance, permissions and limitations

## Metadata

Downloaded 13 October 2026 from the public educational repository:
https://github.com/TjanMichela/Movie-Data-Analysis

Direct source: https://raw.githubusercontent.com/TjanMichela/Movie-Data-Analysis/main/movies_complete.csv

44,691 records in the raw snapshot (35,162,226 bytes). SHA-256:
`1d5577a113f0bf99e954884c6f37924703930c300a4ac31ed50494a11f050aa9`

The repository includes an MIT license, copyright 2024 Michela Tjan; an unmodified copy is retained at `data/raw/LICENSE`. This repository license must not be interpreted as transferring ownership of third-party movie artwork or granting unrestricted rights to the underlying TMDB data. The metadata is an existing public TMDB-derived snapshot, not a new official TMDB API collection.

## TMDB attribution and image rights

The Movie Database (TMDB) is the underlying data and poster source.
https://www.themoviedb.org

This product uses the TMDB API but is not endorsed or certified by TMDB.

The statement above is TMDB’s prescribed attribution wording. This implementation itself consumes an existing TMDB-derived metadata snapshot and public TMDB CDN images rather than authenticated API endpoints.

TMDB's official FAQ states that its API/data/images can be used for noncommercial purposes with attribution. It explicitly says TMDB does not claim ownership of the images or data. Movie artwork remains subject to its owners' rights. This project is an attributed, noncommercial educational prototype; do not assume this authorizes commercial use or unrestricted redistribution of posters. Recheck applicable rights before public deployment. The downloader caches small 185px posters from the public official TMDB image CDN for this educational project; it does not create an account, use API credentials, or bypass access restrictions.

- Official FAQ: https://developer.themoviedb.org/docs/faq
- Official image URL documentation: https://developer.themoviedb.org/docs/image-basics
- Official attribution/logo guidance: https://www.themoviedb.org/about/logos-attribution

## Reproduction

Run `python scripts/prepare_data.py` from the project environment. Existing raw files are reused. The acquisition script supports small poster caches, but network poster acquisition for this package is now disabled by `data/DOWNLOAD_STOPPED.json` after broad HTTP 403 failures. It previously used bounded 150-item batches, up to 32 concurrent requests, a two-request-starts-per-second cap, and a 60-second cooldown on HTTP 429. Do not bypass denied access. Use the included cache for the submitted experiment. Each poster path is extracted from the source's HTML image field; only the official HTTPS image.tmdb.org host and w185 path are accepted. Failed/missing posters are logged, not fabricated. Original poster URLs are retained for provenance; `poster_path` is relative to `data/`.

## Cohort and units

`data/movies.csv` contains 4,910 unique movies with reported budget at least USD 1 million, positive reported revenue, and valid release years 1920–2016. The actual range is 1922-01-11 to 2016-12-25. The source budget_musd and revenue_musd fields were multiplied by 1,000,000 to produce nominal USD. Ten plots are absent and retained as empty text. Poster acquisition order is shuffled with random seed 42, independently of outcomes; failed downloads are logged and rerunning reuses successful cached files. No outcome-based removal of extreme positive ROI observations is performed. See `data/data_audit.json` for the audit.

This is a historical cohort with reporting/selection bias: zero or unknown revenues are not assumed to mean zero box office and are excluded. The figures are unverified community-reported nominal values, may mix geographic scopes, and do not account for inflation. Budgets, release metadata, plots and posters in a modern snapshot are not guaranteed to have been available in exactly that form before release. Contemporary reissue artwork may encode later information. Exclude post-release popularity, ratings and vote counts from predictive inputs.

The target revenue/budget minus one is a **gross box-office-to-production-budget proxy**, not investor ROI or net profit. It excludes marketing, distribution and exhibitor shares, financing, taxes and ancillary income. Any risk class/probability refers only to the stated gross proxy threshold.

The implemented experiment uses four chronological partitions, grouping release dates to prevent a same-day boundary overlap: training, validation, probability/interval calibration, and final test. Exact row counts and date boundaries are recorded in the generated metrics report; use that report as authoritative. Fit all encoders and text vocabulary only on training, select/tune on validation, and calibrate only on the calibration partition; do not optimize against test outcomes. Report modality ablations honestly even if images do not improve performance. A poster can be absent; report missingness rather than synthesizing an image.


## Frozen complete-case experiment

Acquisition stopped on 13 October 2026 after broad access failures. The final cache contains 2,601 valid 185px posters totaling 50,916,030 bytes. This is 52.97% of the 4,910-movie financially complete cohort. The remaining 2,309 rows comprise 82 access-denied assets, 14 unavailable (HTTP 404) assets, and 2,213 not acquired when downloading stopped. No substitute or synthetic movie images are used.

`data/movies_multimodal.csv` is the final 2,601-row complete-case cohort. Every model in the reported modality ablation must use these same movies and the same chronological partitions. `data/movies.csv` retains all 4,910 rows for provenance and separately labeled metadata-only work; its evaluation is not directly comparable as a modality effect.

Acquisition order was a deterministic seed-42 shuffle independent of revenue/ROI; eight earliest films were cached before this shuffle. The complete-case cohort is still selected by access and image availability, not a representative random population sample. State this limitation explicitly. Per-year counts for both cohorts are in `data/cohort_by_year.csv`.

Run `python scripts/freeze_data.py` to rebuild the complete-case CSV from the existing local cache, validate every image, and create the hash manifest. This is offline. Do not rerun it or change the cache during model training. Exact CSV/image hashes and coverage are recorded in `data/cohort_audit.json` and `data/poster_manifest.json`.
