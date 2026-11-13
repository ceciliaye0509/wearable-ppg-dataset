# ECG polarity research utilities

The `beatwise`, `direction_v3`, evaluation and packet-building modules retain the historical scientific methods. Legacy Python review servers and their duplicate HTML/JS frontends have been removed. The single browser application is [`review_platform`](../../../review_platform/README.md), deployed to Cloudflare Pages.

Direction decisions, method preferences and R-peak annotations are separate semantic types in the read-only history package. Browsing exports never activates a new polarity policy or changes frozen labels. Offline migration support is in `ecg_review_packets/legacy_labels.py`; it is not a web backend.
