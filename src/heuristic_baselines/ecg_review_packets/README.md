# Offline ECG packet helpers

This directory retains waveform/QC and packet-building helpers used by historical research. It contains no HTTP service, login, shared annotation database or browser frontend. The sole review website is [`review_platform`](../../../review_platform/README.md), deployable to Cloudflare Pages.

Historical task exports and waveform packets remain private, outside Git. Use the static website to open reviewer ZIPs or the read-only history ZIP. These tools do not modify frozen ECG ground truth.
