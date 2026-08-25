# Third-party sources

`scripts/setup.sh` clones the official LTX-2 repository into `LTX-2/` and
checks out commit `780984275fd47128b02bef9b5c085404276866ee`. The checkout is
generated locally and intentionally excluded from this repository.

Do not copy model weights into this directory. Use `scripts/download_models.sh`;
downloaded files belong under the ignored `models/` directory.
