"""Optional public dataset acquisition with provenance tracking.

Downloads are always user-controlled: nothing is fetched unless the user
explicitly requests it (from the GUI data manager or the command line).
Every downloaded file is recorded in ``<data dir>/manifest.json`` with its
source URL, byte size, retrieval time, licence note and genome assembly.

Command line usage (run inside the project virtual environment)::

    python -m hidden_locus.core.datasets --list
    python -m hidden_locus.core.datasets --fetch gwas_catalog
    python -m hidden_locus.core.datasets --fetch-all
    python -m hidden_locus.core.datasets --verify
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .. import config

USER_AGENT = f"HiddenLocus/{config.APP_VERSION} (research tool; local downloads)"

ProgressCallback = Callable[[int, "int | None"], None]


class DatasetError(RuntimeError):
    """Raised when a dataset cannot be fetched or verified."""


@dataclass(frozen=True)
class DatasetSpec:
    """Description of one downloadable public resource."""

    key: str
    name: str
    purpose: str
    source: str
    license_note: str
    approx_size: str
    url: str | None = None  # direct file URL
    index_url: str | None = None  # directory listing to harvest files from
    pattern: str | None = None  # regex applied to index links
    filenames: tuple[str, ...] = ()  # expected file names on disk
    assembly: str | None = config.SUPPORTED_ASSEMBLY
    archive: str | None = None  # "zip" -> integrity-checked as zip
    required: bool = False  # required for the default IBD pipeline

    @property
    def directory(self) -> Path:
        return config.DATASETS_DIR / self.key


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="gwas_catalog",
        name="GWAS Catalog associations (ontology annotated)",
        purpose=(
            "Disease-associated index variants, reported loci, p-values and "
            "effect estimates for selecting the study loci."
        ),
        source="NHGRI-EBI GWAS Catalog",
        license_note="EMBL-EBI Terms of Use; summary statistics CC0",
        approx_size="73 MB",
        url=(
            "https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/"
            "gwas-catalog-associations_ontology-annotated-full.zip"
        ),
        archive="zip",
        required=True,
    ),
    DatasetSpec(
        key="gene_annotation",
        name="GENCODE v26 gene annotation (GRCh38, GTEx reference)",
        purpose=(
            "Gene coordinates, biotypes and transcript structure for distance "
            "features, candidate-gene windows and consequence context."
        ),
        source="GENCODE / GTEx portal reference tables",
        license_note="GENCODE data are freely available (data use policy on gencodegenes.org)",
        approx_size="134 MB",
        url=(
            "https://storage.googleapis.com/adult-gtex/references/v8/"
            "reference-tables/gencode.v26.GRCh38.genes.gtf"
        ),
        required=True,
    ),
    DatasetSpec(
        key="encode_screen",
        name="ENCODE SCREEN cCRE registry (GRCh38, all cell types)",
        purpose=(
            "Candidate cis-regulatory elements (enhancers, promoters, CTCF) "
            "for regulatory-overlap features."
        ),
        source="ENCODE SCREEN / Weng lab registry",
        license_note="ENCODE data use policy: free for unrestricted use",
        approx_size="64 MB",
        url="https://storage.googleapis.com/gcp.wenglab.org/GRCh38-cCREs.bed",
        required=True,
    ),
    DatasetSpec(
        key="gtex_eqtl",
        name="GTEx v8 cis-eQTL summary statistics (significant pairs, all tissues)",
        purpose=(
            "Tissue-specific expression-QTL evidence (whole blood, small "
            "intestine, transverse colon) used as feature evidence."
        ),
        source="GTEx Consortium (v8)",
        license_note=(
            "GTEx Portal data licence: no charges; acknowledge GTEx Portal "
            "and dbGaP phs000424"
        ),
        approx_size="1.5 GB",
        url=(
            "https://storage.googleapis.com/adult-gtex/bulk-qtl/v8/"
            "single-tissue-cis-qtl/GTEx_Analysis_v8_eQTL.tar"
        ),
        required=True,
    ),
    DatasetSpec(
        key="label_eqtl",
        name="eQTL Catalogue: BLUEPRINT monocyte eQTLs (QTD000021)",
        purpose=(
            "Independent immune-cell eQTL study used to derive positive "
            "variant-to-gene labels; kept separate from the GTEx feature "
            "evidence to avoid circular evaluation."
        ),
        source="eQTL Catalogue / BLUEPRINT (PMID 27863251)",
        license_note="CC BY 4.0",
        approx_size="2.8 GB",
        url=(
            "https://ftp.ebi.ac.uk/pub/databases/spot/eQTL/sumstats/"
            "QTS000002/QTD000021/QTD000021.all.tsv.gz"
        ),
        required=True,
    ),
    DatasetSpec(
        key="label_eqtl_permuted",
        name="eQTL Catalogue: BLUEPRINT monocyte permutation results",
        purpose="Gene-level eGene significance used to filter positive labels.",
        source="eQTL Catalogue / BLUEPRINT (PMID 27863251)",
        license_note="CC BY 4.0",
        approx_size="0.6 MB",
        url=(
            "https://ftp.ebi.ac.uk/pub/databases/spot/eQTL/sumstats/"
            "QTS000002/QTD000021/QTD000021.permuted.tsv.gz"
        ),
        required=True,
    ),
    DatasetSpec(
        key="opentargets_l2g",
        name="Open Targets locus-to-gene (L2G) scores",
        purpose=(
            "Published L2G scores used only as an external comparator "
            "baseline, never as training labels."
        ),
        source="Open Targets Platform (2026-09 release)",
        license_note="CC0 1.0",
        approx_size="70 MB",
        index_url=(
            "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/"
            "latest/output/l2g_prediction/"
        ),
        pattern=r'href="(part-[^"]+\.parquet)"',
        assembly=None,
        required=False,
    ),
    DatasetSpec(
        key="opentargets_credible_set",
        name="Open Targets GWAS credible sets (fine-mapping)",
        purpose=(
            "Credible-set lead variants and posterior probabilities used to "
            "resolve studyLocusId coordinates for the L2G comparator and to "
            "derive fine-mapping PIP features."
        ),
        source="Open Targets Platform (2026-09 release)",
        license_note="CC0 1.0",
        approx_size="2.5 GB",
        index_url=(
            "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/"
            "latest/output/credible_set/"
        ),
        pattern=r'href="(part-[^"]+\.zstd\.parquet)"',
        assembly=None,
        required=False,
    ),
    DatasetSpec(
        key="abc_model",
        name="ENCODE ABC Model enhancer-gene predictions (GRCh38)",
        purpose=(
            "Predicted enhancer-gene links from the Activity-By-Contact (ABC) "
            "Model using chromatin accessibility, H3K27ac, and Hi-C contact maps. "
            "Provides distal regulatory connections beyond linear distance."
        ),
        source="ENCODE Consortium / Kundaje Lab (ABC Model v1.1)",
        license_note="ENCODE data use policy: free for unrestricted use",
        approx_size="500 MB",
        index_url=(
            "https://storage.googleapis.com/encode-abc-model/"
            "v1.1/predictions/"
        ),
        pattern=r'href="([^"]+\.bed\.gz)"',
        assembly="GRCh38",
        required=False,
    ),
    DatasetSpec(
        key="gut_cell_atlas_eqtl",
        name="Gut Cell Atlas cell-type-specific eQTL (scRNA-seq + scATAC-seq)",
        purpose=(
            "Cell-type-specific eQTL from human gut (ileum, colon) scRNA-seq + scATAC-seq. "
            "Covers epithelial (stem, enterocyte, goblet, Paneth, Tuft), immune "
            "(T cells, B cells, macrophages, dendritic), and stromal cells. "
            "Provides IBD-relevant regulatory evidence beyond bulk GTEx."
        ),
        source="Human Gut Cell Atlas / eQTL Catalogue (Smillie et al. 2019, Martin et al. 2019)",
        license_note="CC BY 4.0 (eQTL Catalogue licence)",
        approx_size="2 GB",
        index_url=(
            "https://ftp.ebi.ac.uk/pub/databases/spot/eQTL/sumstats/"
            "GUT_CELL_ATLAS/"
        ),
        pattern=r'href="([^"]+\.tsv\.gz)"',
        assembly="GRCh38",
        required=False,
    ),
    DatasetSpec(
        key="clinvar",
        name="ClinVar GRCh38 VCF",
        purpose=(
            "Optional clinical-variant context for inspected variants; not "
            "used as regulatory variant-to-gene truth."
        ),
        source="NCBI ClinVar",
        license_note="NCBI places no restrictions on use of molecular data",
        approx_size="200 MB",
        url=(
            "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/"
            "clinvar_20261004.vcf.gz"
        ),
        assembly="GRCh38",
        required=False,
    ),
)

DATASET_INDEX: dict[str, DatasetSpec] = {d.key: d for d in DATASETS}


def get_spec(key: str) -> DatasetSpec:
    try:
        return DATASET_INDEX[key]
    except KeyError as exc:
        raise DatasetError(f"Unknown dataset key: {key}") from exc


def _request(url: str, timeout: int = 60) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def remote_size(url: str) -> "int | None":
    """Return Content-Length for *url*, or None if unavailable."""
    try:
        with urllib.request.urlopen(_request(url), timeout=timeout()) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def timeout() -> int:
    return 30


def local_files(spec: DatasetSpec) -> list[Path]:
    """Complete local files of *spec* (partial ``.part`` downloads excluded)."""
    if not spec.directory.exists():
        return []
    return sorted(
        p
        for p in spec.directory.iterdir()
        if p.is_file()
        and not p.name.endswith(".meta.json")
        and not p.name.endswith(".part")
    )


def dataset_status(spec: DatasetSpec) -> dict:
    """Return a status dict describing local availability of *spec*."""
    files = local_files(spec)
    total = sum(p.stat().st_size for p in files)
    if not files:
        state = "missing"
    elif any(p.stat().st_size == 0 for p in files):
        # A directory with only zero-byte files is incomplete, not missing.
        state = "incomplete"
    else:
        state = "present"
    verified = (spec.directory / ".verified").exists()
    return {
        "key": spec.key,
        "name": spec.name,
        "state": state,
        "bytes": total,
        "n_files": len(files),
        "verified": verified,
        "approx_size": spec.approx_size,
        "required": spec.required,
    }


def _download_file(
    url: str,
    dest: Path,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
) -> None:
    """Download *url* to *dest* with resume support and progress reporting."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    existing = part.stat().st_size if part.exists() else 0
    total: "int | None" = None

    headers = {"User-Agent": USER_AGENT}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and existing:  # already complete
            part.replace(dest)
            return
        raise DatasetError(f"HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise DatasetError(f"Network error for {url}: {exc.reason}") from exc

    with resp:
        if resp.status == 206:
            content_range = resp.headers.get("Content-Range", "")
            match = re.search(r"/(\d+)$", content_range)
            total = int(match.group(1)) if match else None
            mode = "ab"
            done = existing
        else:
            length = resp.headers.get("Content-Length")
            total = int(length) if length else None
            mode = "wb"
            done = 0
            part.unlink(missing_ok=True)

        chunk = 1 << 20
        with open(part, mode) as handle:
            while True:
                if cancel is not None and cancel.is_set():
                    raise DatasetError("Download cancelled")
                block = resp.read(chunk)
                if not block:
                    break
                handle.write(block)
                done += len(block)
                if progress is not None:
                    progress(done, total)

    if total is not None and part.stat().st_size != total:
        raise DatasetError(
            f"Incomplete download for {Path(url).name}: "
            f"{part.stat().st_size} of {total} bytes"
        )
    part.replace(dest)


def _harvest_index(spec: DatasetSpec) -> list[tuple[str, str]]:
    """Return (url, filename) pairs harvested from a directory listing."""
    if not spec.index_url or not spec.pattern:
        raise DatasetError(f"Dataset {spec.key} has no index URL/pattern")
    try:
        with urllib.request.urlopen(_request(spec.index_url), timeout=60) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DatasetError(f"Cannot read index {spec.index_url}: {exc}") from exc
    found = []
    base = spec.index_url if spec.index_url.endswith("/") else spec.index_url + "/"
    for name in re.findall(spec.pattern, html):
        found.append((urllib.parse.urljoin(base, name), name))
    if not found:
        raise DatasetError(f"No files matched pattern {spec.pattern!r}")
    return found


def fetch(
    key: str,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    force: bool = False,
) -> list[Path]:
    """Download one dataset.  Returns the list of local file paths."""
    spec = get_spec(key)
    ensure = config.ensure_data_dirs
    ensure()

    if spec.url is not None:
        dest = spec.directory / Path(spec.url).name
        if dest.exists() and dest.stat().st_size > 0 and not force:
            return [dest]
        _download_file(spec.url, dest, progress=progress, cancel=cancel)
        _record(spec, [dest])
        return [dest]

    targets = _harvest_index(spec)
    paths: list[Path] = []
    for url, name in targets:
        dest = spec.directory / name
        if dest.exists() and dest.stat().st_size > 0 and not force:
            paths.append(dest)
            continue
        _download_file(url, dest, progress=progress, cancel=cancel)
        paths.append(dest)
    _record(spec, paths)
    return paths


def _integrity_ok(spec: DatasetSpec, paths: Iterable[Path]) -> bool:
    """Format-level integrity check without re-downloading anything."""
    for path in paths:
        if path.stat().st_size == 0:
            return False
        if spec.archive == "zip":
            try:
                with zipfile.ZipFile(path) as zf:
                    if zf.testzip() is not None:
                        return False
            except zipfile.BadZipFile:
                return False
    return True


def verify(key: str, online: bool = False) -> dict:
    """Verify local files of one dataset; record provenance on success."""
    spec = get_spec(key)
    paths = local_files(spec)
    if not paths:
        return {"key": key, "state": "missing"}
    zero = [p.name for p in paths if p.stat().st_size == 0]
    if zero:
        return {"key": key, "state": "incomplete", "files": zero}

    size_ok = True
    if online and spec.url:
        remote = remote_size(spec.url)
        if remote is not None and remote != paths[0].stat().st_size:
            size_ok = False
    ok = size_ok and _integrity_ok(spec, paths)
    if ok:
        _record(spec, paths)
        (spec.directory / ".verified").write_text(
            time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8"
        )
    return {"key": key, "state": "verified" if ok else "failed", "files": [p.name for p in paths]}


def _record(spec: DatasetSpec, paths: list[Path]) -> None:
    config.ensure_data_dirs()
    manifest = load_manifest()
    manifest[spec.key] = {
        "name": spec.name,
        "source": spec.source,
        "license": spec.license_note,
        "assembly": spec.assembly,
        "files": [
            {
                "path": str(p.relative_to(config.DATASETS_DIR)),
                "bytes": p.stat().st_size,
            }
            for p in paths
        ],
        "retrieved": time.strftime("%Y-%m-%d"),
        "urls": [spec.url] if spec.url else [spec.index_url],
    }
    config.MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_manifest() -> dict:
    if config.MANIFEST_FILE.exists():
        try:
            return json.loads(config.MANIFEST_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def require(key: str) -> list[Path]:
    """Return local files for *key* or raise a helpful error."""
    spec = get_spec(key)
    paths = [p for p in local_files(spec) if p.stat().st_size > 0]
    if not paths:
        raise DatasetError(
            f"Dataset '{spec.name}' is not available locally. "
            f"Download it from {spec.url or spec.index_url} "
            "(Data manager -> Fetch, or python -m hidden_locus.core.datasets --fetch "
            f"{key})."
        )
    return paths


def statuses() -> list[dict]:
    return [dataset_status(spec) for spec in DATASETS]


def extract_gtex_tissues(tissues: Iterable[str]) -> list[Path]:
    """Extract the per-tissue eQTL files of *tissues* from the GTEx tar.

    The archive is kept as the cache of the original download; extracted
    files land in ``<data>/datasets/gtex_eqtl/extracted/``.
    """
    import tarfile

    wanted = set(tissues)
    tar_paths = [p for p in local_files(get_spec("gtex_eqtl")) if p.suffix == ".tar"]
    if not tar_paths:
        raise DatasetError("GTEx eQTL archive not downloaded")
    out_dir = get_spec("gtex_eqtl").directory / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = {
        tissue: out_dir / f"{tissue}.signif_variant_gene_pairs.txt.gz"
        for tissue in wanted
    }
    egenes = {tissue: out_dir / f"{tissue}.egenes.txt.gz" for tissue in wanted}

    def _usable(p: Path) -> bool:
        # A previously interrupted extraction can leave a truncated file;
        # only non-empty files count as present.
        return p.exists() and p.stat().st_size > 0

    missing = [t for t, p in {**pending, **egenes}.items() if not _usable(p)]
    if not missing:
        return [p for p in list(pending.values()) + list(egenes.values()) if _usable(p)]

    def _match(name: str, tissue: str, suffix: str) -> bool:
        # Archive members look like "GTEx_Analysis_v8_eQTL/Whole_Blood.v8.egenes.txt.gz".
        return name in {f"{tissue}.{suffix}", f"{tissue}.v8.{suffix}"}

    with tarfile.open(tar_paths[0], "r:") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name).name
            for tissue in missing:
                if _match(name, tissue, "signif_variant_gene_pairs.txt.gz"):
                    if not _usable(pending[tissue]):
                        _extract_member(tar, member, pending[tissue])
                elif _match(name, tissue, "egenes.txt.gz"):
                    if not _usable(egenes[tissue]):
                        _extract_member(tar, member, egenes[tissue])
            # Both files found for every requested tissue: stop scanning
            # the multi-GB archive.
            if all(_usable(p) for p in
                   list(pending.values()) + list(egenes.values())):
                break
    return [p for p in list(pending.values()) + list(egenes.values()) if _usable(p)]


def _extract_member(tar: "tarfile.TarFile", member, dest: Path) -> None:
    # Write to a temp file and rename so an interrupted extraction never
    # leaves a truncated .gz that later looks complete.
    part = dest.with_suffix(dest.suffix + ".part")
    with tar.extractfile(member) as src, open(part, "wb") as out:
        assert src is not None
        while True:
            block = src.read(1 << 20)
            if not block:
                break
            out.write(block)
    part.replace(dest)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hidden_locus.core.datasets",
        description="Optional public dataset downloads for Hidden Locus.",
    )
    parser.add_argument("--list", action="store_true", help="list known datasets")
    parser.add_argument("--fetch", metavar="KEY", help="download one dataset")
    parser.add_argument("--fetch-all", action="store_true", help="download all required datasets")
    parser.add_argument("--verify", action="store_true", help="verify local files")
    parser.add_argument("--online", action="store_true", help="verify against remote sizes")
    args = parser.parse_args(argv)

    if args.list or not (args.fetch or args.fetch_all or args.verify):
        for status in statuses():
            print(
                f"{status['key']:20s} {status['state']:10s} "
                f"{status['bytes']:>12,d} B  {status['name']}"
            )
        return 0

    keys: list[str] = []
    if args.fetch_all:
        keys = [d.key for d in DATASETS if d.required]
    elif args.fetch:
        keys = [args.fetch]

    for key in keys:
        def _progress(done: int, total: "int | None", key: str = key) -> None:
            if total:
                pct = 100.0 * done / total
                print(f"\r{key}: {done:,}/{total:,} bytes ({pct:.1f}%)", end="")
            else:
                print(f"\r{key}: {done:,} bytes", end="")

        try:
            paths = fetch(key, progress=_progress)
            print(f"\n{key}: OK -> {paths[0]}")
        except DatasetError as exc:
            print(f"\n{key}: FAILED ({exc})")
            return 1

    if args.verify:
        for key in [d.key for d in DATASETS]:
            result = verify(key, online=args.online)
            print(f"{key}: {result['state']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
