#!/usr/bin/env bash
# ============================================================================
# Parallel copy of ColliderML data from EOS to scratch
# ============================================================================
#
# Copies the filtered truth particles (particles_recorded_only) and tracker
# hits from EOS to /scratch for faster local I/O during training.
#
# Uses GNU parallel to run N concurrent cp jobs (one file per job).
# Skips files that already exist with the correct size on scratch.
#
# Data summary:
#   p0:   particles_recorded_only  300 files ×  42 MB =  12 GB
#         tracker_hits            1000 files × 129 MB = 121 GB
#                                                Total = 133 GB
#
#   p200: particles_recorded_only   51 files × 186 MB =   9 GB
#         tracker_hits            1000 files × 708 MB = 660 GB
#                                                Total = 669 GB
#
# Usage:
#   ./parallel_copy_to_scratch.sh p0              # copy p0 only
#   ./parallel_copy_to_scratch.sh p200            # copy p200 only
#   ./parallel_copy_to_scratch.sh all             # copy both
#   ./parallel_copy_to_scratch.sh p0 --jobs 30    # use 30 parallel jobs
#   ./parallel_copy_to_scratch.sh p0 --dry-run    # show what would be copied
#
# Launch in background:
#   nohup ./parallel_copy_to_scratch.sh all > /shared/tracking/logs/copy_to_scratch.log 2>&1 &
# ============================================================================

set -euo pipefail

# ── Defaults ──
JOBS=20
DRY_RUN=false
SCRATCH="/scratch"

# ── Source paths ──
P0_SRC="/eos/project/e/end-to-end-muon-tracking/tracking/colliderml/p0/CERN__ColliderML-Release-1"
P200_SRC="/eos/project/n/ngt2-4/data/ColliderML-Release-1.old/data"

# ── Target layout (mirrors source naming) ──
P0_DST="${SCRATCH}/colliderml/p0"
P200_DST="${SCRATCH}/colliderml/p200"

# ── Parse arguments ──
DATASET="${1:-}"
shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs|-j)  JOBS="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        --scratch)  SCRATCH="$2"; P0_DST="${SCRATCH}/colliderml/p0"; P200_DST="${SCRATCH}/colliderml/p200"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATASET" ]] || [[ ! "$DATASET" =~ ^(p0|p200|all)$ ]]; then
    echo "Usage: $0 {p0|p200|all} [--jobs N] [--dry-run] [--scratch /path]"
    exit 1
fi

# ── Helper: copy a single directory of parquets in parallel ──
# Args: $1=source_dir $2=dest_dir $3=label
copy_dir_parallel() {
    local src_dir="$1"
    local dst_dir="$2"
    local label="$3"

    if [[ ! -d "$src_dir" ]]; then
        echo "  SKIP $label — source not found: $src_dir"
        return
    fi

    mkdir -p "$dst_dir"

    # Collect files to copy (skip existing with matching size)
    local file_list
    file_list=$(mktemp /tmp/copy_list_XXXXXX.txt)

    local total=0
    local skipped=0
    local to_copy=0
    local bytes_to_copy=0

    for f in "${src_dir}"/*.parquet; do
        [[ -f "$f" ]] || continue
        total=$((total + 1))
        local base
        base=$(basename "$f")
        local dst="${dst_dir}/${base}"
        local src_size
        src_size=$(stat --format="%s" "$f")

        if [[ -f "$dst" ]]; then
            local dst_size
            dst_size=$(stat --format="%s" "$dst")
            if [[ "$src_size" == "$dst_size" ]]; then
                skipped=$((skipped + 1))
                continue
            fi
        fi

        echo "$f" >> "$file_list"
        to_copy=$((to_copy + 1))
        bytes_to_copy=$((bytes_to_copy + src_size))
    done

    local gb_to_copy
    gb_to_copy=$(awk "BEGIN {printf \"%.1f\", ${bytes_to_copy}/1e9}")

    echo "  $label: ${total} files total, ${skipped} already on scratch, ${to_copy} to copy (${gb_to_copy} GB)"

    if [[ "$to_copy" -eq 0 ]]; then
        rm -f "$file_list"
        return
    fi

    if $DRY_RUN; then
        echo "    [DRY RUN] Would copy ${to_copy} files with ${JOBS} parallel jobs"
        head -5 "$file_list" | while read -r f; do echo "      $(basename "$f")"; done
        [[ "$to_copy" -gt 5 ]] && echo "      ... and $((to_copy - 5)) more"
        rm -f "$file_list"
        return
    fi

    echo "    Copying with ${JOBS} parallel jobs..."
    local t_start
    t_start=$(date +%s)

    # Use GNU parallel: copy file, print filename when done
    cat "$file_list" | parallel --jobs "$JOBS" --bar \
        "cp {} ${dst_dir}/\$(basename {}) && echo '    ✓ \$(basename {})'"

    local t_end
    t_end=$(date +%s)
    local elapsed=$((t_end - t_start))
    local rate
    rate=$(awk "BEGIN {r=${bytes_to_copy}/${elapsed:-1}/1e6; printf \"%.0f\", r}")
    echo "    Done: ${to_copy} files in ${elapsed}s (${rate} MB/s)"
    echo ""

    rm -f "$file_list"
}

# ── Main ──
echo "========================================================================"
echo "Parallel Copy: EOS → scratch"
echo "========================================================================"
echo "  Dataset:    $DATASET"
echo "  Jobs:       $JOBS"
echo "  Scratch:    $SCRATCH"
echo "  Dry run:    $DRY_RUN"
echo ""
echo "  Scratch free: $(df -h "$SCRATCH" | tail -1 | awk '{print $4}')"
echo ""

if [[ "$DATASET" == "p0" || "$DATASET" == "all" ]]; then
    echo "── P0 ──────────────────────────────────────────────────────────────"
    # p0 recorded_only: flat layout
    copy_dir_parallel \
        "${P0_SRC}/ttbar_pu0_particles_recorded_only" \
        "${P0_DST}/ttbar_pu0_particles_recorded_only" \
        "particles_recorded_only"

    # p0 tracker_hits: nested HF layout — parquets are one level deeper
    copy_dir_parallel \
        "${P0_SRC}/ttbar_pu0_tracker_hits/data/ttbar_pu0_tracker_hits" \
        "${P0_DST}/ttbar_pu0_tracker_hits" \
        "tracker_hits"
    echo ""
fi

if [[ "$DATASET" == "p200" || "$DATASET" == "all" ]]; then
    echo "── P200 ─────────────────────────────────────────────────────────────"
    # p200: flat layout
    copy_dir_parallel \
        "${P200_SRC}/ttbar_pu200_particles_recorded_only" \
        "${P200_DST}/ttbar_pu200_particles_recorded_only" \
        "particles_recorded_only"

    copy_dir_parallel \
        "${P200_SRC}/ttbar_pu200_tracker_hits" \
        "${P200_DST}/ttbar_pu200_tracker_hits" \
        "tracker_hits"
    echo ""
fi

echo "========================================================================"
echo "All done."
echo "========================================================================"
df -h "$SCRATCH" | tail -1 | awk '{print "  Scratch used: "$3" / "$2"  free: "$4}'
