#!/usr/bin/env bash
# ============================================================================
# Parallel copy of ColliderML data from EOS to scratch
# ============================================================================
#
# By default, copies the preprocessed (shard) data for p0.
# Use --raw to copy the original parquet files instead.
#
# Uses GNU parallel to run N concurrent cp jobs.
# Skips files that already exist with the correct size on scratch.
#
# Data summary:
#   p0 preprocessed: 1000 shards × ~231 MB         = ~224 GB
#   p0 raw:          particles_recorded_only         =  12 GB
#                    tracker_hits                     = 121 GB
#                    tracks (ACTS reco)               =   3 GB
#                                              Total = 136 GB
#
#   p200:            particles_recorded_only          =   9 GB
#                    tracker_hits                     = 660 GB
#                    tracks (ACTS reco)               =  14 GB
#                                              Total = 683 GB
#
#   p200_compact:    preprocessed compact shards       = TBD (est. ~200-400 GB)
#                    (no background hits, ACTS augmented)
#
# Usage:
#   ./parallel_copy_to_scratch.sh p0              # copy p0 preprocessed (default)
#   ./parallel_copy_to_scratch.sh p0 --raw        # copy p0 raw parquets instead
#   ./parallel_copy_to_scratch.sh p200            # copy p200 raw parquets
#   ./parallel_copy_to_scratch.sh p200_compact    # copy p200 compact preprocessed
#   ./parallel_copy_to_scratch.sh all             # copy p0 preprocessed + p200 raw + p200 compact
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
RAW=false
SCRATCH="/scratch"

# ── Source paths ──
P0_PREPROCESSED_SRC="/eos/project/e/end-to-end-muon-tracking/tracking/colliderml/p0/p0_preprocessed"
P0_RAW_SRC="/eos/project/e/end-to-end-muon-tracking/tracking/colliderml/p0/CERN__ColliderML-Release-1"
P200_SRC="/eos/project/n/ngt2-4/data/ColliderML-Release-1.old/data"
P200_COMPACT_SRC="/eos/project/e/end-to-end-colliderml/data/p200_preprocessed_plus_qcd"

# ── Target layout ──
P0_PREPROCESSED_DST="${SCRATCH}/colliderml/p0/p0_preprocessed"
P0_RAW_DST="${SCRATCH}/colliderml/p0"
P200_DST="${SCRATCH}/colliderml/p200"
P200_COMPACT_DST="${SCRATCH}/colliderml/p200_preprocessed_plus_qcd"

# ── Parse arguments ──
DATASET="${1:-}"
shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs|-j)  JOBS="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        --raw)      RAW=true; shift ;;
        --scratch)  SCRATCH="$2"
                    P0_PREPROCESSED_DST="${SCRATCH}/colliderml/p0/p0_preprocessed"
                    P0_RAW_DST="${SCRATCH}/colliderml/p0"
                    P200_DST="${SCRATCH}/colliderml/p200"
                    P200_COMPACT_DST="${SCRATCH}/colliderml/p200_preprocessed_plus_qcd"
                    shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATASET" ]] || [[ ! "$DATASET" =~ ^(p0|p200|p200_compact|all)$ ]]; then
    echo "Usage: $0 {p0|p200|p200_compact|all} [--jobs N] [--dry-run] [--raw] [--scratch /path]"
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

# ── Helper: copy a directory of shards (each shard is a subdirectory) ──
# Args: $1=source_dir $2=dest_dir $3=label
copy_shards_parallel() {
    local src_dir="$1"
    local dst_dir="$2"
    local label="$3"

    if [[ ! -d "$src_dir" ]]; then
        echo "  SKIP $label — source not found: $src_dir"
        return
    fi

    mkdir -p "$dst_dir"

    # Copy the manifest if it exists
    if [[ -f "${src_dir}/manifest.json" ]]; then
        cp -n "${src_dir}/manifest.json" "${dst_dir}/manifest.json" 2>/dev/null || true
    fi

    # Copy the split file if it exists (created by create_split.py)
    if [[ -f "${src_dir}/split.json" ]]; then
        cp -n "${src_dir}/split.json" "${dst_dir}/split.json" 2>/dev/null || true
    fi

    # Build list of files to copy (all files in all shard_* dirs)
    local file_list
    file_list=$(mktemp /tmp/copy_list_XXXXXX.txt)

    local total=0
    local skipped=0
    local to_copy=0
    local bytes_to_copy=0

    for shard_dir in "${src_dir}"/shard_*; do
        [[ -d "$shard_dir" ]] || continue
        local shard_name
        shard_name=$(basename "$shard_dir")
        mkdir -p "${dst_dir}/${shard_name}"

        # Handle files directly in the shard dir
        for f in "${shard_dir}"/*.npy; do
            [[ -f "$f" ]] || continue
            total=$((total + 1))
            local base
            base=$(basename "$f")
            local dst="${dst_dir}/${shard_name}/${base}"
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

            echo "$f ${dst}" >> "$file_list"
            to_copy=$((to_copy + 1))
            bytes_to_copy=$((bytes_to_copy + src_size))
        done

        # Handle files in subdirectories (e.g. selected_tracks/)
        for subdir in "${shard_dir}"/*/; do
            [[ -d "$subdir" ]] || continue
            local sub_name
            sub_name=$(basename "$subdir")
            mkdir -p "${dst_dir}/${shard_name}/${sub_name}"

            for f in "${subdir}"*.npy; do
                [[ -f "$f" ]] || continue
                total=$((total + 1))
                local base
                base=$(basename "$f")
                local dst="${dst_dir}/${shard_name}/${sub_name}/${base}"
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

                echo "$f ${dst}" >> "$file_list"
                to_copy=$((to_copy + 1))
                bytes_to_copy=$((bytes_to_copy + src_size))
            done
        done
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
        head -5 "$file_list" | while read -r src dst; do echo "      $(basename "$src")"; done
        [[ "$to_copy" -gt 5 ]] && echo "      ... and $((to_copy - 5)) more"
        rm -f "$file_list"
        return
    fi

    echo "    Copying with ${JOBS} parallel jobs..."
    local t_start
    t_start=$(date +%s)

    cat "$file_list" | parallel --jobs "$JOBS" --bar --colsep ' ' \
        "cp {1} {2} && echo '    ✓ {1/}'"

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
echo "  Mode:       $(if $RAW; then echo 'raw'; else echo 'preprocessed'; fi)"
echo "  Jobs:       $JOBS"
echo "  Scratch:    $SCRATCH"
echo "  Dry run:    $DRY_RUN"
echo ""
echo "  Scratch free: $(df -h "$SCRATCH" | tail -1 | awk '{print $4}')"
echo ""

if [[ "$DATASET" == "p0" || "$DATASET" == "all" ]]; then
    echo "── P0 ──────────────────────────────────────────────────────────────"
    if $RAW; then
        # p0 raw: parquet files
        copy_dir_parallel \
            "${P0_RAW_SRC}/ttbar_pu0_particles_recorded_only" \
            "${P0_RAW_DST}/ttbar_pu0_particles_recorded_only" \
            "particles_recorded_only (raw)"

        copy_dir_parallel \
            "${P0_RAW_SRC}/ttbar_pu0_tracker_hits/data/ttbar_pu0_tracker_hits" \
            "${P0_RAW_DST}/ttbar_pu0_tracker_hits" \
            "tracker_hits (raw)"

        copy_dir_parallel \
            "${P0_RAW_SRC}/ttbar_pu0_tracks/data/ttbar_pu0_tracks" \
            "${P0_RAW_DST}/ttbar_pu0_tracks" \
            "tracks / ACTS reco (raw)"
    else
        # p0 preprocessed: shard directories with .npy files (default)
        copy_shards_parallel \
            "${P0_PREPROCESSED_SRC}" \
            "${P0_PREPROCESSED_DST}" \
            "preprocessed shards"

        # Also copy ACTS reco tracks (needed for augmentation / evaluation)
        copy_dir_parallel \
            "${P0_RAW_SRC}/ttbar_pu0_tracks/data/ttbar_pu0_tracks" \
            "${P0_RAW_DST}/ttbar_pu0_tracks" \
            "tracks / ACTS reco"
    fi
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

    copy_dir_parallel \
        "${P200_SRC}/ttbar_pu200_tracks" \
        "${P200_DST}/ttbar_pu200_tracks" \
        "tracks / ACTS reco"
    echo ""
fi

if [[ "$DATASET" == "p200_compact" || "$DATASET" == "all" ]]; then
    echo "── P200 Compact (preprocessed) ──────────────────────────────────────"
    copy_shards_parallel \
        "${P200_COMPACT_SRC}" \
        "${P200_COMPACT_DST}" \
        "p200 compact preprocessed shards"
    echo ""
fi

echo "========================================================================"
echo "All done."
echo "========================================================================"
df -h "$SCRATCH" | tail -1 | awk '{print "  Scratch used: "$3" / "$2"  free: "$4}'
