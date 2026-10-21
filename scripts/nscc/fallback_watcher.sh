#!/bin/bash
set -uo pipefail

# Login-node watcher that makes sure every NSCC run's final model reaches RoboColosseum/<job>.
#
# For each "<pbs job id>:<job name>" argument it waits until PBS reports the job as finished
# (qstat -x job_state F; a failed PBS query just means "check again later"), then runs the same
# idempotent script the job itself uses, from a single process on the login node:
#   lerobot (molmoact2 / pi05 / groot) -> scripts/nscc/push_lerobot_final.sh <job>
#   lingbot                            -> scripts/lingbot/upload_droid_put_marker_pen_10ep_nscc.sh 16150
#                                         (also copies the TensorBoard curves into W&B)
#   g05                                -> EXPORT_STRIP=false scripts/g05/upload_droid_put_marker_pen_10ep_nscc.sh
#                                         (copies the inference-only checkpoint instead of loading it)
# Each script does nothing if the job's own upload already completed. A job leaves the list when its
# script exits 0, when a lerobot job has no final checkpoint (exit 2), or after 5 attempts.
#
# Start it detached so it survives the login session:
#   setsid nohup bash scripts/nscc/fallback_watcher.sh 163007.pbs201:molmoact2-droid-put-marker-pen-10ep-nscc ... \
#       >> logs/fallback_watcher.log 2>&1 < /dev/null &

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
cd "$ROBO_ROOT"

declare -A NAME ATTEMPTS
for arg in "$@"; do
    NAME[${arg%%:*}]="${arg#*:}"
    ATTEMPTS[${arg%%:*}]=0
done
log() { echo "$(date '+%F %T') $*"; }
log "watcher pid $$ watching: $*"

while [ ${#NAME[@]} -gt 0 ]; do
    for id in "${!NAME[@]}"; do
        job="${NAME[$id]}"
        state="$(qstat -x -f "$id" 2>/dev/null | awk -F' = ' '/^ *job_state = /{print $2; exit}')"
        [ "$state" = "F" ] || continue
        ATTEMPTS[$id]=$(( ATTEMPTS[$id] + 1 ))
        log "$id ($job) finished; fallback attempt ${ATTEMPTS[$id]}"
        case "$job" in
            lingbot-*) cmd=(bash scripts/lingbot/upload_droid_put_marker_pen_10ep_nscc.sh 16150) ;;
            g05-*)     cmd=(env EXPORT_STRIP=false bash scripts/g05/upload_droid_put_marker_pen_10ep_nscc.sh) ;;
            *)         cmd=(bash scripts/nscc/push_lerobot_final.sh "$job") ;;
        esac
        mkdir -p "logs/$job"
        "${cmd[@]}" >> "logs/$job/fallback_push.log" 2>&1
        rc=$?
        log "  -> exit $rc (see logs/$job/fallback_push.log)"
        if [ "$rc" -eq 0 ] || { [ "$rc" -eq 2 ] && [[ "$job" != lingbot-* && "$job" != g05-* ]]; } \
            || [ "${ATTEMPTS[$id]}" -ge 5 ]; then
            unset "NAME[$id]"
        fi
    done
    [ ${#NAME[@]} -gt 0 ] && sleep 300
done
log "all jobs handled"
