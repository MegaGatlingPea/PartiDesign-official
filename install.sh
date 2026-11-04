#!/usr/bin/env bash
# The two shipped profiles remain the only package definitions.
set -uo pipefail

usage() {
    cat <<'HELP'
Usage: bash install.sh [--cuda 12|13] [--name NAME] [--manager auto|mamba|conda]
                       [--update | --pip-only] [--plan]

Install PartiDesign with visible native-package and pip progress.
  --cuda       CUDA profile (default: 12)
  --name       Override the profile's environment name
  --manager    auto prefers Mamba on PATH, then CONDA_EXE/Conda (default: auto)
  --update     Update an existing environment before pip
  --pip-only   Retry pip after the native stage completed
  --plan       Print commands and ordered pip entries without installing or writing
HELP
}
argument_error() { printf 'Error: %s\n' "$*" >&2; exit 2; }
cuda=12 name='' manager=auto mode=create plan=0 name_set=0
while (($#)); do
    case $1 in
        --cuda|--name|--manager)
            (($# >= 2)) || argument_error "$1 requires a value"
            case $1 in --cuda) cuda=$2;; --name) name=$2; name_set=1;; --manager) manager=$2;; esac
            shift 2;;
        --update|--pip-only)
            next_mode=update; [[ $1 == --pip-only ]] && next_mode=pip-only
            [[ $mode == create || $mode == "$next_mode" ]] || argument_error '--update and --pip-only are mutually exclusive'
            mode=$next_mode; shift;;
        --plan) plan=1; shift;;
        -h|--help) usage; exit 0;;
        *) argument_error "unknown argument: $1";;
    esac
done
case $cuda in 12|13) ;; *) argument_error '--cuda must be 12 or 13';; esac
case $manager in auto|mamba|conda) ;; *) argument_error '--manager must be auto, mamba, or conda';; esac
source_dir=.; [[ ${BASH_SOURCE[0]} == */* ]] && source_dir=${BASH_SOURCE[0]%/*}
repo_root=$(CDPATH= cd -- "$source_dir" && pwd -P) || exit 1
cd -- "$repo_root" || exit 1
profile="partidesign_cu${cuda}.yaml"

# Only the controlled unquoted, single-line scalar-list layout is supported.
# Keep native lines verbatim; emit each pip list item once in source order.
profile_data() {
    awk -v output="$1" '
    function bad() {
        printf "%s:%d: unsupported profile layout; expected name, channels, native dependencies, then one terminal pip list of unquoted single-line entries\n", FILENAME, FNR > "/dev/stderr"
        invalid=1; exit 2
    }
    function scalar(value) {
        if (value == "" || value ~ /^[[:space:]]|[[:space:]]$/ ||
            index("\"\047[{&*!|>@`%", substr(value,1,1)) ||
            value ~ /:([[:space:]]|$)|[[:space:]]#/) bad()
    }
    {
        if (index($0,"\t")) bad()
        if ($0 ~ /^[[:space:]]*($|#)/) {
            if (section != "pip" && output == "native") print
            next
        }
        if (section == "" && name == "" && $0 ~ /^name: [A-Za-z0-9_.-]+$/) name=substr($0,7)
        else if (section == "" && name != "" && $0 == "channels:") section="channels"
        else if (section == "channels" && channels && $0 == "dependencies:") section="native"
        else if (section == "channels" && index($0,"  - ") == 1) { scalar(substr($0,5)); channels++ }
        else if (section == "native" && $0 == "  - pip:") section="pip"
        else if (section == "native" && index($0,"  - ") == 1) { scalar(substr($0,5)); native_count++ }
        else if (section == "pip" && index($0,"      - ") == 1) {
            value=substr($0,9); scalar(value); pip_count++
            if (output == "pip") print value
        } else bad()
        if (section != "pip" && output == "native") print
    }
    END {
        if (invalid) exit 2
        if (name == "" || !native_count || !pip_count || section != "pip") bad()
        if (output == "meta") print name, native_count, pip_count
    }' "$profile"
}
metadata=$(profile_data meta) || exit $?
read -r profile_name native_count pip_count <<< "$metadata"
((name_set)) || name=$profile_name
case $name in ''|-*|*/*|*\\*|*:*|*[[:space:]]*) argument_error '--name must be an environment name without whitespace or path separators';; esac
if [[ $manager == auto ]]; then
    if command -v mamba >/dev/null 2>&1; then manager=mamba; else manager=conda; fi
fi
manager_exe=$manager
[[ $manager == conda ]] && manager_exe=${CONDA_EXE:-conda}
print_command() { printf '%q ' "$@"; printf '\n'; }
build_commands() {
    if [[ $manager == mamba && $mode == create ]]; then
        native_command=("$manager_exe" create -n "$name" -f "$native_file")
    else
        native_mode=create; [[ $mode == update ]] && native_mode=update
        native_command=("$manager_exe" env "$native_mode" -n "$name" -f "$native_file")
    fi
    pip_command=("$manager_exe" run)
    [[ $manager == conda ]] && pip_command+=(--no-capture-output)
    pip_command+=(-n "$name" pip install --progress-bar on -r "$requirements_file")
}
printf 'Profile: %s\nEnvironment: %s\nManager: %s\nWorking directory: %s\nNative packages: %s; pip entries: %s\n' \
    "$profile" "$name" "$manager" "$repo_root" "$native_count" "$pip_count"
if ((plan)); then
    native_file='<native.yaml>' requirements_file='<requirements.txt>'
    build_commands
    if [[ $mode == pip-only ]]; then printf 'Conda/Mamba native: skipped (--pip-only)\n'
    else printf 'Native command: '; print_command "${native_command[@]}"; fi
    printf 'Python packages: '; print_command "${pip_command[@]}"
    printf 'Pip entries (one requirements transaction, in profile order):\n'
    profile_data pip
    exit $?
fi

stages=('Conda/Mamba native' 'Python packages')
states=(pending pending) starts=(-1 -1) elapsed=('-' '-')
current=0 total_start=$SECONDS temp_dir=''
show_table() {
    local i duration
    printf '\n%-20s %-13s %10s\n' Stage Status Elapsed
    for i in 0 1; do
        duration=${elapsed[i]}
        [[ ${states[i]} == running ]] && duration="$((SECONDS - starts[i]))s"
        printf '%-20s %-13s %10s\n' "${stages[i]}" "${states[i]}" "$duration"
    done
    printf '\n'
}
change_stage() {
    local index=$1 status=$2
    if [[ $status == running ]]; then starts[index]=$SECONDS
    elif ((starts[index] >= 0)); then elapsed[index]="$((SECONDS - starts[index]))s"; fi
    states[index]=$status
    show_table
}
finish() {
    local code=$1 i
    ((current < 2)) || current=1
    trap '' INT
    if ((code)); then
        if ((code == 130)); then change_stage "$current" interrupted
        else change_stage "$current" failed; fi
        for i in 0 1; do [[ ${states[i]} != pending ]] || change_stage "$i" skipped; done
    fi
    printf 'Total elapsed: %ss\n' "$((SECONDS - total_start))"
    if ((code)); then
        printf 'Retry from %s:\n  ' "$repo_root"
        retry=(bash install.sh --cuda "$cuda" --name "$name" --manager "$manager")
        if ((current == 1)); then retry+=(--pip-only)
        elif [[ $mode == update ]]; then retry+=(--update); fi
        print_command "${retry[@]}"
        if ((current == 0)) && [[ $mode == create ]]; then
            printf 'If the environment was already created, add --update to retry.\n'
        fi
    fi
    exit "$code"
}
cleanup() { if [[ -n $temp_dir ]]; then rm -rf -- "$temp_dir"; fi; }
trap cleanup EXIT
# Foreground children inherit our terminal/process group and receive Ctrl-C too.
# Bash runs this trap after the foreground child has exited and been reaped.
trap 'finish 130' INT
show_table
if [[ $mode == pip-only ]]; then current=1; change_stage 0 skipped; fi
if ! command -v "$manager_exe" >/dev/null 2>&1; then
    printf '%s not found; put it on PATH or set CONDA_EXE for Conda.\n' "$manager_exe" >&2
    finish 127
fi
temp_dir=$(mktemp -d "${TMPDIR:-/tmp}/partidesign-install.XXXXXX") || finish $?
native_file="$temp_dir/native.yaml" requirements_file="$temp_dir/requirements.txt"
if [[ $mode != pip-only ]]; then profile_data native > "$native_file" || finish $?; fi
profile_data pip > "$requirements_file" || finish $?
build_commands
while ((current < 2)); do
    change_stage "$current" running
    if ((current == 0)); then command_args=("${native_command[@]}")
    else command_args=("${pip_command[@]}"); fi
    printf '$ '; print_command "${command_args[@]}"
    if "${command_args[@]}"; then change_stage "$current" complete
    else code=$?; printf '%s exited with code %s.\n' "${stages[current]}" "$code" >&2; finish "$code"; fi
    ((current += 1))
done
finish 0
