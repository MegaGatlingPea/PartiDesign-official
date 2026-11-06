#!/usr/bin/env bash
# Conda profiles supply native packages; uv uses the committed Python lock.
set -uo pipefail

usage() {
    cat <<'HELP'
Usage: bash install.sh [--installer conda|uv] [--cuda 12|13]
                       [--name NAME] [--manager auto|mamba|conda]
                       [--venv PATH] [--native-prefix PATH]
                       [--update | --pip-only] [--plan]

Install PartiDesign with visible native-package and Python-package progress.
  --installer  Conda/pip or uv with its own native Conda prefix (default: uv)
  --cuda       CUDA profile (default: 12)
  --name       Override the Conda environment name (Conda installer only)
  --manager    auto prefers Mamba on PATH, then CONDA_EXE/Conda (default: auto)
  --venv       uv virtual environment (default: checkout/.venv)
  --native-prefix  uv native environment (default: checkout/.cache/environments/native-cu12 or native-cu13)
  --update     Update an existing environment; uv targets must be a recorded pair
  --pip-only   Retry only Python packages after native installation (uv: create/check venv, then sync)
  --plan       Print commands and ordered pip entries without installing or writing
HELP
}
argument_error() { printf 'Error: %s\n' "$*" >&2; exit 2; }
cuda=12 name='' manager=auto mode=create plan=0 name_set=0 installer=uv
venv='' native_prefix='' venv_set=0 native_prefix_set=0 invocation_dir=$PWD
while (($#)); do
    case $1 in
        --cuda|--name|--manager|--installer|--venv|--native-prefix)
            (($# >= 2)) || argument_error "$1 requires a value"
            case $1 in
                --cuda) cuda=$2;; --name) name=$2; name_set=1;;
                --manager) manager=$2;; --installer) installer=$2;;
                --venv) venv=$2; venv_set=1;;
                --native-prefix) native_prefix=$2; native_prefix_set=1;;
            esac
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
case $installer in conda|uv) ;; *) argument_error '--installer must be conda or uv';; esac
if [[ $installer == conda ]]; then
    (( !venv_set && !native_prefix_set )) || argument_error '--venv and --native-prefix require --installer uv'
else
    (( !name_set )) || argument_error '--name is only available with --installer conda; use --native-prefix for uv'
fi
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
        else if (section == "native" && index($0,"  - ") == 1) {
            value=substr($0,5); scalar(value); native_count++
            if (output == "python" && value ~ /^python=[0-9]+\.[0-9]+\.[0-9]+$/) print substr(value,8)
        }
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
uv_exe=${UV:-uv}
if [[ $installer == uv ]]; then
    command -v realpath >/dev/null 2>&1 || argument_error 'realpath is required for the uv installer'
    absolute_target() {
        local value=$1
        [[ -n $value && $value != *$'\n'* && $value != *$'\r'* ]] || argument_error 'installation paths must be nonempty single-line paths'
        [[ $value == /* ]] || value="$invocation_dir/$value"
        realpath -m -- "$value"
    }
    ((venv_set)) || venv="$repo_root/.venv"
    ((native_prefix_set)) || native_prefix="$repo_root/.cache/environments/native-cu$cuda"
    venv=$(absolute_target "$venv") || exit $?
    native_prefix=$(absolute_target "$native_prefix") || exit $?
    case "$venv/" in "$native_prefix/"*) argument_error '--venv must not equal or be inside --native-prefix';; esac
    case "$native_prefix/" in "$venv/"*) argument_error '--native-prefix must not be inside --venv';; esac
    pair_file="$native_prefix.partidesign-install"
    helper="$repo_root/partidesign/_installation.py"
    expected_python=$(profile_data python) || exit $?
    [[ $expected_python =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || argument_error 'the native profile must pin one exact Python version'
fi
print_command() { printf '%q ' "$@"; printf '\n'; }
build_commands() {
    local target_flags
    if [[ $installer == uv ]]; then target_flags=(-p "$native_prefix")
    else target_flags=(-n "$name"); fi
    if [[ $manager == mamba && $mode == create ]]; then
        native_command=("$manager_exe" create "${target_flags[@]}" -f "$native_file")
    else
        native_mode=create; [[ $mode == update ]] && native_mode=update
        native_command=("$manager_exe" env "$native_mode" "${target_flags[@]}" -f "$native_file")
    fi
    pip_command=("$manager_exe" run)
    [[ $manager == conda ]] && pip_command+=(--no-capture-output)
    if [[ $installer == uv ]]; then
        pip_command+=(-p "$native_prefix" env "UV_PROJECT_ENVIRONMENT=$venv" UV_PYTHON_DOWNLOADS=never PYTHONNOUSERSITE=1
                      "$uv_exe" sync --locked --python "$native_prefix/bin/python" --no-python-downloads --extra "cu$cuda" --extra full)
        venv_command=("$uv_exe" venv --python "$native_prefix/bin/python" --system-site-packages --no-python-downloads --allow-existing "$venv")
        native_check=("$native_prefix/bin/python" -B -I "$helper" native --native-prefix "$native_prefix" --python-version "$expected_python" --cuda "$cuda")
        venv_check=("$native_prefix/bin/python" -B -I "$helper" venv --native-prefix "$native_prefix" --venv "$venv" --python-version "$expected_python")
    else
        pip_command+=(-n "$name" pip install --progress-bar on -r "$requirements_file")
    fi
}
environment=$name; [[ $installer != uv ]] || environment=$venv
printf 'Installer: %s\n' "$installer"
printf 'Profile: %s\nEnvironment: %s\nManager: %s\nWorking directory: %s\nNative packages: %s; pip entries: %s\n' \
    "$profile" "$environment" "$manager" "$repo_root" "$native_count" "$pip_count"
if ((plan)); then
    native_file='<native.yaml>' requirements_file='<requirements.txt>'
    build_commands
    if [[ $mode == pip-only ]]; then printf 'Conda/Mamba native: skipped (--pip-only)\n'
    else printf 'Native command: '; print_command "${native_command[@]}"; fi
    if [[ $installer == uv ]]; then
        printf 'Native prefix: %s\nVirtual environment: %s\nPairing state: %s (actual installation only)\n' "$native_prefix" "$venv" "$pair_file"
        printf 'Native validation: '; print_command "${native_check[@]}"
        printf 'Virtual environment command (only when missing/incomplete): '; print_command "${venv_command[@]}"
        printf 'Virtual environment validation: '; print_command "${venv_check[@]}"
        printf 'Python packages: '; print_command "${pip_command[@]}"
        printf 'Python packages come from uv.lock (cu%s + full); --pip-only skips native packages and resumes this layer.\n' "$cuda"
        exit 0
    fi
    printf 'Python packages: '; print_command "${pip_command[@]}"
    printf 'Pip entries (one requirements transaction, in profile order):\n'
    profile_data pip
    exit $?
fi

stages=('Conda/Mamba native' 'Python packages')
[[ $installer != uv ]] || stages=('Conda/Mamba native' 'uv virtual environment' 'uv Python packages')
states=() starts=() elapsed=()
for i in "${!stages[@]}"; do states+=(pending); starts+=(-1); elapsed+=('-'); done
stage_count=${#stages[@]}
current=0 total_start=$SECONDS temp_dir=''
show_table() {
    local i duration
    printf '\n%-20s %-13s %10s\n' Stage Status Elapsed
    for i in "${!stages[@]}"; do
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
    ((current < stage_count)) || current=$((stage_count - 1))
    trap '' INT
    if ((code)); then
        if ((code == 130)); then change_stage "$current" interrupted
        else change_stage "$current" failed; fi
        for i in "${!stages[@]}"; do [[ ${states[i]} != pending ]] || change_stage "$i" skipped; done
    fi
    printf 'Total elapsed: %ss\n' "$((SECONDS - total_start))"
    if ((code)); then
        printf 'Retry from %s:\n  ' "$repo_root"
        if [[ $installer == uv ]]; then
            retry=(bash install.sh --installer uv --cuda "$cuda" --venv "$venv" --native-prefix "$native_prefix" --manager "$manager")
            if ((current > 0)); then retry+=(--pip-only)
            elif [[ -f $pair_file ]]; then retry+=(--update); fi
        else
            retry=(bash install.sh --installer conda --cuda "$cuda" --name "$name" --manager "$manager")
            if ((current == 1)); then retry+=(--pip-only)
            elif [[ $mode == update ]]; then retry+=(--update); fi
        fi
        print_command "${retry[@]}"
        if ((current == 0)) && [[ $mode == create && $installer == conda ]]; then
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
if [[ $installer == uv ]]; then
    if ! uv_exe=$(command -v "$uv_exe"); then printf 'uv not found; install uv and put it on PATH.\n' >&2; finish 127; fi
    [[ $uv_exe == /* ]] || uv_exe="$repo_root/$uv_exe"
    [[ -f $repo_root/uv.lock ]] || { printf 'uv.lock is missing; use a complete locked checkout.\n' >&2; finish 2; }
    [[ -f $helper ]] || { printf 'The installation helper is missing: %s\n' "$helper" >&2; finish 2; }
    read_pairing() {
        local records=()
        [[ -f $pair_file && ! -L $pair_file ]] || { printf 'No installer pairing state at %s; choose unused targets.\n' "$pair_file" >&2; return 2; }
        mapfile -t records < "$pair_file"
        [[ ${#records[@]} == 5 && ${records[0]} == partidesign-uv-install-v1 && ${records[1]} == "$cuda" && ${records[2]} == "$native_prefix" && ${records[3]} == "$venv" ]] || {
            printf 'Native/virtual environment targets or CUDA profile do not match the recorded installation; choose a separate pair.\n' >&2; return 2;
        }
    }
    if [[ $mode == create ]]; then
        if [[ -e $native_prefix || -L $native_prefix || -e $venv || -L $venv || -e $pair_file || -L $pair_file ]]; then
            printf 'An installation target already exists. Use --update for its recorded pair, or choose unused --venv and --native-prefix paths.\n' >&2
            finish 2
        fi
    else
        read_pairing || finish $?
        if [[ $mode == pip-only && ! -x $native_prefix/bin/python ]]; then
            printf 'Native Python is unavailable; use --update to finish the native stage first.\n' >&2; finish 2
        fi
        if [[ -x $native_prefix/bin/python ]]; then
            native_file='<native.yaml>' requirements_file='<requirements.txt>'; build_commands
            "${native_check[@]}" --allow-incomplete || finish $?
            "${venv_check[@]}" --allow-incomplete || finish $?
        elif [[ -f $venv/pyvenv.cfg ]]; then
            printf 'Cannot validate an existing virtual environment without its native Python; choose a separate installation pair.\n' >&2; finish 2
        fi
    fi
fi
temp_dir=$(mktemp -d "${TMPDIR:-/tmp}/partidesign-install.XXXXXX") || finish $?
native_file="$temp_dir/native.yaml" requirements_file="$temp_dir/requirements.txt"
if [[ $mode != pip-only ]]; then profile_data native > "$native_file" || finish $?; fi
profile_data pip > "$requirements_file" || finish $?
build_commands
if [[ $installer == uv && $mode == create ]]; then
    mkdir -p -- "${native_prefix%/*}" || finish $?
    if ! (umask 077; set -o noclobber; printf '%s\n' partidesign-uv-install-v1 "$cuda" "$native_prefix" "$venv" "$repo_root" > "$pair_file"); then
        printf 'Could not reserve the installation pair: %s\n' "$pair_file" >&2; finish 2
    fi
fi
while ((current < stage_count)); do
    change_stage "$current" running
    if ((current == 0)); then command_args=("${native_command[@]}")
    elif [[ $installer == uv ]] && ((current == 1)); then
        "${native_check[@]}" || finish $?
        if [[ -f $venv/pyvenv.cfg ]]; then
            "${venv_check[@]}" || finish $?
            change_stage "$current" skipped
            ((current += 1)); continue
        fi
        command_args=("${venv_command[@]}")
    else
        [[ $installer != uv ]] || "${venv_check[@]}" || finish $?
        command_args=("${pip_command[@]}")
    fi
    printf '$ '; print_command "${command_args[@]}"
    if "${command_args[@]}"; then
        if [[ $installer == uv ]] && ((current == 0)); then "${native_check[@]}" || finish $?; fi
        if [[ $installer == uv ]] && ((current == 1)); then "${venv_check[@]}" || finish $?; fi
        change_stage "$current" complete
    else code=$?; printf '%s exited with code %s.\n' "${stages[current]}" "$code" >&2; finish "$code"; fi
    ((current += 1))
done
finish 0
