#!/bin/bash

_erchomai_complete() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Pull Bible filenames from $ERCHOMAI
    local dir="${ERCHOMAI}"
    if [[ -z "$dir" ]]; then
        COMPREPLY=()
        return
    fi

    local files=()
    if [[ -d "$dir" ]]; then
        while IFS= read -r f; do
            files+=("$f")
        done < <(ls "$dir"/*.json 2>/dev/null | xargs -n1 basename)
    fi

    case "$prev" in
        --list)
            COMPREPLY=()
            ;;
        *)
            COMPREPLY=( $(compgen -W "${files[*]}" -- "$cur") )
            ;;
    esac
}

complete -F _erchomai_complete ερχο
complete -F _erchomai_complete ερχομαι
complete -F _erchomai_complete erchomai
complete -F _erchomai_complete come
