#!/bin/bash

_syncall() {
    local hosts=( $(syncall --list-hosts) )
    local cur_word="${COMP_WORDS[COMP_CWORD]}"

    # Completions for everything else
    case $cur_word in
        *)  COMPREPLY=( $(compgen -W "${hosts[*]}" -- ${cur_word}) ); return;;
    esac
}

complete -F _syncall syncall
