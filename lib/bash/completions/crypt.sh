#!/bin/bash

_crypt() {
    local opts=(open close create)
    local cur_word="${COMP_WORDS[COMP_CWORD]}"
    local prev_word="${COMP_WORDS[COMP_CWORD - 1]}"

    case "$prev_word" in
        open|close|create) COMPREPLY=( $(compgen -f ${cur_word}) ); return;;
    esac

    COMPREPLY=( $(compgen -W "${opts[*]}" -- ${cur_word}) )
}
complete -F {_,}crypt
