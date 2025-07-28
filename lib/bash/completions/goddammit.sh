#!/bin/bash

# bash completion for my goddammit command

_goddammit() {
    local options=( $(goddammit --show) )
    local cur_word="${COMP_WORDS[COMP_CWORD]}"
    local prev_word="${COMP_WORDS[COMP_CWORD-1]}"
    case $cur_word in
        *) COMPREPLY=( $(compgen -W "${options[*]}"    -- ${cur_word}) );;
    esac
}

complete -F _goddammit goddammit
