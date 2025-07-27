#!/bin/awk -f

function sep() {
    print "========================="
}

BEGIN {
    sep()
    printf "Beginning: argc=%s\n", ARGC

    for (i=0; i<ARGC; i++){
        printf " * argv[%s]=%s\n", i, ARGV[i]
        ARGV[i]=""
    }
    num = 0

    system("cowsay get ready")

    sep()
}

!/^\s*#/ {
    printf "%03i: %s\n", num, $0
    num++
}

END {
    sep()
    printf "Ending: num=%s\n", num
    sep()
}
