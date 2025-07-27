; ================================================
;
; BSF: Bit Scan Forward
;
; find the index of the first nonzero bit.
; in a r16/32/64 or an m16/32/64.
; i.e., a [16, 32, or 64 bit] [register or memory].
;
; https://www.felixcloutier.com/x86/bsf
;
; ================================================

; compile with this shell function
;
; nasmcompilecrt () { 
;   file="$1"
;   stem="${file%.*}"
;   nasm -f elf64 -o "${stem}.o" "${file}" &&
;   ld -o "$stem" -dynamic-linker /lib/ld-linux-x86-64.so.2 /lib/crt1.o /lib/crti.o "${stem}.o" -lc /lib/crtn.o &&
;   rm "${stem}.o"
; };

; syscall numbers
; see /usr/include/asm/unistd_64.h
%define sys_write       1
%define sys_exit        60

; file descriptor names
%define stdin           0
%define stdout          1
%define stderr          2

; other macros
%define EXIT_SUCCESS    0


section .data

    fmtstring: db "The first set bit is in index: %d", 0x0a, 0x00


section .text
extern printf

; if we're linking in the c runtime, crti.o already contains _start,
; so we have to define main instead.
global main
main:
    mov rax, 0b0000000000010000 ; 5th bit from LSB side is set (4th with zero indexing)
    bsf rsi, rax                ; use the BSF instruction to find the first set bit's index.
    mov rdi, fmtstring          ; fmtstring will be the first argument to printf
    xor rax, rax                ; zero rax to indicate no SSE registers used
    call printf

    mov rax, 0b0000000000000100 ; 3rd bit from LSB side is set (2nd with zero indexing)
    bsf rsi, rax
    mov rdi, fmtstring
    xor rax, rax
    call printf

exit:
    mov     rax, sys_exit
    mov     rdi, EXIT_SUCCESS
    syscall

