; compile with this shell function
;
; nasmcompilecrt () { 
;   file="$1"
;   stem="${file%.*}"
;   nasm -f elf64 -o "${stem}.o" "${file}" &&
;   ld -o "$stem" -dynamic-linker /lib/ld-linux-x86-64.so.2 /lib/crt1.o /lib/crti.o "${stem}.o" -lc /lib/crtn.o &&
;   rm "${stem}.o"
; };

; calling convention for c libraries: user space (system-v abi)
; -------------------------------------------------------------------
; pass first six arguments in:
; rdi, rsi, rdx, rcx, r8, r9
; any remaining arguments are passed on the stack.
;
; calling convention for syscalls: kernel space (x86_64)
; -------------------------------
; rax: syscall number
; pass first six arguments in:
; rdi, rsi, rdx, r10, r8, r9
; (same as system-v, with rcx replaced by r10)
;
;
; see also:
; https://wiki.osdev.org/System_V_ABI
; https://wiki.osdev.org/Executable_and_Linkable_Format

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

    msg: db "Hello, world!", 0x00

    b: db " there.", 0
    a: db "Hey", 0
    c: times 1024 db 0

    fmt: db "This is a string: %s", 0x0a, 0x00

    counter: db 0x00

section .text
extern puts
extern printf
extern strcat

global main
main:
    mov byte [counter], 3
    loop:
    call print_stuff
    sub byte [counter], 1
    cmp byte [counter], 0
    jne loop
    jmp exit

print_stuff:

    push rbp
    mov rbp, rsp
    sub rsp, 32

    ; zero out the first byte so strcat works
    ; across multiple invocations of this function
    mov byte [rsp], 0

    ; call puts by passing a string pointer in rdi
    mov rdi, msg
    call puts

    ; STRCAT ON THE STACK
    ; call strcat with a dst in rdi and a src in rsi,
    ; storing the result in a preallocated buffer pointed to by rsp.
    mov rdi, rsp
    mov rsi, a
    call strcat

    ; call strcat again, concatenating the result to the rsp buffer.
    mov rdi, rsp
    mov rsi, b
    call strcat

    ; move the string pointer rsp into rdi and call puts.
    mov rdi, rsp
    call puts

    ; STRCAT IN THE .DATA SECTION
    ; call strcat with a dst in rdi and a src in rsi,
    ; storing the result in a preallocated buffer in c.
    mov rdi, c
    mov rsi, a
    call strcat

    ; call strcat again, concatenating the result to the c buffer.
    mov rdi, c
    mov rsi, b
    call strcat

    ; move the string pointer c into rdi and call puts.
    mov rdi, rsp
    call puts

    ; PRINTF
    mov rdi, fmt
    mov rsi, msg
    mov al, 0 ; indicates no SSE (streaming SIMD) registers used
    call printf

    add rsp, 32
    pop rbp

    ret

exit:
    mov     rax, sys_exit
    mov     rdi, EXIT_SUCCESS
    syscall

