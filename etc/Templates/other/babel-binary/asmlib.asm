;------;
; NASM ;
;------;

; syscall macros
%define sys_write       1
%define sys_open        2
%define sys_close       3
%define sys_fork        57
%define sys_exit        60
%define sys_mkdir       83
%define sys_rmdir       84
%define sys_creat       85
%define sys_unlink      87


; variable macros
%define stdin           0
%define stdout          1
%define stderr          2
%define byte_at(reg) byte [reg]

DEFAULT REL

section .data
message: db "asm_function", 0x3a, " got ?", 0x0a, 0x00
message_len equ $-message

section .text
global asm_function:function

asm_function:

    push rax
    push rdx
    push rsi
    push rdi
    push rbx

    ; input is received in register rdi
    ; 1. move it to rbx
    ; 2. add 0x30 to rbx as a hacky version of atoi
    ; 3. use repne scasb to scan string in rdi for char in rax
    ; 4. repne scasb seems to overshoot by one, so subtract one from
    ;    rdi to overwrite the '?' in message with the value we received
    ;    from the c code that called us.
    mov rbx, rdi
    add rbx, 0x30
    mov rcx, message_len
    mov rdi, message
    mov rax, '?'
    repne scasb
    mov byte [rdi-1], bl

    ; 5. use the sys_write syscall to print the message to the screen
    mov     rdx, message_len    ; write string length
    mov     rsi, message        ; where to start writing
    mov     rdi, stdout         ; file descriptor
    mov     rax, sys_write      ; sys_write kernel opcode in x86_64
    syscall

    pop rbx
    pop rdi
    pop rsi
    pop rdx
    pop rax

    ; multiply input by 2
    imul rdi, 2
    mov rax, rdi

    ; c calling convention returns values in the rax register
    ret

