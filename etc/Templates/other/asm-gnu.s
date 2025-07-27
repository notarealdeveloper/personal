/* Declare an 8-bit variable */
mychar:
.byte  0x01


/* Declare a 16-bit variable, aligned to a 2-byte (2^1) boundary */
.align 1
myshort:
.hword 0x0123


/* Declare an integer variable, aligned to a 4-byte (2^2) boundary */
.align 2
myint:
.long  0x01234567


/* Declare a string */
dirname:
.string "/home/jason/Desktop/boop"


mystr:
.string "Hey babycakes\n\0"
eostr:


# Declare "array_initialized" with 10 32-bit words
.align 2
array:
.word 5,3,6,7,2,8,7,1,9,5


# Declare an uninitialized array (buffer) with space for 10 32-bit ints
buf:
.skip 40


.globl main
main:
    # Syscalls in arch/x86/include/generated/uapi/asm/unistd_64.h

    # sys_mkdir
    movq    $83,        %rax    # sys_mkdir on x86_64
    movq    $dirname,   %rdi    # directory name
    movq    $0777,      %rsi    # mode: this is 0o777
    syscall

    # sys_write
    movq    $1,         %rax    # sys_write on x86_64
    movq    $1,         %rdi    # stdout fd
    movq    $mystr,     %rsi    # char *buf
    movq    $eostr,     %rdx    # number of bytes to write 
    subq    $mystr,     %rdx    # = $eostr - $mystr
    syscall

    # sys_exit
    movq    $60,        %rax    # sys_exit on x86_64
    movq    $0,         %rdi    # exit status
    syscall
