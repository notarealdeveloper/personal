; By calling this a "machine code" file, rather than an assembly file (which it is)
; I mean that we manually construct the ELF header, program header, etc., so that
; we can simply call the assembler in raw binary mode, chmod +x it, and run it,
; without having to use (a) nasm's ability to assemble into ELF object files, and
; (b) without having to use the linker at all.

; Note: Most of this information is available in the manpage found by typing "man elf"

; Note: The length of the fields e_entry, e_phoff, and e_shoff is either 4 or 8 bytes (32 or 64 bits)
; depending on the value set for the 5th byte of the ELF header, namely e_ident[EI_CLASS]
; for more details, see http://en.wikipedia.org/wiki/Executable_and_Linkable_Format

; p_align: The value to which the segments are aligned in memory and in the file. 
; values 0 and 1 mean no alignment is required. Otherwise, p_align should be a positive, 
; integral power of 2, and we should have p_vaddr mod p_align == p_offset (== 0 here)

; Assemble, mark executable, and run with: 
; F=mytiny && nasm -f bin -o $F $F.asm && chmod +x $F && ./$F ; echo $?
; Note: There's no linker step! We did that shit by our damn selves!
; Even better, we use nasm with the "-f bin" flag, which means we *wrote* the assembly
; for a complete, runnable binary! Baller!

BITS 32                                     ; doesn't seem to be needed.
    ;org     0x08048000                     ; default memory address for executables to load
    org     0x00001000                      ; default memory address for executables to load
ehdr:                                       ; Elf32_Ehdr

    db      0x7F, "ELF"                     ; e_ident[EI_MAG0] through e_ident[EI_MAG3] :: Magic number
    db      1                               ; e_ident[EI_CLASS] :: 1 for 32 bit, 2 for 64 bit
    db      1                               ; e_ident[EI_DATA] :: Endianness: 1 for little, 2 for big. Affects interp starting at offset 0x10
    db      1                               ; e_ident[EI_VERSION] :: Set to 1 for the original version of ELF
    db      0                               ; e_ident[EI_OSABI] :: Target OS abi. Usually set to 0 regardless of target platform.
    db      0                               ; e_ident[EI_ABIVERSION] :: Further specifies ABI version. Linux kernel after 2.6 doesn't define it.
    db      0,0,0,0,0,0,0                   ; e_ident[EI_PAD] :: 7 bytes, currently unused

    dw      2                               ; e_type :: 1,2,3,4 specify whether the object is relocatable, executable, shared, or core
    dw      3                               ; e_machine :: Specifies target architecture. x86 is 3; x86_64 is 3E
    dd      1                               ; e_version :: Set to 1 for the original version of ELF
    dd      _start                          ; e_entry :: Mem address of where execution starts. Either 32 | 64 bits, based on e_ident[EI_CLASS]
    dd      phdr - $$                       ; e_phoff :: Points to the start of the program header table. The $$ is the top of the file
    dd      0                               ; e_shoff :: Points to the start of the section header table (not required, so set to 0 here)
    dd      0                               ; e_flags :: Interpretation of this field depends on the target architecture (set to 0 here)
    dw      ehdrsize                        ; e_ehsize :: Contains the size of this header, (usually 52 bytes for 32 bit, 64 for 64 bit) (we're 52)
    dw      phdrsize                        ; e_phentsize :: Contains the size of a program header table entry
    dw      1                               ; e_phnum :: Contains the number of entries in the program header table
    dw      0                               ; e_shentsize :: Contains the size of a section header table entry
    dw      0                               ; e_shnum :: Contains the number of entries in the section header table
    dw      0                               ; e_shstrndx :: Contains index of the section header table entry that contains the section names

    ehdrsize equ $-ehdr                     ; ehdrsize :: (here) - ehdr

phdr:                                       ; Elf32_Phdr
    dd      1                               ; p_type :: what kind of segment this array element describes. 1 is "PT_LOAD"
    dd      0                               ; p_offset :: offset from the beginning of the file at which the first byte of the segment resides.
    dd      $$                              ; p_vaddr :: the virtual address at which the first byte of the segment resides in memory.
    dd      $$                              ; p_paddr :: the segment's physical address (on systems for which physical addressing is relevant)
    dd      filesize                        ; p_filesz :: the number of bytes in the file image of the segment; it may be zero.
    dd      filesize                        ; p_memsz :: the number of bytes in the memory image of the segment; it may be zero.
    dd      5                               ; p_flags :: flags relevant to the segment (0x1 = execute, 0x2 = write, 0x4 = read)
    dd      0x1000                          ; p_align :: pad segments so that they begin at multiples of 0x1000.

    phdrsize      equ     $ - phdr          ; phdrsize :: (here) - phdr

_start:
    %define stdout      1
    %define sys_write   4

    mov     eax, sys_write
    mov     ebx, stdout
    mov     ecx, msg
    mov     edx, msg.len
    int     0x80

    mov     al, 1                           ; sys_exit kernel opcode
    mov     bl, 42                          ; exit code 42
    int     0x80                            ; poke the kernel

msg:            db      "Hey there babycakes!", 0x0A, 0x00
.len:           equ     $ - msg

filesize        equ     $ - $$
