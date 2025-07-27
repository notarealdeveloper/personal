#define _GNU_SOURCE
#include <sys/wait.h>
#include <sys/utsname.h>
#include <sched.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>

#define STACK_SIZE (1024 * 1024)    /* Stack size for cloned child */

#define err_exit(msg) ({ perror(msg); exit(EXIT_FAILURE); })

unsigned long var = 0;
unsigned long count_to = (unsigned long)1000000000;
unsigned long print_every = (unsigned long)10000000;

static int child_func(void *arg)
{
    printf("In child_func, pid is %d\n", getpid());

    unsigned long *ptr = arg;
    unsigned long i;
    for (i=0; i<=count_to; i++) {
        if (i % print_every == 0) {
            printf("child_func:  &var = %p, var = %lu\n", ptr, *ptr);
        }
        *ptr += 1;
    }

    /* Keep the namespace open for a while, by sleeping.
       This allows some experimentation -- for example,
       another process might join the namespace. */

    /* Child terminates now */
    return 0;
}

int main(int argc, char *argv[])
{
    char *stack;        /* Start of stack buffer */
    char *stack_top;    /* End of stack buffer */
    pid_t pid;
    pid_t pid2;

    if (argc > 1) {
        count_to = atol(argv[1]);
    }

    /* Allocate memory to be used for the stack of the child. */
    int mmap_prot = PROT_READ | PROT_WRITE;
    int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK;
    stack = mmap(NULL, STACK_SIZE, mmap_prot, mmap_flags, -1, 0);

    if (stack == MAP_FAILED) {
        err_exit("mmap");
    }

    /* Assume stack grows downward */
    stack_top = stack + STACK_SIZE;

    /* Create child that has its own UTS namespace;
      child commences execution in child_func(). */
    int clone_flags;
    // clone_flags = 0;
    // clone_flags = SIGCHLD;
    clone_flags = CLONE_VM | SIGCHLD;
    // clone_flags = CLONE_VFORK | CLONE_VM | SIGCHLD;
    pid = clone(child_func, stack_top, clone_flags, &var);
    if (pid == -1) {
       err_exit("clone");
    }

    /* Parent falls through to here */
    printf("clone() returned %jd\n", (intmax_t) pid);

    unsigned long i;
    for (i=0; i<=count_to; i++) {
        if (i % print_every == 0) {
            printf("parent_func: &var = %p, var = %lu\n", &var, var);
        }
        var += 1;
    }

    /* Wait for child */
    if (waitpid(pid, NULL, 0) == -1) {
        err_exit("waitpid");
    }
    printf("child has terminated\n");

    exit(EXIT_SUCCESS);
}

