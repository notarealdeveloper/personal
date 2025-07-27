#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <pthread.h>

/* Software Transactional Memory
 * =============================
 * gcc -o stm c-stm.c -fgnu-tm -pthread
 *
 * Running this under strace with and without the transaction blocks
 * shows that gcc is implementing STM using the futex system call.
 * See futex(2) for more details.
 */

#define STACK_SIZE (1024 * 1024)    /* Stack size for cloned children */
#define err_exit(msg) ({ perror(msg); exit(EXIT_FAILURE); })

struct vars {
    unsigned long a;
    unsigned long b;
    unsigned long read_flag;
    unsigned long write_flag;
    unsigned long read_fails;
    unsigned long write_fails;
    unsigned long read_wins;
    unsigned long write_wins;
    unsigned long tripwire;

};

struct vars v = {
    .a = 0,
    .b = 0,
    .read_flag = 0,
    .write_flag = 0,
    .read_fails = 0,
    .write_fails = 0,
    .read_wins = 0,
    .write_wins = 0,
    .tripwire = 0,
};

unsigned long count_to = 1000000;


static void *transact_thread(void *arg)
{
    struct vars *p = (struct vars *)arg;
    unsigned long i;

    for (i=0; i<count_to; i++) {

        // begin transaction retry block

        retry:

        p->write_flag = 1;

        __transaction_atomic {
            p->write_flag = 0;
            p->a += 1;
            p->b += 1;
            if (p->tripwire > 0) {
                // writer fails if the reader read different values.
                // in that case, the reader sets a tripwire to punish the writer.
                __transaction_cancel;
                // p->write_flag = 1;
            }
        }

        if (p->write_flag) {
            p->tripwire -= 1;
            p->write_fails += 1;
            //printf("cancelled write: decrementing tripwire and retrying\n");
            goto retry;
        }
        // end transaction retry block

        p->write_wins += 1;
    }

    return NULL;
}


int main() {

    int ret;
    pid_t pid;
    unsigned long i;
    unsigned long a=0, b=0;
    pthread_t thread1;
    struct vars *p = &v;
    double read_ratio, write_ratio;

    ret = pthread_create(&thread1, NULL, transact_thread, (void *) p);

    while (p->a < count_to) {

        // begin transaction retry block

        retry:

        p->read_flag = 1;

        __transaction_atomic {
            p->read_flag = 0;
            a = p->a;
            b = p->b;
            if (a != b) {
                __transaction_cancel;
                // p->read_flag = 1;
            }
        }

        if (p->read_flag) {
            p->read_fails += 1;
            p->tripwire += 1;
            //printf("cancelled read: incrementing tripwire and retrying\n");
            goto retry;
        }
        // end transaction retry block

        p->read_wins += 1;

        i++;
    }
    pthread_join(thread1, NULL);
    printf("child exited, parent now exiting.\n");

    read_ratio = ((double) p->read_wins) / (p->read_wins + p->read_fails);
    write_ratio = ((double) p->write_wins) / (p->write_wins + p->write_fails);

    printf("read_wins=%lu, read_fails=%lu, read_ratio=%f\n"
           "write_wins=%lu, write_fails=%lu, write_ratio=%f\n"
            "a=%lu, b=%lu\n",
            p->read_wins, p->read_fails, read_ratio,
            p->write_wins, p->write_fails, write_ratio,
            p->a, p->b
    );
    return 0;
}
